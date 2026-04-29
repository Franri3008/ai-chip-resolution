import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"
INGEST = SCRIPTS / "ingest"
LLM = SCRIPTS / "llm"
CLASSIFIERS = SCRIPTS / "classifiers"
VIZ = SCRIPTS / "viz"

sys.path.insert(0, str(CLASSIFIERS))
sys.path.insert(0, str(LLM))
sys.path.insert(0, str(VIZ))
from signals import (  # noqa: E402
    apply_training_disclosure_cap,
    has_explicit_training_chip_evidence,
    launcher_implied_chip,
    snippet_is_training_disclosure,
    TRAINING_DISCLOSURE_CAP,
)
from llm_client import (  # noqa: E402
    llm_enabled,
    validate_provider,
    LLMDisabled,
    LLMUnavailable,
    VALID_PROVIDERS,
)

TRACKER = None  # Set by main() when --viz is passed

LLM_CHIP_CONFIDENCE_THRESHOLD = 0.5
LLM_CHIP_CONFLICT_THRESHOLD = 0.7  # Skip conflict-check LLM when chip confidence is already high

FRAMEWORK_CHIP_MAP = {
    "pytorch": "nvidia",
    "tensorflow": "nvidia",
    "jax": "google_tpu",
    "paddlepaddle": "nvidia",
    "mxnet": "nvidia",
    "mindspore": "huawei_ascend",
}

# Broad upstream repos often expose framework/runtime support, but not the
# training hardware for a specific checkpoint.
LOW_TRUST_GITHUB_CHIP_REPOS = {
    "facebookresearch/convnext-v2",
    "flagopen/flagembedding",
    "huggingface/pytorch-image-models",
    "huggingface/transformers",
    "stanford-futuredata/colbert",
    "ukplab/sentence-transformers",
    "ultralytics/ultralytics",
}

_RUNTIME_CHIP_NOISE_RE = re.compile(
    r'(?:'
    r'inference|inference\s+computations|throughput|latency|benchmark|self-hosted|speed|runtime|'
    r'deploy|deployment|serving|device_map|supported|compatible|works?\s+with|install|cpuonly|'
    r'cuda\s+gpu\s+machine|TensorRT|OpenVINO|MLX|vLLM|SGLang|Ollama|LM\s?Studio|'
    r'pipeline\.to\(|GPU\s+memory|requirements\s+on\s+GPU\s+memory|device\s+cache|empty_cache|'
    r'evaluation\b|per\s+query|re-rank|validation\s+set|top-1000|langchain|'
    r'HuggingFaceBgeEmbeddings|engine="torch"|onnx\s+inference|processing\s+on\s+gpu|'
    r'run\s+on\s+cpu|send\s+.*\s+to\s+gpu|vram|triton\s+cache'
    r')',
    re.IGNORECASE,
)

# Ground truth provider names → pipeline chip_provider names
_GT_PROVIDER_MAP = {
    "nvidia": "nvidia",
    "google": "google_tpu",
    "apple": "apple",
    "amd": "amd",
    "intel": "intel",
    "aws": "aws",
    "qualcomm": "qualcomm",
    "huawei": "huawei_ascend",
    "huawei_ascend": "huawei_ascend",
    "ascend": "huawei_ascend",
    "cambricon": "cambricon",
    "baidu": "baidu_kunlun",
    "baidu_kunlun": "baidu_kunlun",
    "kunlun": "baidu_kunlun",
    "kunlunxin": "baidu_kunlun",
    "moore_threads": "moore_threads",
    "moorethreads": "moore_threads",
    "mthreads": "moore_threads",
    "musa": "moore_threads",
    "iluvatar": "iluvatar",
    "corex": "iluvatar",
    "hygon": "hygon",
    "dcu": "hygon",
    "metax": "metax",
    "muxi": "metax",
    "unknown": "unknown",
}


def check_tokens():
    """Verify required API tokens. HF and GH always required; LLM creds only when --llm.

    Exits on failure.
    """
    keys = Path(__file__).parent / "keys"
    errors = []

    checks = [
        (".hf_token", "https://huggingface.co/api/whoami-v2", {"Authorization": "Bearer {tok}"}, "HuggingFace"),
        (".gh_token", "https://api.github.com/rate_limit", {"Authorization": "Bearer {tok}", "User-Agent": "model-classifier"}, "GitHub"),
    ]

    for filename, url, headers_tmpl, name in checks:
        path = keys / filename
        if not path.exists():
            errors.append(f"{name} token missing: keys/{filename}")
            continue
        tok = path.read_text().strip()
        if not tok:
            errors.append(f"{name} token is empty: keys/{filename}")
            continue
        headers = {k: v.replace("{tok}", tok) for k, v in headers_tmpl.items()}
        req = urllib.request.Request(url, headers=headers)
        try:
            urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as e:
            errors.append(f"{name} token invalid (HTTP {e.code}): keys/{filename}")
        except Exception as e:
            errors.append(f"{name} token check failed ({e}): keys/{filename}")

    if llm_enabled():
        try:
            validate_provider()
        except LLMUnavailable as e:
            errors.append(str(e))

    if errors:
        print("\nToken validation failed:")
        for err in errors:
            print(f"  - {err}")
        print("\nFix these issues before running the pipeline.")
        sys.exit(1)

    print("All tokens valid.")


def load_ground_truth():
    """Load ground-truth labels from tests/ground_truth.csv and any
    tests/ground_truth_*.csv slice files.

    Returns {model_id: normalized_provider}. Slice files (e.g.
    ground_truth_chinese.csv) let us track per-cohort accuracy without
    bloating the main CSV.
    """
    tests_dir = Path(__file__).parent / "tests"
    if not tests_dir.exists():
        return {}
    gt_files = []
    main_csv = tests_dir / "ground_truth.csv"
    if main_csv.exists():
        gt_files.append(main_csv)
    gt_files.extend(sorted(p for p in tests_dir.glob("ground_truth_*.csv")))

    gt = {}
    for path in gt_files:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_id = row["id"].strip()
                provider = row["provider"].strip().lower()
                gt[model_id] = _GT_PROVIDER_MAP.get(provider, provider)
    return gt


def evaluate_ground_truth(results, ground_truth):
    """Add 'correct' field to each result and print accuracy decomposition."""
    correct = 0
    incorrect = 0
    not_in_gt = 0
    mismatches = []

    for r in results:
        model_id = r["id"]
        conclusion = r["conclusion"]
        predicted = conclusion["chip_provider"]

        if model_id not in ground_truth:
            conclusion["correct"] = -1
            not_in_gt += 1
            continue

        expected = ground_truth[model_id]

        conclusion["expected_provider"] = expected
        if predicted == expected:
            conclusion["correct"] = 1
            correct += 1
        else:
            conclusion["correct"] = 0
            incorrect += 1
            mismatches.append((model_id, expected, predicted, conclusion.get("chip_provider_source")))

    evaluated = correct + incorrect

    print(f"\n{'='*60}")
    print(f"Ground Truth Evaluation")
    print(f"{'='*60}")
    print(f"  Models in ground truth:     {evaluated}")
    print(f"  Not in ground truth:        {not_in_gt}")
    if evaluated > 0:
        pct = correct / evaluated * 100
        print(f"  Correct:                    {correct}/{evaluated} ({pct:.1f}%)")
        print(f"  Incorrect:                  {incorrect}/{evaluated} ({100-pct:.1f}%)")
    else:
        print(f"  Correct:                    0/0 (N/A)")

    if mismatches:
        print(f"\n  Mismatches:")
        for model_id, expected, predicted, src in mismatches:
            print(f"    {model_id:48s}  expected={expected:12s}  got={predicted:12s}  (via {src or '?'})")


def run(script, cwd=None, extra_args=None, viz_stage=None, viz_note=""):
    cmd = [sys.executable, str(script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"Running: {script.name}")
    print(f"{'='*60}")
    if TRACKER and viz_stage:
        TRACKER.start_stage(viz_stage, note=viz_note)
    result = subprocess.run(cmd, cwd=str(cwd or script.parent))
    if result.returncode != 0:
        if TRACKER and viz_stage:
            TRACKER.finish_stage(viz_stage, note=f"exit {result.returncode}", errored=True)
        print(f"\nFailed: {script.name} (exit {result.returncode})")
        sys.exit(result.returncode)
    if TRACKER and viz_stage:
        TRACKER.finish_stage(viz_stage)


def run_parallel(scripts, extra_args=None, viz_stage=None, viz_note=""):
    """Launch multiple scripts in parallel and wait for all to finish.

    Output from each script is buffered and printed as a block when the
    script finishes, so lines from different scripts don't interleave.
    Exits with the first non-zero return code if any script fails.
    """
    extra_args = extra_args or []
    results = {}  # script → (returncode, output)
    lock = threading.Lock()

    if TRACKER and viz_stage:
        TRACKER.start_stage(viz_stage, note=viz_note)

    def _run_one(script):
        cmd = [sys.executable, str(script)] + extra_args
        proc = subprocess.run(cmd, capture_output=True, text=True)
        with lock:
            results[script] = proc
            if TRACKER and viz_stage:
                done = len(results)
                TRACKER.update_stage(
                    viz_stage,
                    completed=done,
                    current=script.name,
                )

    threads = [threading.Thread(target=_run_one, args=(s,)) for s in scripts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = None
    for script in scripts:
        proc = results[script]
        print(f"\n{'='*60}")
        print(f"Output: {script.name}")
        print(f"{'='*60}")
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        if proc.returncode != 0 and failed is None:
            failed = (script.name, proc.returncode)

    if failed:
        if TRACKER and viz_stage:
            TRACKER.finish_stage(viz_stage, note=f"{failed[0]} exit {failed[1]}", errored=True)
        print(f"\nFailed: {failed[0]} (exit {failed[1]})")
        sys.exit(failed[1])
    if TRACKER and viz_stage:
        TRACKER.finish_stage(viz_stage)


def _normalize_github_repo(url):
    if not url:
        return ""
    match = re.search(r'github\.com/([^/]+)/([^/]+)', url, re.IGNORECASE)
    if not match:
        return ""
    return f"{match.group(1)}/{match.group(2)}".lower()
def _is_docs_only_detection_files(files):
    if not files:
        return False
    lowered = [f.lower() for f in files]
    return all(
        "readme" in path
        or any(name in path for name in ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "makefile"))
        for path in lowered
    )


_has_explicit_training_chip_evidence = has_explicit_training_chip_evidence


def _is_runtime_only_chip_evidence(snippets):
    if not snippets or has_explicit_training_chip_evidence(snippets):
        return False
    return all(_RUNTIME_CHIP_NOISE_RE.search(snippet.get("snippet", "")) for snippet in snippets)


def resolve_initial_conclusion(mc, mca, gha, axa):
    """Resolve chip/framework before optional LLM fallback."""
    mc_chip = mca.get("chip_provider", "unknown")
    gh_chip = gha.get("chip_provider", "unknown")
    ax_chip = axa.get("chip_provider", "unknown")

    mc_chip_conf = round(mca.get("chip_provider_confidence", 0.0), 2)
    gh_chip_conf = round(gha.get("chip_provider_confidence", 0.0), 2)
    ax_chip_conf = round(axa.get("chip_provider_confidence", 0.0), 2)
    mc_fw_conf = round(mca.get("framework_confidence", 0.0), 2)
    gh_fw_conf = round(gha.get("framework_confidence", 0.0), 2)
    ax_fw_conf = round(axa.get("framework_confidence", 0.0), 2)

    quality_blocked_chip = False

    if (
        mc_chip != "unknown"
        and mc_chip_conf < 0.5
        and "training" not in mca.get("matched_sections", [])
        and not _has_explicit_training_chip_evidence(mca.get("chip_snippets"))
        and _is_runtime_only_chip_evidence(mca.get("chip_snippets"))
    ):
        mc_chip_conf = 0.0
        quality_blocked_chip = True

    selected_repo = _normalize_github_repo(mc.get("main_github"))
    if gh_chip != "unknown":
        if selected_repo in LOW_TRUST_GITHUB_CHIP_REPOS and not _has_explicit_training_chip_evidence(gha.get("chip_snippets")):
            gh_chip_conf = 0.0
            quality_blocked_chip = True
        elif (
            _is_docs_only_detection_files(gha.get("detection_files", []))
            and gh_chip_conf >= 0.5
            and _is_runtime_only_chip_evidence(gha.get("chip_snippets"))
        ):
            gh_chip_conf = 0.0
            quality_blocked_chip = True

    if (
        ax_chip != "unknown"
        and not _has_explicit_training_chip_evidence(axa.get("chip_snippets"))
        and _is_runtime_only_chip_evidence(axa.get("chip_snippets"))
    ):
        ax_chip_conf = 0.0
        quality_blocked_chip = True

    # Derivative models: linked arXiv paper describes the BASE, not this checkpoint.
    if mca.get("is_derivative", False) and ax_chip != "unknown" and ax_chip_conf > 0:
        ax_chip_conf = 0.0
        quality_blocked_chip = True

    # Training-disclosure cap: snippets without explicit training language cap at 0.6.
    if mc_chip != "unknown" and mc_chip_conf > 0:
        mc_chip_conf = round(apply_training_disclosure_cap(mc_chip_conf, mca.get("chip_snippets")), 2)
    if gh_chip != "unknown" and gh_chip_conf > 0:
        gh_chip_conf = round(apply_training_disclosure_cap(gh_chip_conf, gha.get("chip_snippets")), 2)
    if ax_chip != "unknown" and ax_chip_conf > 0:
        ax_chip_conf = round(apply_training_disclosure_cap(ax_chip_conf, axa.get("chip_snippets")), 2)

    ax_chip_source = "arxiv_paper"

    MIN_PREFER_GITHUB = 0.5
    MIN_PREFER_ARXIV = 0.5
    MIN_MARGIN = 0.15
    CONFLICT_CONF_CAP = 0.55
    CONFLICT_DISAGREE_MIN = 0.3

    if (
        ax_chip != "unknown"
        and ax_chip_conf >= MIN_PREFER_ARXIV
        and ax_chip_conf - max(gh_chip_conf, mc_chip_conf) >= MIN_MARGIN
    ):
        chip, chip_conf, chip_src = ax_chip, ax_chip_conf, ax_chip_source
    elif (
        gh_chip != "unknown"
        and gh_chip_conf >= MIN_PREFER_GITHUB
        and gh_chip_conf - mc_chip_conf >= MIN_MARGIN
    ):
        chip, chip_conf, chip_src = gh_chip, gh_chip_conf, "github_code"
    elif mc_chip != "unknown" and mc_chip_conf > 0:
        chip, chip_conf, chip_src = mc_chip, mc_chip_conf, "modelcard"
    elif gh_chip != "unknown" and gh_chip_conf > 0:
        chip, chip_conf, chip_src = gh_chip, gh_chip_conf, "github_code"
    elif ax_chip != "unknown" and ax_chip_conf > 0:
        chip, chip_conf, chip_src = ax_chip, ax_chip_conf, ax_chip_source
    else:
        chip, chip_conf, chip_src = "unknown", 0.0, None

    chip_conflict = False
    if chip != "unknown":
        for other_chip, other_conf in (
            (mc_chip, mc_chip_conf),
            (gh_chip, gh_chip_conf),
            (ax_chip, ax_chip_conf),
        ):
            if (
                other_chip != "unknown"
                and other_chip != chip
                and other_conf >= CONFLICT_DISAGREE_MIN
            ):
                chip_conflict = True
                break
        if chip_conflict:
            chip_conf = min(chip_conf, CONFLICT_CONF_CAP)

    mc_fw = mca.get("framework", "unknown")
    gh_fw = gha.get("framework", "unknown")
    ax_fw = axa.get("framework", "unknown")

    if (
        ax_fw != "unknown"
        and ax_fw_conf >= MIN_PREFER_ARXIV
        and ax_fw_conf - max(gh_fw_conf, mc_fw_conf) >= MIN_MARGIN
    ):
        fw, fw_conf, fw_src = ax_fw, ax_fw_conf, "arxiv_paper"
    elif (
        gh_fw != "unknown"
        and gh_fw_conf >= MIN_PREFER_GITHUB
        and gh_fw_conf - mc_fw_conf >= MIN_MARGIN
    ):
        fw, fw_conf, fw_src = gh_fw, gh_fw_conf, "github_code"
    elif mc_fw != "unknown":
        fw, fw_conf, fw_src = mc_fw, mc_fw_conf, "modelcard"
    elif gh_fw != "unknown":
        fw, fw_conf, fw_src = gh_fw, gh_fw_conf, "github_code"
    elif ax_fw != "unknown":
        fw, fw_conf, fw_src = ax_fw, ax_fw_conf, "arxiv_paper"
    else:
        fw, fw_conf, fw_src = "unknown", 0.0, None

    fw_conflict = False
    if fw != "unknown":
        for other_fw, other_conf in (
            (mc_fw, mc_fw_conf),
            (gh_fw, gh_fw_conf),
            (ax_fw, ax_fw_conf),
        ):
            if (
                other_fw != "unknown"
                and other_fw != fw
                and other_conf >= CONFLICT_DISAGREE_MIN
            ):
                fw_conflict = True
                break
        if fw_conflict:
            fw_conf = min(fw_conf, CONFLICT_CONF_CAP)

    return {
        "chip_provider": chip,
        "chip_provider_confidence": chip_conf,
        "chip_provider_source": chip_src,
        "framework": fw,
        "framework_confidence": fw_conf,
        "framework_source": fw_src,
        "quality_blocked_chip": quality_blocked_chip,
        "source_conflict": chip_conflict or fw_conflict,
    }


def main():
    global TRACKER

    parser = argparse.ArgumentParser(description="Model hardware classifier pipeline")
    parser.add_argument("--top", type=int, default=None,
                        help="When --years/--quarters is set: top N models per BUCKET "
                             "(month or quarter) within the range "
                             "(e.g. --top 50 --quarters 2022-2026 → up to 50 × 20 = 1000 models). "
                             "Otherwise: total cap.")
    parser.add_argument("--years", type=str, default=None,
                        help="Filter by model creation year(s) with monthly buckets: "
                             "2023 | 2022,2023 | 2022-2024")
    parser.add_argument("--quarters", type=str, default=None,
                        help="Like --years but with quarterly buckets (4 per year). "
                             "Example: --quarters 2022-2026 --top 50")
    parser.add_argument("--source-csv", type=str, default=None,
                        help="Override source CSV (default: database/models.csv). "
                             "Use database/models_dedup.csv to sample from family-deduplicated rows.")
    parser.add_argument("--update-models", action="store_true", default=False,
                        help="Re-fetch models.csv from HuggingFace (default: use existing)")
    parser.add_argument("--ids-file", type=str, default=None,
                        help="Newline-separated file of model_ids to evaluate exactly. "
                             "Skips top-N popularity sampling and the get_models.py step. "
                             "Useful for focused eval against a curated ground-truth slice.")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel workers per classifier script (default: 16). "
                             "Most stages are network-bound (HF API, ar5iv HTML fetch, "
                             "GitHub repo download) and scale well with concurrency.")
    parser.add_argument("--llm-concurrency", type=int, default=64,
                        help="Max concurrent in-flight LLM requests (default: 64). "
                             "Decoupled from --workers so the vLLM server can batch "
                             "without being throttled by the classifier thread pool. "
                             "Tune against the vLLM server's --max-num-seqs.")
    parser.add_argument("--llm", action="store_true", default=False,
                        help="Enable LLM fallback for chip classification and candidate selection (default: disabled)")
    parser.add_argument("--provider", choices=list(VALID_PROVIDERS), default="OPENAI",
                        help="LLM provider when --llm is set: OPENAI (gpt-4o-mini, default), "
                             "LOCAL (vLLM at localhost:8000 serving gemma-4-e2b-it), or OPENROUTER")
    parser.add_argument("--viz", action="store_true", default=False,
                        help="Open a live pipeline dashboard in the browser during the run.")
    args = parser.parse_args()

    if args.llm:
        os.environ["LLM_ENABLED"] = "1"
        os.environ["LLM_PROVIDER"] = args.provider

    check_tokens()

    if args.viz:
        from viz_server import broadcast, start_viz_server
        from status_tracker import StatusTracker
        started = start_viz_server()
        ui_live = Path(__file__).parent / "ui" / "live"
        run_id = time.strftime("%Y%m%d-%H%M%S")
        parts = []
        if args.years: parts.append(f"years={args.years}")
        if args.top: parts.append(f"top={args.top}")
        if args.llm: parts.append(f"llm={args.provider}")
        parts.append(f"workers={args.workers}")
        TRACKER = StatusTracker(
            ui_live, run_id,
            total_models=0,
            broadcast_fn=(broadcast if started else None),
            args_summary=" ".join(parts),
        )

    ingest_args = []
    if args.ids_file:
        # Subprocesses run with cwd=scripts/ingest, so resolve to absolute first.
        ids_abs = str(Path(args.ids_file).resolve())
        if not Path(ids_abs).exists():
            print(f"--ids-file not found: {args.ids_file}")
            sys.exit(1)
        ingest_args += ["--ids-file", ids_abs]
    else:
        if args.top:
            ingest_args += ["--top", str(args.top)]
        if args.years:
            ingest_args += ["--years", args.years]
        if args.quarters:
            ingest_args += ["--quarters", args.quarters]
        if args.source_csv:
            csv_abs = str(Path(args.source_csv).resolve())
            if not Path(csv_abs).exists():
                print(f"--source-csv not found: {args.source_csv}")
                sys.exit(1)
            ingest_args += ["--source-csv", csv_abs]
    modelcard_args = ingest_args + ["--workers", str(args.workers)]
    worker_args = ["--workers", str(args.workers)]
    llm_concurrency_args = ["--llm-concurrency", str(args.llm_concurrency)]

    if args.ids_file:
        # Explicit-ID mode: skip the bulk models.csv fetch entirely.
        if TRACKER:
            TRACKER.start_stage("fetch_models", note=f"explicit ids: {args.ids_file}")
            TRACKER.finish_stage("fetch_models", note="explicit ids")
    elif args.update_models:
        run(INGEST / "get_models.py", viz_stage="fetch_models",
            viz_note="fetching from HuggingFace")
    elif TRACKER:
        TRACKER.start_stage("fetch_models", note="using cached models.csv")
        TRACKER.finish_stage("fetch_models", note="cached")

    run(INGEST / "get_modelcard.py", extra_args=modelcard_args,
        viz_stage="modelcards", viz_note="downloading model cards")
    _update_total_models()

    run(INGEST / "get_github.py",
        viz_stage="github_urls", viz_note="extracting GitHub URLs")
    run(INGEST / "get_arxiv.py",
        viz_stage="arxiv_urls", viz_note="extracting arXiv URLs")
    run(CLASSIFIERS / "evaluate_github.py",
        extra_args=worker_args + llm_concurrency_args,
        viz_stage="eval_github", viz_note="LLM eval on GitHub")
    run(CLASSIFIERS / "evaluate_arxiv.py",
        extra_args=worker_args + llm_concurrency_args,
        viz_stage="eval_arxiv", viz_note="LLM eval on arXiv")
    run_parallel(
        [
            CLASSIFIERS / "from_modelcard.py",
            CLASSIFIERS / "from_githubcode.py",
            CLASSIFIERS / "from_arxiv.py",
        ],
        extra_args=worker_args,
        viz_stage="classify",
        viz_note="heuristic classifiers in parallel",
    )

    build_results(llm_concurrency=args.llm_concurrency)

    if TRACKER:
        TRACKER.finish()


def _update_total_models():
    """After modelcards land, set the progress denominator."""
    if not TRACKER:
        return
    mc_path = Path(__file__).parent / "database" / "modelcards.json"
    try:
        with open(mc_path, encoding="utf-8") as f:
            data = json.load(f)
        TRACKER.set_total(len(data))
        TRACKER.update_stage("modelcards", completed=len(data))
    except Exception:
        pass


def build_results(llm_concurrency=32):
    import asyncio

    db = Path(__file__).parent / "database"

    sys.path.insert(0, str(LLM))
    if llm_enabled():
        import llm_client as _llm_client
        _llm_client.set_concurrency(llm_concurrency)
        from ask_llm_chip import ask_llm_chip
    else:
        ask_llm_chip = None

    if TRACKER:
        TRACKER.start_stage("resolve", note="aggregating signals")

    with open(db / "modelcards.json", encoding="utf-8") as f:
        modelcards = {m["id"]: m for m in json.load(f)}
    with open(db / "modelcard_chip_analysis.json", encoding="utf-8") as f:
        mc_analysis = {m["id"]: m for m in json.load(f)}
    with open(db / "github_chip_analysis.json", encoding="utf-8") as f:
        gh_analysis = {m["id"]: m for m in json.load(f)}

    ax_analysis_path = db / "arxiv_chip_analysis.json"
    if ax_analysis_path.exists():
        with open(ax_analysis_path, encoding="utf-8") as f:
            ax_analysis = {m["id"]: m for m in json.load(f)}
    else:
        ax_analysis = {}

    results = []
    llm_queue = []  # (result_index, model_id, kwargs_for_ask_llm_chip)
    queued_indices: set[int] = set()

    total_models = len(modelcards)
    for _i, (model_id, mc) in enumerate(modelcards.items()):
        mca = mc_analysis.get(model_id, {})
        gha = gh_analysis.get(model_id, {})
        axa = ax_analysis.get(model_id, {})

        heuristic_conf = round(mc.get("main_github_confidence", 0.0), 2)
        source = mc.get("main_github_source", "heuristic") if mc.get("main_github") else None
        github_resolution = {
            "candidate_links": mc.get("github_links", []),
            "selected_link": mc.get("main_github"),
            "heuristic_confidence": heuristic_conf,
            "source": source,
            "llm_answer": mc.get("main_github") if source == "llm" else None,
        }

        ax_heuristic_conf = round(mc.get("main_arxiv_confidence", 0.0), 2)
        ax_source = mc.get("main_arxiv_source", "heuristic") if mc.get("main_arxiv") else None
        arxiv_resolution = {
            "candidate_links": mc.get("arxiv_links", []),
            "selected_link": mc.get("main_arxiv"),
            "heuristic_confidence": ax_heuristic_conf,
            "source": ax_source,
            "llm_answer": mc.get("main_arxiv") if ax_source == "llm" else None,
        }

        mc_chip_conf = round(mca.get("chip_provider_confidence", 0.0), 2)
        gh_chip_conf = round(gha.get("chip_provider_confidence", 0.0), 2)
        ax_chip_conf = round(axa.get("chip_provider_confidence", 0.0), 2)
        mc_fw_conf = round(mca.get("framework_confidence", 0.0), 2)
        gh_fw_conf = round(gha.get("framework_confidence", 0.0), 2)
        ax_fw_conf = round(axa.get("framework_confidence", 0.0), 2)

        resolved = resolve_initial_conclusion(mc, mca, gha, axa)
        chip = resolved["chip_provider"]
        chip_conf = resolved["chip_provider_confidence"]
        chip_src = resolved["chip_provider_source"]
        fw = resolved["framework"]
        fw_conf = resolved["framework_confidence"]
        fw_src = resolved["framework_source"]
        quality_blocked_chip = resolved["quality_blocked_chip"]

        is_derivative = mca.get("is_derivative", False)
        base_model_id = mca.get("base_model")
        runtime_library = mca.get("runtime_library")

        gh_launcher_chip = launcher_implied_chip(gha.get("training_snippets") or [])

        needs_llm = False
        if chip == "unknown" and quality_blocked_chip:
            needs_llm = False
        elif chip_conf < LLM_CHIP_CONFIDENCE_THRESHOLD:
            needs_llm = True
        elif chip != "unknown" and abs(chip_conf - TRAINING_DISCLOSURE_CAP) < 0.01:
            # Cap fired → defer to LLM, except for chips whose distinctive tokens
            # (mindspore/cnnl/xpurt/musa/ixrt/hy-smi/mxmaca) rarely collide with
            # runtime-noise vocabulary. Small LLMs over-correct these to unknown.
            if chip in ("huawei_ascend", "cambricon", "baidu_kunlun",
                        "moore_threads", "iluvatar", "hygon", "metax"):
                needs_llm = False
            else:
                needs_llm = True
        elif chip != "unknown" and chip_conf < LLM_CHIP_CONFLICT_THRESHOLD:
            implied = FRAMEWORK_CHIP_MAP.get(fw)
            if implied and chip != implied:
                needs_llm = True

        # For derivative models with a base_model in our dataset, defer to Pass 2
        if is_derivative and needs_llm and base_model_id and base_model_id in mc_analysis:
            needs_llm = False

        # Heuristic + matching github launcher → skip LLM. Don't promote unknown
        # → launcher-chip; library repos (FlagEmbedding etc.) get correctly zeroed.
        if (gh_launcher_chip and not is_derivative
                and chip != "unknown" and chip == gh_launcher_chip):
            needs_llm = False
            chip_conf = max(chip_conf, 0.55)

        if needs_llm:
            mc_text = mc.get("modelcard", "")
            yaml_library = None
            lib_m = re.search(r'library_name:\s*(\S+)', mc_text[:2000].lower())
            if lib_m:
                yaml_library = lib_m.group(1)

            extra_context = ""
            if is_derivative:
                extra_context = (
                    f"\nIMPORTANT: This model is a DERIVATIVE (quantized/converted) version. "
                    f"Base model: {base_model_id or 'unknown'}. "
                    f"library_name '{runtime_library}' indicates an inference runtime, NOT training hardware. "
                    f"Determine the chip used to train the ORIGINAL model, not the conversion target."
                )

            # Generic library repos (transformers etc.) leak polyglot backend
            # strings — drop their snippets so the LLM doesn't commit on them.
            selected_repo_for_llm = _normalize_github_repo(mc.get("main_github"))
            if selected_repo_for_llm in LOW_TRUST_GITHUB_CHIP_REPOS:
                gh_snips = []
            else:
                gh_snips = (gha.get("chip_snippets") or []) + (gha.get("training_snippets") or [])
            ax_snips = (axa.get("chip_snippets") or []) + (axa.get("training_snippets") or [])
            llm_queue.append((len(results), model_id, {
                "model_name": model_id,
                "yaml_library": yaml_library,
                "framework": fw,
                "modelcard_excerpt": mc_text[:3000],
                "section_labeled_text": mca.get("section_labeled_text", ""),
                "extra_context": extra_context,
                "github_snippets": gh_snips,
                "arxiv_snippets": ax_snips,
            }))
            queued_indices.add(len(results))

        result_rec = {
            "id": model_id,
        }
        if mc.get("year") is not None:
            result_rec["year"] = mc.get("year")
        if mc.get("month") is not None:
            result_rec["month"] = mc.get("month")
        result_rec.update({
            "modelcard_analysis": {
                "chip_provider": mca.get("chip_provider", "unknown"),
                "chip_provider_confidence": mc_chip_conf,
                "chip_providers_all": mca.get("chip_providers_all", {}),
                "framework": mca.get("framework", "unknown"),
                "framework_confidence": mc_fw_conf,
                "frameworks_all": mca.get("frameworks_all", {}),
                "matched_sections": mca.get("matched_sections", []),
            },
            "github_resolution": github_resolution,
            "github_code_analysis": {
                "chip_provider": gha.get("chip_provider", "unknown"),
                "chip_provider_confidence": gh_chip_conf,
                "chip_providers_all": gha.get("chip_providers_all", {}),
                "framework": gha.get("framework", "unknown"),
                "framework_confidence": gh_fw_conf,
                "frameworks_all": gha.get("frameworks_all", {}),
                "detection_files": gha.get("detection_files", []),
            },
            "arxiv_resolution": arxiv_resolution,
            "arxiv_paper_analysis": {
                "chip_provider": axa.get("chip_provider", "unknown"),
                "chip_provider_confidence": ax_chip_conf,
                "chip_providers_all": axa.get("chip_providers_all", {}),
                "framework": axa.get("framework", "unknown"),
                "framework_confidence": ax_fw_conf,
                "frameworks_all": axa.get("frameworks_all", {}),
                "detection_sections": axa.get("detection_sections", []),
            },
            "conclusion": {
                "chip_provider": chip,
                "chip_provider_source": chip_src,
                "chip_provider_confidence": chip_conf,
                "framework": fw,
                "framework_source": fw_src,
                "framework_confidence": fw_conf,
            },
        })
        results.append(result_rec)

        if TRACKER:
            idx = len(results) - 1
            TRACKER.update_stage(
                "resolve",
                current=model_id,
                completed=idx + 1,
                note=f"initial pass {idx + 1}/{total_models}",
            )
            if idx not in queued_indices:
                TRACKER.record_resolution(
                    model_id, chip, chip_conf, chip_src or "unknown",
                    year=mc.get("year"),
                )

    # ── LLM chip fallback (async parallelized) ────────────────────────
    llm_chip_calls = 0
    total_llm_chip_cost = 0.0

    if not llm_enabled():
        if llm_queue:
            print(f"  LLM disabled (--llm not set). Skipping {len(llm_queue)} chip fallback call(s).")
        if TRACKER:
            for idx, model_id, _kwargs in llm_queue:
                r = results[idx]
                TRACKER.record_resolution(
                    model_id,
                    r["conclusion"]["chip_provider"],
                    r["conclusion"]["chip_provider_confidence"],
                    r["conclusion"]["chip_provider_source"] or "unknown",
                    year=r.get("year"),
                )
    else:
        async def _run_chip_llm_queue():
            async def _call_one(item):
                idx, model_id, kwargs = item
                try:
                    llm_chip, llm_conf, cost = await ask_llm_chip(**kwargs)
                    return idx, model_id, llm_chip, llm_conf, cost, None
                except Exception as e:
                    return idx, model_id, None, 0.0, 0.0, str(e)

            return await asyncio.gather(*[_call_one(item) for item in llm_queue])

        if TRACKER:
            TRACKER.update_stage("resolve", note=f"LLM fallback on {len(llm_queue)} model(s)")

        for idx, model_id, llm_chip, llm_conf, cost, err in asyncio.run(_run_chip_llm_queue()):
            total_llm_chip_cost += cost
            r = results[idx]
            final_chip = r["conclusion"]["chip_provider"]
            final_conf = r["conclusion"]["chip_provider_confidence"]
            final_src = r["conclusion"]["chip_provider_source"] or "unknown"
            # CRITICAL: re-look-up the per-model analyses here. The earlier
            # for-loop's `mc`/`mca`/`gha`/`axa` are stale by the time the LLM
            # results come back, so reading them would mix evidence from a
            # different model into our guard checks.
            mc = modelcards.get(model_id, {})
            mca = mc_analysis.get(model_id, {})
            gha = gh_analysis.get(model_id, {})
            axa = ax_analysis.get(model_id, {})

            if err:
                print(f"  LLM chip failed for {model_id}: {err}")
            elif llm_chip:
                # Trust-but-verify the LLM flip. Distinctive tokens stand alone;
                # branded tokens (Ascend/MLU/DCU/XPU) need a training verb nearby.
                _LLM_OVERRIDE_DISTINCTIVE = {
                    "huawei_ascend": (
                        r"\bmindspore\b|\bcann\b|\bhccl\b|\bdavinci\b|"
                        r"vllm[-_]ascend|\bmindformers\b|\b昇腾\b|"
                        r"\bnpu[-_]?smi\b|ASCEND_RT_VISIBLE_DEVICES"
                    ),
                    "cambricon": (
                        r"\bcambricon\b|\bcnnl\b|\bcnml\b|\bbangpy\b|"
                        r"\bMLUDevice\b|torch[-_]?mlu|\bcndrv\b"
                    ),
                    "baidu_kunlun": (
                        r"\bkunlun(?:xin)?\b|\bxpurt\b|\b昆仑(?:芯|核)?\b|"
                        r"paddle\.set_device\s*\(\s*[\"']xpu|"
                        r"XPU_VISIBLE_DEVICES|paddlepaddle[-_]xpu"
                    ),
                    "moore_threads": (
                        r"\btorch[-_]?musa\b|\bmthreads\b|\bmoore[-_ ]?threads\b|"
                        r"\bmusatoolkit\b|\bmusart\b|\bmccl\b|MUSA_VISIBLE_DEVICES|"
                        r"vllm[-_]musa|\bmtgpu\b"
                    ),
                    "iluvatar": (
                        r"\biluvatar\b|\bixrt\b|\bcorex\b|\b天数智芯\b|\bixsmi\b"
                    ),
                    "hygon": (
                        r"\bhygon\b|\b海光\b|hy[-_]?smi|hygon[-_]?dtk|"
                        r"hygon[-_]?dcu|DCU_VISIBLE_DEVICES"
                    ),
                    "metax": (
                        r"\bmetax\b|\bmuxi\b|\b沐曦\b|\bmxmaca\b|mx[-_]?smi|"
                        r"METAX_VISIBLE_DEVICES"
                    ),
                }
                _LLM_OVERRIDE_BRANDED = {
                    "huawei_ascend": r"\bascend\b|\b910[ABCDabcd]\b|\bAtlas\s*\d{3}\b",
                    "cambricon":     r"\bMLU\s*\d{3}\b",
                    "baidu_kunlun":  r"\bP800\b|\bR[23]00\b|\bXPU\b",
                    "moore_threads": r"\bMUSA\b|\bMTT\s*S\d{3,4}\b|\bS4000\b",
                    "iluvatar":      r"\bBI[-_ ]?V?1\d{2}\b|\bMR[-_ ]?V?\d{3}\b",
                    "hygon":         r"\bDCU\b|\bDTK\b|\b[KZ]100\b",
                    "metax":         r"\bC[5-6]\d{2}\b",
                }
                _TRAINING_VERB_RE = re.compile(
                    r"\btrain(?:ed|ing|s)?\b|\bpre[- ]?train(?:ed|ing)?\b|"
                    r"\bfine[- ]?tun(?:ed|ing|e)?\b|\bGPU\s+hours?\b|\bNPU\s+hours?\b|"
                    r"国产算力|完全基于国产|训练",
                    re.IGNORECASE,
                )
                allow_override = True

                # Reject LLM flips that have no training-disclosure language in
                # any source the LLM saw — guards against runtime-mention pickups.
                if llm_chip and llm_chip != "unknown" and llm_chip != final_chip:
                    card_disc = snippet_is_training_disclosure(
                        {"snippet": (mc.get("modelcard", "") or "")[:6000]}
                    )
                    gh_disc_any = any(
                        snippet_is_training_disclosure(s)
                        for s in (gha.get("chip_snippets") or [])
                    )
                    ax_disc_any = any(
                        snippet_is_training_disclosure(s)
                        for s in (axa.get("chip_snippets") or [])
                    )
                    if not (card_disc or gh_disc_any or ax_disc_any):
                        allow_override = False
                        print(f"  LLM chip override REJECTED (no training disclosure) "
                              f"for {model_id}: {llm_chip} (was {final_chip}@{final_src})")

                if allow_override and (llm_chip in _LLM_OVERRIDE_DISTINCTIVE and llm_chip != final_chip):
                    # Verify against modelcard + filtered training snippets only;
                    # raw github/arxiv text often lists multiple vendors as targets.
                    card_text = mc.get("modelcard", "") or ""
                    gh_disclosure_snippets = " ".join(
                        s.get("snippet", "")
                        for s in (gha.get("chip_snippets") or [])
                        if snippet_is_training_disclosure(s)
                    )
                    ax_disclosure_snippets = " ".join(
                        s.get("snippet", "")
                        for s in (axa.get("chip_snippets") or [])
                        if snippet_is_training_disclosure(s)
                    )
                    combined = card_text + " " + gh_disclosure_snippets + " " + ax_disclosure_snippets
                    distinctive_hit = re.search(
                        _LLM_OVERRIDE_DISTINCTIVE[llm_chip], combined, re.IGNORECASE,
                    )
                    branded_hit_with_training = False
                    if not distinctive_hit:
                        for m in re.finditer(
                            _LLM_OVERRIDE_BRANDED[llm_chip], combined, re.IGNORECASE,
                        ):
                            window = combined[max(0, m.start() - 120):m.end() + 120]
                            if _TRAINING_VERB_RE.search(window):
                                branded_hit_with_training = True
                                break
                    if not (distinctive_hit or branded_hit_with_training):
                        allow_override = False
                        print(f"  LLM chip override REJECTED for {model_id}: {llm_chip} "
                              f"has no distinctive token (or training-context branded "
                              f"mention) in modelcard / disclosed snippets "
                              f"(was {final_chip}@{final_src})")
                if allow_override:
                    r["conclusion"]["chip_provider"] = llm_chip
                    r["conclusion"]["chip_provider_confidence"] = llm_conf
                    r["conclusion"]["chip_provider_source"] = "llm_chip"
                    llm_chip_calls += 1
                    final_chip, final_conf, final_src = llm_chip, llm_conf, "llm_chip"
                    print(f"  LLM chip override for {model_id}: {llm_chip} (conf={llm_conf}, Cost: ${cost:.6f})")
            else:
                # LLM said unknown — override only weak heuristic hits. High-conf
                # github/arXiv evidence stays (the LLM only read the modelcard).
                safe_to_override = (
                    final_src == "modelcard"
                    or final_src == "unknown"
                    or final_src is None
                    or final_conf <= TRAINING_DISCLOSURE_CAP + 0.01
                )
                # Protect heuristic when modelcard already has explicit training
                # disclosure for the same chip — LLM context window often truncates it.
                mc_chip_snippets = [
                    s for s in (mca.get("chip_snippets") or [])
                    if s.get("provider") == final_chip
                ]
                modelcard_has_disclosure_for_chip = (
                    final_src == "modelcard"
                    and has_explicit_training_chip_evidence(mc_chip_snippets)
                )
                if final_chip != "unknown" and safe_to_override and not modelcard_has_disclosure_for_chip:
                    prev_chip, prev_src = final_chip, final_src
                    r["conclusion"]["chip_provider"] = "unknown"
                    r["conclusion"]["chip_provider_confidence"] = 0.0
                    r["conclusion"]["chip_provider_source"] = None
                    final_chip, final_conf, final_src = "unknown", 0.0, None
                    print(f"  LLM confirmed unknown for {model_id} (was {prev_chip}@{prev_src})")
                elif final_chip != "unknown" and modelcard_has_disclosure_for_chip:
                    print(f"  LLM said unknown for {model_id} but modelcard has explicit "
                          f"{final_chip} training disclosure — keeping heuristic answer")

            if TRACKER:
                TRACKER.record_llm_chip(model_id, final_chip, final_conf, cost)
                TRACKER.record_resolution(
                    model_id, final_chip, final_conf, final_src,
                    year=r.get("year"),
                )

    # ── Pass 2: Resolve derivative models via base_model ──────────────
    RUNTIME_CHIP_MAP = {
        "mlx": "apple",
        "coreml": "apple",
        "openvino": "intel",
        "onnx": None,
    }
    results_by_id = {r["id"]: r for r in results}

    if TRACKER:
        TRACKER.update_stage("resolve", note="Pass 2: derivatives")

    for r in results:
        mca = mc_analysis.get(r["id"], {})
        if not mca.get("is_derivative", False):
            continue

        conclusion = r["conclusion"]
        current_chip = conclusion["chip_provider"]
        runtime_library = mca.get("runtime_library")
        base_model_id = mca.get("base_model")
        prev_chip = conclusion["chip_provider"]
        prev_src = conclusion["chip_provider_source"] or "unknown"

        # Check if current chip matches the runtime (i.e., likely wrong)
        runtime_chip = RUNTIME_CHIP_MAP.get(runtime_library)
        chip_matches_runtime = (current_chip == runtime_chip) if runtime_chip else False

        # Try to resolve via base model
        if base_model_id and base_model_id in results_by_id:
            base_result = results_by_id[base_model_id]
            base_chip = base_result["conclusion"]["chip_provider"]
            base_chip_conf = base_result["conclusion"]["chip_provider_confidence"]
            if base_chip != "unknown" and base_chip_conf >= 0.3:
                conclusion["chip_provider"] = base_chip
                conclusion["chip_provider_confidence"] = min(base_chip_conf, 0.7)
                conclusion["chip_provider_source"] = "base_model"
                print(f"  Base model resolution for {r['id']}: {base_model_id} -> {base_chip}")
                if TRACKER:
                    TRACKER.override_resolution(
                        r["id"], base_chip, conclusion["chip_provider_confidence"],
                        "base_model", prev_chip=prev_chip, prev_source=prev_src,
                        year=r.get("year"),
                    )
                continue

        # No base model resolved → prefer unknown over a framework-default guess.
        # If the current prediction matches the runtime (apple for MLX, intel for OpenVINO),
        # it's almost certainly an artifact of the conversion target rather than training
        # hardware — collapse to unknown.
        if chip_matches_runtime:
            conclusion["chip_provider"] = "unknown"
            conclusion["chip_provider_confidence"] = 0.0
            conclusion["chip_provider_source"] = None
            print(f"  Derivative with unresolved base for {r['id']}: collapsing to unknown")
            if TRACKER:
                TRACKER.override_resolution(
                    r["id"], "unknown", 0.0, None,
                    prev_chip=prev_chip, prev_source=prev_src,
                    year=r.get("year"),
                )

    # ── Ground truth evaluation ────────────────────────────────────────
    ground_truth = load_ground_truth()
    if ground_truth:
        evaluate_ground_truth(results, ground_truth)

    if TRACKER:
        TRACKER.finish_stage("resolve", note=f"{len(results)} models resolved")

    out_path = db / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"{'='*60}")
    for r in results:
        c = r["conclusion"]
        gh = r["github_resolution"]
        src_tag = f" [via {gh['source']}]" if gh["source"] else ""
        print(f"  {r['id']:48s}  chip={c['chip_provider']:10s} ({c['chip_provider_source'] or '?':11s} {c['chip_provider_confidence']:.2f})  "
              f"fw={c['framework']:10s} ({c['framework_source'] or '?':11s} {c['framework_confidence']:.2f}){src_tag}")

    if total_llm_chip_cost > 0:
        print(f"\nLLM chip fallback: {llm_chip_calls} calls, total cost: ${total_llm_chip_cost:.6f}")


if __name__ == "__main__":
    main()
