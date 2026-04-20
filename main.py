import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"
INGEST = SCRIPTS / "ingest"
LLM = SCRIPTS / "llm"
CLASSIFIERS = SCRIPTS / "classifiers"

sys.path.insert(0, str(CLASSIFIERS))
sys.path.insert(0, str(LLM))
from signals import apply_training_disclosure_cap  # noqa: E402
from llm_client import (  # noqa: E402
    llm_enabled,
    validate_provider,
    LLMDisabled,
    LLMUnavailable,
    VALID_PROVIDERS,
)

LLM_CHIP_CONFIDENCE_THRESHOLD = 0.5
LLM_CHIP_CONFLICT_THRESHOLD = 0.7  # Skip conflict-check LLM when chip confidence is already high

FRAMEWORK_CHIP_MAP = {
    "pytorch": "nvidia",
    "tensorflow": "nvidia",
    "jax": "google_tpu",
    "paddlepaddle": "nvidia",
    "mxnet": "nvidia",
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

_EXPLICIT_TRAINING_DISCLOSURE_RE = re.compile(
    r'(?:trained?\s+on|training\s+(?:was\s+)?(?:done|performed|conducted|run)\s+on|'
    r'training\s+hardware|training\s+infrastructure|'
    r'fine[- ]?tun(?:ed|ing)\s+on|pre[- ]?train(?:ed|ing)\s+on|'
    r'we\s+train(?:ed)?\b|experiments?\b.{0,60}?\bconducted\s+on|'
    r'required\b.{0,60}?\b(?:gpu|gpus|tpu|tpus|h100|v100|a100|h200|p100|t4))',
    re.IGNORECASE,
)
_HARDWARE_LITERAL_RE = re.compile(
    r'(?:\bTPU(?:\s*v\d+)?\b|\bA100\b|\bH100\b|\bV100\b|\bH200\b|\bP100\b|\bT4\b|\bGPU(?:s)?\b|NVIDIA)',
    re.IGNORECASE,
)
_HARDWARE_DURATION_RE = re.compile(
    r'(?:'
    r'\b\d+\s*(?:x\s*)?(?:A100|H100|V100|H200|P100|T4|TPU(?:\s*v\d+)?|GPU)\b.{0,40}?\b(?:training|compute)|'
    r'\b(?:A100|H100|V100|H200|P100|T4|TPU(?:\s*v\d+)?)\b.{0,40}?\b(?:training|compute)'
    r')',
    re.IGNORECASE,
)
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
    """Load tests/ground_truth.csv and return {model_id: normalized_provider}."""
    gt_path = Path(__file__).parent / "tests" / "ground_truth.csv"
    if not gt_path.exists():
        return {}
    gt = {}
    with open(gt_path, encoding="utf-8") as f:
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


def run(script, cwd=None, extra_args=None):
    cmd = [sys.executable, str(script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"Running: {script.name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(cwd or script.parent))
    if result.returncode != 0:
        print(f"\nFailed: {script.name} (exit {result.returncode})")
        sys.exit(result.returncode)


def run_parallel(scripts, extra_args=None):
    """Launch multiple scripts in parallel and wait for all to finish.

    Output from each script is buffered and printed as a block when the
    script finishes, so lines from different scripts don't interleave.
    Exits with the first non-zero return code if any script fails.
    """
    extra_args = extra_args or []
    results = {}  # script → (returncode, output)
    lock = threading.Lock()

    def _run_one(script):
        cmd = [sys.executable, str(script)] + extra_args
        proc = subprocess.run(cmd, capture_output=True, text=True)
        with lock:
            results[script] = proc

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
        print(f"\nFailed: {failed[0]} (exit {failed[1]})")
        sys.exit(failed[1])


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


def _has_explicit_training_chip_evidence(snippets):
    for snippet in snippets or []:
        text = snippet.get("snippet", "")
        if ((_EXPLICIT_TRAINING_DISCLOSURE_RE.search(text) and _HARDWARE_LITERAL_RE.search(text))
                or _HARDWARE_DURATION_RE.search(text)):
            return True
    return False


def _is_runtime_only_chip_evidence(snippets):
    if not snippets or _has_explicit_training_chip_evidence(snippets):
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

    # Training-disclosure cap (safety net): if the scored snippets show no
    # explicit training-disclosure phrasing, cap at 0.6 so LLM fallback runs.
    # Classifiers also apply this, but re-applying here protects the aggregator
    # from any classifier that forgets.
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
    parser = argparse.ArgumentParser(description="Model hardware classifier pipeline")
    parser.add_argument("--top", type=int, default=None,
                        help="Max models to process per year (or total if --years not set)")
    parser.add_argument("--years", type=str, default=None,
                        help="Filter by model creation year(s): 2023 | 2022,2023 | 2022-2024")
    parser.add_argument("--update-models", action="store_true", default=False,
                        help="Re-fetch models.csv from HuggingFace (default: use existing)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers per classifier script (default: 4)")
    parser.add_argument("--llm", action="store_true", default=False,
                        help="Enable LLM fallback for chip classification and candidate selection (default: disabled)")
    parser.add_argument("--provider", choices=list(VALID_PROVIDERS), default="OPENAI",
                        help="LLM provider when --llm is set: OPENAI (gpt-4o-mini, default), "
                             "LOCAL (vLLM at localhost:8000 serving gemma-4-e2b-it), or OPENROUTER")
    args = parser.parse_args()

    if args.llm:
        os.environ["LLM_ENABLED"] = "1"
        os.environ["LLM_PROVIDER"] = args.provider

    check_tokens()

    ingest_args = []
    if args.top:
        ingest_args += ["--top", str(args.top)]
    if args.years:
        ingest_args += ["--years", args.years]
    worker_args = ["--workers", str(args.workers)]

    if args.update_models:
        run(INGEST / "get_models.py")
    run(INGEST / "get_modelcard.py", extra_args=ingest_args)
    run(INGEST / "get_github.py")
    run(INGEST / "get_arxiv.py")
    run(CLASSIFIERS / "evaluate_github.py", extra_args=worker_args)
    run(CLASSIFIERS / "evaluate_arxiv.py", extra_args=worker_args)
    run_parallel(
        [
            CLASSIFIERS / "from_modelcard.py",
            CLASSIFIERS / "from_githubcode.py",
            CLASSIFIERS / "from_arxiv.py",
        ],
        extra_args=worker_args,
    )

    build_results(workers=args.workers)


def build_results(workers=4):
    import asyncio

    db = Path(__file__).parent / "database"

    sys.path.insert(0, str(LLM))
    if llm_enabled():
        import llm_client as _llm_client
        _llm_client.set_concurrency(workers)
        from ask_llm_chip import ask_llm_chip
    else:
        ask_llm_chip = None

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

    for model_id, mc in modelcards.items():
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

        needs_llm = False
        if chip == "unknown" and quality_blocked_chip:
            needs_llm = False
        elif chip_conf < LLM_CHIP_CONFIDENCE_THRESHOLD:
            needs_llm = True
        elif chip != "unknown" and chip_conf < LLM_CHIP_CONFLICT_THRESHOLD:
            implied = FRAMEWORK_CHIP_MAP.get(fw)
            if implied and chip != implied:
                needs_llm = True

        # For derivative models with a base_model in our dataset, defer to Pass 2
        if is_derivative and needs_llm and base_model_id and base_model_id in mc_analysis:
            needs_llm = False

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

            llm_queue.append((len(results), model_id, {
                "model_name": model_id,
                "yaml_library": yaml_library,
                "framework": fw,
                "modelcard_excerpt": mc_text[:3000],
                "section_labeled_text": mca.get("section_labeled_text", ""),
                "extra_context": extra_context,
            }))

        results.append({
            "id": model_id,
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

    # ── LLM chip fallback (async parallelized) ────────────────────────
    llm_chip_calls = 0
    total_llm_chip_cost = 0.0

    if not llm_enabled():
        if llm_queue:
            print(f"  LLM disabled (--llm not set). Skipping {len(llm_queue)} chip fallback call(s).")
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

        for idx, model_id, llm_chip, llm_conf, cost, err in asyncio.run(_run_chip_llm_queue()):
            total_llm_chip_cost += cost
            if err:
                print(f"  LLM chip failed for {model_id}: {err}")
            elif llm_chip:
                results[idx]["conclusion"]["chip_provider"] = llm_chip
                results[idx]["conclusion"]["chip_provider_confidence"] = llm_conf
                results[idx]["conclusion"]["chip_provider_source"] = "llm_chip"
                llm_chip_calls += 1
                print(f"  LLM chip override for {model_id}: {llm_chip} (conf={llm_conf}, Cost: ${cost:.6f})")

    # ── Pass 2: Resolve derivative models via base_model ──────────────
    RUNTIME_CHIP_MAP = {
        "mlx": "apple",
        "coreml": "apple",
        "openvino": "intel",
        "onnx": None,
    }
    results_by_id = {r["id"]: r for r in results}

    for r in results:
        mca = mc_analysis.get(r["id"], {})
        if not mca.get("is_derivative", False):
            continue

        conclusion = r["conclusion"]
        current_chip = conclusion["chip_provider"]
        runtime_library = mca.get("runtime_library")
        base_model_id = mca.get("base_model")

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
                continue

        # If chip matches runtime and no base model resolved,
        # fall back to framework-default chip
        if chip_matches_runtime and conclusion.get("framework", "unknown") != "unknown":
            implied = FRAMEWORK_CHIP_MAP.get(conclusion["framework"])
            if implied:
                conclusion["chip_provider"] = implied
                conclusion["chip_provider_confidence"] = 0.4
                conclusion["chip_provider_source"] = "framework_default_derivative"
                print(f"  Derivative framework default for {r['id']}: {conclusion['framework']} -> {implied}")

    # ── Ground truth evaluation ────────────────────────────────────────
    ground_truth = load_ground_truth()
    if ground_truth:
        evaluate_ground_truth(results, ground_truth)

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
