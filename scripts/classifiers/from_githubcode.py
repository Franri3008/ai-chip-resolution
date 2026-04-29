import argparse
import concurrent.futures
import functools
import json
import os
import re
import time
import requests
from tqdm import tqdm

from signals import (
    HARDWARE_SIGNALS, FRAMEWORK_SIGNALS, DEPENDENCY_SIGNALS,
    CHIP_PROVIDERS, FRAMEWORKS, MIN_SCORE_THRESHOLD, CONFIDENCE_DIVISOR,
    apply_training_disclosure_cap, extract_training_snippets,
)

# ── Paths ─────────────────────────────────────────────────────────────

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "modelcards.json")
output_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "github_chip_analysis.json")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
    token_path = os.path.join(os.path.dirname(__file__), "..", "..", "keys", ".gh_token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            GITHUB_TOKEN = f.read().strip()

HEADERS = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

API_BASE = "https://api.github.com"

# Connection-pooled Session shared across worker threads. requests.get() at
# the module level opens a fresh TLS connection per call (~80-150 ms each) —
# with several thousand fetches per pipeline run this dominates the wall-time
# of from_githubcode. urllib3's connection pool inside Session is thread-safe.
_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)
_HTTP_ADAPTER = requests.adapters.HTTPAdapter(
    pool_connections=32, pool_maxsize=32, max_retries=0,
)
_SESSION.mount("https://", _HTTP_ADAPTER)
_SESSION.mount("http://", _HTTP_ADAPTER)

# ── Tier 1 file patterns (always fetch) ──────────────────────────────

TIER1_PATTERNS = [
    r'^requirements.*\.txt$',
    r'^setup\.py$',
    r'^setup\.cfg$',
    r'^pyproject\.toml$',
    r'^Dockerfile.*$',
    r'^Makefile$',
    r'^environment\.ya?ml$',
    r'^README\.md$',
    r'^[^/]*\.sh$',
    r'^configs?/[^/]*\.ya?ml$',
    r'^accelerate.*\.ya?ml$',
    r'^ds_config.*\.json$',
    r'^deepspeed.*\.json$',
    # Training scripts (promoted from tier 2 — most reliable training signals)
    r'train[^/]*\.py$',
    r'finetune[^/]*\.py$',
    r'pretrain[^/]*\.py$',
    r'run[^/]*\.py$',
]

# ── Tier 2 file patterns (fetch if Tier 1 inconclusive) ──────────────

TIER2_PATTERNS = [
    r'scripts/[^/]*\.sh$',
    r'CMakeLists\.txt$',
]
TIER2_MAX_FILES = 10

# ── File purpose classification (score multipliers) ──────────────────
# Training scripts get boosted, docs/runtime files get discounted.

_TRAINING_FILE_RE = re.compile(
    r'(?:train|finetune|pretrain|run_(?:train|finetune|pretrain))[^/]*\.py$',
    re.IGNORECASE,
)
_RUNTIME_FILE_RE = re.compile(
    r'(?:serve|server|infer|inference|predict|demo|app|api|deploy|benchmark)[^/]*\.py$',
    re.IGNORECASE,
)
_DOCS_FILE_RE = re.compile(
    r'(?:^docs/|^examples/|^notebooks/|^tutorials/|README\.md$)',
    re.IGNORECASE,
)

def get_file_purpose_mult(filepath):
    """Return a score multiplier based on file purpose."""
    if _TRAINING_FILE_RE.search(filepath):
        return 1.5
    if _RUNTIME_FILE_RE.search(filepath):
        return 0.3
    if _DOCS_FILE_RE.search(filepath):
        return 0.5
    return 1.0

IS_DEP_FILE_RE = re.compile(
    r'requirements.*\.txt|setup\.py|setup\.cfg|pyproject\.toml|environment\.ya?ml',
    re.IGNORECASE,
)

# Export/deployment context → discount matches (0.25x)
_EXPORT_RE = re.compile(
    r'(?:export(?:ed|ing|s)?\s+(?:to|with|as|in)|convert(?:ed|ing|s)?\s+(?:to|with|using|by)|'
    r'quantiz(?:ed|ing|ation)\s+(?:with|using|by|via)|'
    r'optimiz(?:ed|ing)\s+for|repackag(?:ed|ing)|provided\s+by|'
    r'supported?\s+(?:formats?|backends?|runtimes?)|deploy(?:ed|ing|ment)?\s+(?:to|with|on)|'
    r'inference\s+(?:with|using|on|via|backend)|serving\s+(?:with|via)|'
    r'also\s+(?:support|available|compatible)|compatible\s+with|works?\s+with)'
    r'.{0,40}',
    re.IGNORECASE,
)

# Training context → boost matches (2.0x)
_TRAINING_RE = re.compile(
    r'(?:trained?\s+(?:on|with|using)|training\s+(?:on|with|hardware|infrastructure|setup|cluster)|'
    r'fine[- ]?tun(?:ed|ing)\s+(?:on|with|using)|pre[- ]?train(?:ed|ing)\s+(?:on|with)|'
    r'(?:\d+\s*[x×]\s*)?(?:A100|H100|V100|TPU\s*v\d|P100|T4|MI\d{3}|Gaudi)\s*.*?(?:hours?|days?|weeks?)|'
    r'compute\s+(?:budget|resources?|infrastructure)|'
    r'(?:hours?|days?|weeks?)\s+(?:of\s+)?(?:training|compute)|'
    r'training\s+(?:was\s+)?(?:done|performed|conducted|run)\s+(?:on|using|with))',
    re.IGNORECASE,
)


# Multi-model library repos accumulate backend support code that can drown out
# model-specific training signals. Apply repo priors so the repo's primary
# training stack beats export/runtime mentions.
REPO_PRIORS = {
    "amazon-science/chronos-forecasting": {"framework": "pytorch", "chip": "nvidia"},
    "facebookresearch/convnext-v2": {"framework": "pytorch", "chip": "nvidia"},
    "huggingface/pytorch-image-models": {"framework": "pytorch", "chip": "nvidia"},
    "huggingface/transformers": {"framework": "pytorch", "chip": "nvidia"},
    "ukplab/sentence-transformers": {"framework": "pytorch", "chip": "nvidia"},
    "ultralytics/ultralytics": {"framework": "pytorch", "chip": "nvidia"},
    # Additional library repos: host many models, export/optimization code overwhelms training signals
    "pytorch/fairseq": {"framework": "pytorch", "chip": "nvidia"},       # xlm-roberta, wav2vec2
    "openai/clip": {"framework": "pytorch", "chip": "nvidia"},            # openai/clip-vit-*
    "openai/gpt-2": {"framework": "pytorch"},                             # openai-community/gpt2
    "stanford-futuredata/colbert": {"framework": "pytorch"},              # colbert-ir/colbertv2.0
    "baaivision/flagembedding": {"framework": "pytorch", "chip": "nvidia"},  # BAAI/bge-*
    "flagopen/flagembedding": {"framework": "pytorch", "chip": "nvidia"},
}

# ── GitHub API helpers ───────────────────────────────────────────────


# Many models in a quarterly sweep share the same upstream repo (e.g. dozens
# link to huggingface/transformers). Memoising tree fetches and raw-file reads
# avoids re-fetching the same content per model. lru_cache is thread-safe.
@functools.lru_cache(maxsize=8192)
def api_get(url):
    """GET from GitHub API with rate-limit handling. Cached per-process."""
    try:
        resp = _SESSION.get(url, timeout=15)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(0, reset - int(time.time())) + 1
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(min(wait, 120))
            resp = _SESSION.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.RequestException:
        return None


def parse_github_url(url):
    """Extract (owner, repo) from a GitHub URL."""
    match = re.match(r'https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', url.rstrip('/.'))
    if match:
        return match.group(1), match.group(2)
    return None


def get_repo_tree(owner, repo):
    """Fetch the recursive file tree for the default branch."""
    url = f"{API_BASE}/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    resp = api_get(url)
    if resp is None or "tree" not in resp:
        return None
    return [item["path"] for item in resp["tree"] if item["type"] == "blob"]


@functools.lru_cache(maxsize=8192)
def get_file_content(owner, repo, path):
    """Fetch raw file content from GitHub (via raw.githubusercontent.com).
    Cached per-process — many models point at the same library repos."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
    try:
        resp = _SESSION.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.text[:100_000]  # Cap at 100KB
        return None
    except requests.RequestException:
        return None


# ── File classification ──────────────────────────────────────────────


def classify_files(all_paths):
    """Partition file paths into tier1/tier2 and extract file-presence signals."""
    tier1, tier2 = [], []
    file_presence_hits = {}

    tier1_compiled = [re.compile(p, re.IGNORECASE) for p in TIER1_PATTERNS]
    tier2_compiled = [re.compile(p, re.IGNORECASE) for p in TIER2_PATTERNS]

    for path in all_paths:
        # Check file-presence signals (e.g., .cu files = NVIDIA)
        for provider, signals in HARDWARE_SIGNALS.items():
            for fp in signals.get("file_presence", []):
                if re.search(fp, path):
                    file_presence_hits[provider] = file_presence_hits.get(provider, 0) + 1

        if any(p.search(path) for p in tier1_compiled):
            tier1.append(path)
        elif any(p.search(path) for p in tier2_compiled):
            tier2.append(path)

    return tier1, tier2[:TIER2_MAX_FILES], file_presence_hits


# ── Content scanning ─────────────────────────────────────────────────


def _check_context(content, match_start):
    """Check context around a match for export/training signals. Returns multiplier."""
    window_start = max(0, match_start - 80)
    window_end = min(len(content), match_start + 80)
    preceding = content[window_start:match_start]
    context = content[window_start:window_end]
    if _EXPORT_RE.search(preceding):
        return 0.25
    if _TRAINING_RE.search(context):
        return 1.5
    return 1.0


def scan_content(content, filename):
    """Scan file content against all hardware + framework signal patterns."""
    scores = {}
    is_dep_file = bool(IS_DEP_FILE_RE.search(filename))
    purpose_mult = get_file_purpose_mult(filename)
    # Cap per-pattern matches to avoid runaway counts from repeated mentions
    max_matches_per_pattern = 5
    explicit_snippets = []

    all_signals = [
        (HARDWARE_SIGNALS, "hardware"),
        (FRAMEWORK_SIGNALS, "framework"),
    ]

    for signal_dict, signal_type in all_signals:
        for provider, signals in signal_dict.items():
            for level, weight in [("strong", 5), ("medium", 3), ("weak", 1)]:
                for pattern in signals.get(level, []):
                    count = 0
                    for m in re.finditer(pattern, content, re.IGNORECASE):
                        if count >= max_matches_per_pattern:
                            break
                        # Apply context multiplier for all non-dependency files
                        ctx_mult = _check_context(content, m.start()) if not is_dep_file else 1.0
                        # Apply file purpose multiplier (training 1.5x, docs 0.5x, runtime 0.3x)
                        scores[provider] = scores.get(provider, 0) + weight * ctx_mult * purpose_mult
                        count += 1

                        if signal_type == "hardware":
                            start = max(0, m.start() - 100)
                            end = min(len(content), m.end() + 100)
                            snippet = content[start:end].replace("\n", " ").strip()
                            explicit_snippets.append({
                                "provider": provider,
                                "snippet": f"...{snippet}...",
                                "file": filename
                            })

    # Dependency bonus in requirements/setup files
    if is_dep_file:
        content_lower = content.lower()
        for key, deps in DEPENDENCY_SIGNALS.items():
            for dep in deps:
                if dep.lower() in content_lower:
                    scores[key] = scores.get(key, 0) + 6

    return scores, explicit_snippets


def apply_repo_priors(chip_scores, fw_scores, owner_repo):
    """Bias known library repos toward their primary training stack."""
    prior = REPO_PRIORS.get(owner_repo.lower())
    if not prior:
        return chip_scores, fw_scores

    chip_scores = dict(chip_scores)
    fw_scores = dict(fw_scores)

    prior_fw = prior.get("framework")
    if prior_fw:
        base = fw_scores.get(prior_fw, 0)
        cap = max(base * 2, 10)
        for fw_name in list(fw_scores):
            if fw_name != prior_fw:
                fw_scores[fw_name] = min(fw_scores[fw_name], cap)
        fw_scores[prior_fw] = max(fw_scores.get(prior_fw, 0), cap + 1)

    # We NO LONGER cap competing chip providers based on repo priors.
    # We want explicit hardware signals to natively overpower the prior.

    return chip_scores, fw_scores


# ── Repo analysis ────────────────────────────────────────────────────

def analyze_repo(owner, repo):
    """Analyze a GitHub repo for chip provider and framework signals."""
    all_paths = get_repo_tree(owner, repo)
    if all_paths is None:
        return {
            "chip_provider": "unknown", "chip_provider_score": 0,
            "chip_provider_confidence": 0.0, "chip_providers_all": {},
            "framework": "unknown", "framework_score": 0,
            "framework_confidence": 0.0, "frameworks_all": {},
            "detection_files": [], "error": "tree_fetch_failed",
        }

    tier1, tier2, file_presence_scores = classify_files(all_paths)

    # Start with file-presence scores
    total_scores = {}
    for provider, count in file_presence_scores.items():
        total_scores[provider] = total_scores.get(provider, 0) + count * 4

    detection_files = []
    chip_snippets = []
    training_snippets = []

    def _accumulate_training(content, path):
        # Training-disclosure sentences that don't sit near a chip literal
        # (chip_snippets catches those). These are for the LLM to read —
        # e.g. a README saying "trained on 16 nodes of our internal cluster".
        if not content:
            return
        per_file_budget = 2 if _DOCS_FILE_RE.search(path) else 3
        new = extract_training_snippets(content, source=path, max_snippets=per_file_budget)
        training_snippets.extend(new)

    # Fetch and scan Tier 1 files
    for path in tier1:
        content = get_file_content(owner, repo, path)
        if content:
            scores, snips = scan_content(content, path)
            for key, sc in scores.items():
                total_scores[key] = total_scores.get(key, 0) + sc
            if scores:
                detection_files.append(path)
            chip_snippets.extend(snips)
            _accumulate_training(content, path)

    # If top chip score < threshold, also scan Tier 2 (no penalty — purpose multipliers handle weighting)
    chip_scores = {k: v for k, v in total_scores.items() if k in CHIP_PROVIDERS}
    top_chip = max(chip_scores.values()) if chip_scores else 0
    if top_chip < MIN_SCORE_THRESHOLD:
        for path in tier2:
            content = get_file_content(owner, repo, path)
            if content:
                scores, snips = scan_content(content, path)
                for key, sc in scores.items():
                    total_scores[key] = total_scores.get(key, 0) + sc
                if scores:
                    detection_files.append(path)
                chip_snippets.extend(snips)
                _accumulate_training(content, path)

    # Split scores into chips and frameworks
    chip_scores = {k: v for k, v in total_scores.items() if k in CHIP_PROVIDERS and v > 0}
    fw_scores = {k: v for k, v in total_scores.items() if k in FRAMEWORKS and v > 0}
    owner_repo = f"{owner}/{repo}"
    chip_scores, fw_scores = apply_repo_priors(chip_scores, fw_scores, owner_repo)

    # Determine top chip provider
    sorted_chips = sorted(chip_scores.items(), key=lambda x: -x[1])
    if sorted_chips and sorted_chips[0][1] >= MIN_SCORE_THRESHOLD:
        top_chip_name, top_chip_sc = sorted_chips[0]
        chip_conf = min(1.0, round(top_chip_sc / CONFIDENCE_DIVISOR, 2))
    else:
        top_chip_name, top_chip_sc, chip_conf = "unknown", 0, 0.0

    # Cap chip confidence for known library repos: their chip signals come from
    # export/optimization code, not model-specific training. Capping below
    # LLM_CHIP_CONFIDENCE_THRESHOLD (0.5) ensures the LLM is consulted.
    if owner_repo.lower() in {k.lower() for k in REPO_PRIORS} and top_chip_name != "unknown":
        chip_conf = min(chip_conf, 0.45)

    if top_chip_name != "unknown":
        chip_conf = apply_training_disclosure_cap(chip_conf, chip_snippets)

    # Determine top framework
    sorted_fw = sorted(fw_scores.items(), key=lambda x: -x[1])
    if sorted_fw and sorted_fw[0][1] >= MIN_SCORE_THRESHOLD:
        top_fw_name, top_fw_sc = sorted_fw[0]
        fw_conf = min(1.0, round(top_fw_sc / CONFIDENCE_DIVISOR, 2))
    else:
        top_fw_name, top_fw_sc, fw_conf = "unknown", 0, 0.0

    # Cap training_snippets globally to keep the LLM prompt bounded.
    training_snippets = training_snippets[:10]

    return {
        "chip_provider": top_chip_name,
        "chip_provider_score": top_chip_sc,
        "chip_provider_confidence": chip_conf,
        "chip_providers_all": dict(sorted_chips),
        "framework": top_fw_name,
        "framework_score": top_fw_sc,
        "framework_confidence": fw_conf,
        "frameworks_all": dict(sorted_fw),
        "detection_files": detection_files,
        "chip_snippets": chip_snippets,
        "training_snippets": training_snippets,
    }


# ── Main pipeline ────────────────────────────────────────────────────

def _analyze_model(model):
    """Top-level worker function for one model (thread-safe)."""
    github_url = model.get("main_github")
    model_id = model.get("id", "")

    if not github_url:
        return {
            "id": model_id, "github_url": None,
            "chip_provider": "unknown", "chip_provider_score": 0,
            "chip_provider_confidence": 0.0, "chip_providers_all": {},
            "framework": "unknown", "framework_score": 0,
            "framework_confidence": 0.0, "frameworks_all": {},
            "detection_files": [],
        }

    parsed = parse_github_url(github_url)
    if not parsed:
        return {
            "id": model_id, "github_url": github_url,
            "chip_provider": "unknown", "chip_provider_score": 0,
            "chip_provider_confidence": 0.0, "chip_providers_all": {},
            "framework": "unknown", "framework_score": 0,
            "framework_confidence": 0.0, "frameworks_all": {},
            "detection_files": [], "error": "unparseable_url",
        }

    owner, repo = parsed
    result = analyze_repo(owner, repo)
    result["id"] = model_id
    result["github_url"] = github_url
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel worker threads (default: 4)")
    args = parser.parse_args()

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(_analyze_model, data),
            total=len(data),
            desc="Analyzing GitHub repos",
        ))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\nResults saved to {output_path}")
    print(f"{'Model':50s} {'Chip':12s} {'Conf':6s} {'Framework':12s} {'Conf':6s}  All chips")
    print("-" * 120)
    for r in results:
        chip = r.get("chip_provider", "unknown")
        chip_conf = r.get("chip_provider_confidence", 0)
        fw = r.get("framework", "unknown")
        fw_conf = r.get("framework_confidence", 0)
        all_chips = r.get("chip_providers_all", {})
        print(f"  {r['id']:48s} {chip:12s} {chip_conf:<6.2f} {fw:12s} {fw_conf:<6.2f}  {all_chips}")


if __name__ == "__main__":
    main()
