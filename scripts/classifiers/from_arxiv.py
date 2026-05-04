"""Classify chip provider from arXiv paper HTML (via ar5iv).

Fetches the HTML rendering of each model's selected arXiv paper, parses
sections by heading, and scores hardware signals with section-aware
weighting.
"""

import argparse
import concurrent.futures
import json
import os
import re
import urllib.request
from tqdm import tqdm

from signals import (
    HARDWARE_SIGNALS,
    CHIP_PROVIDERS, MIN_SCORE_THRESHOLD, CONFIDENCE_DIVISOR,
    apply_training_disclosure_cap, extract_training_snippets,
)

# ── Paths ─────────────────────────────────────────────────────────────

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "modelcards.json")
output_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "arxiv_chip_analysis.json")

# ── Section classification by heading ─────────────────────────────────

_TRAINING_HEADING_RE = re.compile(
    r'experiment|implementation|training|setup|infrastructure|hardware|compute|'
    r'pre[- ]?training|fine[- ]?tun',
    re.IGNORECASE,
)
_METHOD_HEADING_RE = re.compile(
    r'method|approach|model|architecture|framework|design',
    re.IGNORECASE,
)
_RESULTS_HEADING_RE = re.compile(
    r'result|evaluation|benchmark|comparison|ablation|performance',
    re.IGNORECASE,
)
_REFERENCES_HEADING_RE = re.compile(
    r'reference|bibliograph|acknowledg|appendix|supplement',
    re.IGNORECASE,
)

_SECTION_WEIGHTS = {
    "training": 1.5,
    "method": 1.0,
    "abstract": 1.0,
    "results": 0.5,
    "references": 0.2,
    "body": 0.8,
}

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


# ── HTML parsing helpers ──────────────────────────────────────────────

_TAG_RE = re.compile(r'<[^>]+>')
_HEADING_RE = re.compile(r'<(h[1-6])[^>]*>(.*?)</\1>', re.IGNORECASE | re.DOTALL)
# Full heading block (open + content + close) — captured as one piece by re.split.
# The previous version captured only the opening tag, so the heading-detector
# (which expected <hN>…</hN> in the same part) never matched and every paper
# collapsed into a single 10000-char "body" section, hiding the training section.
_SECTION_SPLIT_RE = re.compile(r'(<h[1-3][^>]*>.*?</h[1-3]>)', re.IGNORECASE | re.DOTALL)


def _strip_tags(html):
    """Remove HTML tags, decode common entities."""
    text = _TAG_RE.sub(' ', html)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ').replace('&#x27;', "'")
    return re.sub(r'\s+', ' ', text).strip()


def _classify_heading(heading_text):
    """Classify a heading into a section type."""
    text = _strip_tags(heading_text).lower()
    if 'abstract' in text:
        return "abstract"
    if _TRAINING_HEADING_RE.search(text):
        return "training"
    if _METHOD_HEADING_RE.search(text):
        return "method"
    if _RESULTS_HEADING_RE.search(text):
        return "results"
    if _REFERENCES_HEADING_RE.search(text):
        return "references"
    return "body"


def parse_paper_sections(html):
    """Split paper HTML into (section_type, plain_text) tuples."""
    sections = []

    # Try to extract abstract separately
    abstract_m = re.search(
        r'<h\d[^>]*>[^<]*abstract[^<]*</h\d>(.*?)(?=<h[1-3])',
        html, re.IGNORECASE | re.DOTALL,
    )
    if abstract_m:
        sections.append(("abstract", _strip_tags(abstract_m.group(1))[:5000]))

    # Split by full <hN>…</hN> heading blocks (captured by _SECTION_SPLIT_RE).
    parts = _SECTION_SPLIT_RE.split(html)
    current_type = "body"
    current_text = ""

    for part in parts:
        heading_m = re.fullmatch(r'<h[1-3][^>]*>(.*?)</h[1-3]>', part, re.IGNORECASE | re.DOTALL)
        if heading_m:
            # Save previous section
            if current_text.strip():
                sections.append((current_type, _strip_tags(current_text)[:15000]))
            current_type = _classify_heading(heading_m.group(1))
            current_text = ""
        else:
            current_text += part

    # Save last section
    if current_text.strip():
        sections.append((current_type, _strip_tags(current_text)[:15000]))

    return sections


# ── Content scanning ──────────────────────────────────────────────────

def _check_context(text, match_start):
    """Check context around a match for export/training signals. Returns multiplier."""
    window_start = max(0, match_start - 80)
    window_end = min(len(text), match_start + 80)
    preceding = text[window_start:match_start]
    context = text[window_start:window_end]
    if _EXPORT_RE.search(preceding):
        return 0.25
    if _TRAINING_RE.search(context):
        return 1.5
    return 1.0


def scan_section(text, section_type):
    """Scan section text for hardware signals. Returns (scores, chip_snippets)."""
    scores = {}
    snippets = []
    section_weight = _SECTION_WEIGHTS.get(section_type, 0.8)
    max_matches_per_pattern = 5

    for provider, signals in HARDWARE_SIGNALS.items():
        for level, weight in [("strong", 5), ("medium", 3), ("weak", 1)]:
            for pattern in signals.get(level, []):
                if level == "file_presence":
                    continue
                count = 0
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    if count >= max_matches_per_pattern:
                        break
                    ctx_mult = _check_context(text, m.start())
                    total_weight = weight * ctx_mult * section_weight
                    scores[provider] = scores.get(provider, 0) + total_weight
                    count += 1

                    start = max(0, m.start() - 150)
                    end = min(len(text), m.end() + 150)
                    snippet = text[start:end].strip()
                    snippets.append({
                        "provider": provider,
                        "snippet": f"...{snippet}...",
                        "section": section_type,
                    })

    return scores, snippets


# ── Paper analysis ────────────────────────────────────────────────────

def fetch_paper_html(arxiv_id):
    """Fetch HTML rendering from ar5iv.

    A 12s timeout is enough for ar5iv when the paper renders. When ar5iv
    can't or won't render the paper (newer submissions, transient errors)
    a longer timeout would just stall the worker pool. Any rendering
    failure → return None and the caller treats the paper as unknown.
    """
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "model-classifier/1.0"})
        resp = urllib.request.urlopen(req, timeout=12)
        return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _extract_arxiv_id(url):
    """Extract bare arxiv ID from an abs/pdf URL."""
    m = re.search(r'arxiv\.org/(?:abs|pdf)/([\w.]+?)(?:v\d+)?(?:\.pdf)?$', url)
    return m.group(1) if m else None


def analyze_paper(arxiv_id):
    """Analyze a single arXiv paper for chip-provider signals."""
    html = fetch_paper_html(arxiv_id)
    if not html:
        return {
            "chip_provider": "unknown", "chip_provider_score": 0,
            "chip_provider_confidence": 0.0, "chip_providers_all": {},
            "detection_sections": [], "chip_snippets": [],
            "error": "fetch_failed",
        }

    sections = parse_paper_sections(html)
    total_scores = {}
    all_snippets = []
    training_snippets = []
    detection_sections = []

    for section_type, section_text in sections:
        scores, snippets = scan_section(section_text, section_type)
        for key, sc in scores.items():
            total_scores[key] = total_scores.get(key, 0) + sc
        if scores and section_type not in detection_sections:
            detection_sections.append(section_type)
        all_snippets.extend(snippets)
        if section_type in ("training", "method", "abstract"):
            training_snippets.extend(extract_training_snippets(
                section_text, source=section_type, max_snippets=3,
            ))

    chip_scores = {k: round(v, 1) for k, v in total_scores.items() if k in CHIP_PROVIDERS and v > 0}

    sorted_chips = sorted(chip_scores.items(), key=lambda x: -x[1])
    if sorted_chips and sorted_chips[0][1] >= MIN_SCORE_THRESHOLD:
        top_chip_name, top_chip_sc = sorted_chips[0]
        chip_conf = min(1.0, round(top_chip_sc / CONFIDENCE_DIVISOR, 2))
        chip_conf = apply_training_disclosure_cap(chip_conf, all_snippets)
    else:
        top_chip_name, top_chip_sc, chip_conf = "unknown", 0, 0.0

    return {
        "chip_provider": top_chip_name,
        "chip_provider_score": top_chip_sc,
        "chip_provider_confidence": chip_conf,
        "chip_providers_all": dict(sorted_chips),
        "detection_sections": detection_sections,
        "chip_snippets": all_snippets[:20],
        "training_snippets": training_snippets[:8],
    }


# ── Main pipeline ────────────────────────────────────────────────────

def _analyze_model(model):
    """Top-level worker function for one model (thread-safe)."""
    arxiv_url = model.get("main_arxiv")
    model_id = model.get("id", "")

    if not arxiv_url:
        return {
            "id": model_id, "arxiv_url": None,
            "chip_provider": "unknown", "chip_provider_score": 0,
            "chip_provider_confidence": 0.0, "chip_providers_all": {},
            "detection_sections": [], "chip_snippets": [],
        }

    arxiv_id = _extract_arxiv_id(arxiv_url)
    if not arxiv_id:
        return {
            "id": model_id, "arxiv_url": arxiv_url,
            "chip_provider": "unknown", "chip_provider_score": 0,
            "chip_provider_confidence": 0.0, "chip_providers_all": {},
            "detection_sections": [], "chip_snippets": [],
            "error": "unparseable_url",
        }

    result = analyze_paper(arxiv_id)
    result["id"] = model_id
    result["arxiv_url"] = arxiv_url
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
            desc="Analyzing arXiv papers",
        ))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    for r in results:
        chip = r.get("chip_provider", "unknown")
        chip_conf = r.get("chip_provider_confidence", 0)
        all_chips = r.get("chip_providers_all", {})
        if r.get("arxiv_url"):
            print(f"  {r['id']:48s} {chip:12s} {chip_conf:<6.2f}  {all_chips}")


if __name__ == "__main__":
    main()
