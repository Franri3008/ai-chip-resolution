import argparse
import concurrent.futures
import json
import os
import re

from tqdm import tqdm

from signals import (
    HARDWARE_SIGNALS,
    CHIP_PROVIDERS, MIN_SCORE_THRESHOLD, CONFIDENCE_DIVISOR,
    apply_training_disclosure_cap,
    HARDWARE_DURATION_RE, EXPLICIT_TRAINING_DISCLOSURE_RE, HARDWARE_LITERAL_RE,
)

# ── Paths ─────────────────────────────────────────────────────────────

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "modelcards.json")
output_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "modelcard_chip_analysis.json")

# ── Section weight config ────────────────────────────────────────────
# Sections of the model card carry different trust levels for
# determining what hardware/framework was actually *used*.

SECTION_WEIGHTS = {
    "yaml_frontmatter": 1.5,
    "training":         1.5,   # "Training", "Hardware", "Infrastructure"
    "body":             1.0,   # Default body text
    "compatibility":    0.5,   # "Compatibility", "Supported hardware", "Requirements"
    "references":       0.3,   # "Citation", "References", "Acknowledgments", "BibTeX"
    "table":            0.7,   # Benchmark / comparison tables
}

# Heading patterns → section type. Tolerates numbered prefixes like "### 6.7 …".
_HEAD_PREFIX = r'#+\s*(?:\d+(?:\.\d+)*\.?\s+)?'

_TRAINING_HEADINGS = re.compile(
    _HEAD_PREFIX + r'(?:training|hardware|infrastructure|compute|setup|pre[- ]?training|fine[- ]?tuning)',
    re.IGNORECASE,
)
_COMPAT_HEADINGS = re.compile(
    _HEAD_PREFIX + r'(?:compatib|supported?\s+hardware|requirements?|installation|deploy|usage|'
    r'how\s+to\s+(?:use|run)|inference|recommended\s+inference|running|run\s+on|'
    r'serving|local\s+(?:run|inference)|getting\s+started|quick\s*start)',
    re.IGNORECASE,
)
_REF_HEADINGS = re.compile(
    _HEAD_PREFIX + r'(?:citation|references?|acknowledg|bibtex|license|paper)',
    re.IGNORECASE,
)

# ── Negation & comparison context ────────────────────────────────────

_NEGATION_RE = re.compile(
    r'(?:not|no|without|doesn\'t\s+support|does\s+not|unlike|except|lack(?:s|ing)?|incompatible\s+with)'
    r'.{0,40}',
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r'(?:compared\s+to|faster\s+than|slower\s+than|vs\.?|versus|outperforms?|benchmark)'
    r'.{0,40}',
    re.IGNORECASE,
)

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

_TRAINING_RE = re.compile(
    r'(?:trained?\s+(?:on|with|using|from\s+scratch)|training\s+(?:on|with|hardware|infrastructure|setup|cluster)|'
    r'fine[- ]?tun(?:ed|ing)\s+(?:on|with|using)|pre[- ]?train(?:ed|ing)\s+(?:on|with)|'
    r'(?:\d+\s*[x×]\s*)?(?:A100|A800|H100|H800|V100|TPU\s*v\d|P100|T4|MI\d{3}|Gaudi|'
    r'Ascend(?:\s*\d{3}[A-Da-d]?)?|910[ABCDabcd]|Atlas\s*\d{3}|MLU\s*\d{3})\s*.*?(?:hours?|days?|weeks?|processors|chips|cores|cluster)|'
    r'compute\s+(?:budget|resources?|infrastructure)|'
    r'(?:hours?|days?|weeks?)\s+(?:of\s+)?(?:training|compute)|'
    r'training\s+(?:was\s+)?(?:done|performed|conducted|run)\s+(?:on|using|with)|'
    r'国产算力|完全基于国产|domestic\s+(?:chinese\s+)?computing)',
    re.IGNORECASE,
)

# ── Derivative / converted model detection ────────────────────────────
# Inference-runtime library_name values indicate the model is a
# converted/quantized derivative, NOT training hardware.
_INFERENCE_RUNTIME_LIBRARIES = {"mlx", "coreml", "openvino", "onnx"}

_DERIVATIVE_NAME_RE = re.compile(
    r'(?:MLX|GGUF|ONNX|AWQ|GPTQ|EXL2|quantized|4bit|8bit|fp16|bf16|'
    r'CoreML|OpenVINO|INT[48]|Q[2-8]_[KM0-9])',
    re.IGNORECASE,
)

# Fine-tune / upstream-derivative markers that appear in the card text itself.
# When the title heading or an early body sentence describes this model as a
# fine-tune of an upstream model, any linked arXiv paper is almost always for
# the BASE model — its hardware disclosure does not apply to this checkpoint.
_FINETUNE_TITLE_RE = re.compile(
    r'^\s*#+\s*[^\n]*\bfine[- ]?tun(?:ed|ing)\b',
    re.IGNORECASE | re.MULTILINE,
)
_FINETUNE_BODY_RE = re.compile(
    # "fine-tuned from <hf-org>/<repo>" or "fine-tune of <hf-org>/<repo>"
    r'\bfine[- ]?tun(?:ed|ing|e)\s+(?:from|of|version\s+of|on\s+top\s+of)\s+'
    r'(?:\[?(?:google|facebook|microsoft|meta|huggingface|nvidia|stabilityai|laion|'
    r'BAAI|sentence-transformers|FacebookAI|deepseek-ai|mistralai|01-ai|'
    r'[\w-]+)/[\w./-]+\]?|the\s+\w+\s+model)',
    re.IGNORECASE,
)

# ── YAML frontmatter keywords → signals ──────────────────────────────

_YAML_TAG_KEYWORDS = {
    "tpu": ("google_tpu", 6),
    "cuda": ("nvidia", 5),
    "gpu": ("nvidia", 2),
    "rocm": ("amd", 6),
    "gaudi": ("intel", 6),
    "habana": ("intel", 6),
    "inferentia": ("aws", 6),
    "trainium": ("aws", 6),
    "ascend": ("huawei_ascend", 6),
    "mindspore": ("huawei_ascend", 5),
    "cambricon": ("cambricon", 6),
    "mlu": ("cambricon", 5),
    "kunlun": ("baidu_kunlun", 6),
    "kunlunxin": ("baidu_kunlun", 6),
    "xpu": ("baidu_kunlun", 3),
    "musa": ("moore_threads", 6),
    "mthreads": ("moore_threads", 6),
    "iluvatar": ("iluvatar", 6),
    "corex": ("iluvatar", 5),
    "hygon": ("hygon", 6),
    "dcu": ("hygon", 5),
    "metax": ("metax", 6),
    "muxi": ("metax", 6),
}


# ── Helpers ───────────────────────────────────────────────────────────


def parse_yaml_frontmatter(text):
    """Extract YAML frontmatter and return (yaml_text, body_text)."""
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end == -1:
        return "", text
    return text[3:end], text[end + 4:]


def detect_derivative(yaml_text, model_id, body_text=""):
    """Detect if a model is a derivative (quantized / converted / fine-tuned).

    Returns (is_derivative, base_model, runtime_library).
    """
    yaml_lower = yaml_text.lower()

    # Check library_name for inference runtime
    runtime_library = None
    lib_m = re.search(r'library_name:\s*(\S+)', yaml_lower)
    if lib_m:
        lib = lib_m.group(1).strip()
        if lib in _INFERENCE_RUNTIME_LIBRARIES:
            runtime_library = lib

    # Check for base_model field in YAML
    base_model = None
    bm = re.search(r'base_model:\s*(\S+)', yaml_text)
    if bm:
        base_model = bm.group(1).strip()
    # Also check list format: base_model:\n  - model_id
    if not base_model:
        bm_list = re.search(r'base_model:\s*\n\s*-\s*(\S+)', yaml_text)
        if bm_list:
            base_model = bm_list.group(1).strip()

    # Check model name for derivative indicators
    name_is_derivative = bool(_DERIVATIVE_NAME_RE.search(model_id))

    # Card-text derivative indicators: a "Fine-Tuned …" title heading or an
    # explicit "fine-tuned from <hf-org>/<repo>" body sentence both mark this
    # as an upstream derivative — its linked arXiv paper is the BASE's paper,
    # not this model's training disclosure.
    finetune_titled = bool(body_text and _FINETUNE_TITLE_RE.search(body_text[:500]))
    finetune_text = bool(body_text and _FINETUNE_BODY_RE.search(body_text))
    finetune_derivative = finetune_titled or finetune_text

    is_derivative = (
        bool(runtime_library)
        or (name_is_derivative and base_model is not None)
        or finetune_derivative
    )

    return is_derivative, base_model, runtime_library


def extract_yaml_signals(yaml_text):
    """Score structured YAML chip-tag signals."""
    scores = {}
    yaml_lower = yaml_text.lower()

    for tag_match in re.finditer(r'(?:^|\n)\s*-\s+(\S+)', yaml_text):
        tag = tag_match.group(1).strip().lower()
        if tag in _YAML_TAG_KEYWORDS:
            key, pts = _YAML_TAG_KEYWORDS[tag]
            scores[key] = scores.get(key, 0) + pts

    inline = re.search(r'tags:\s*\[([^\]]+)\]', yaml_lower)
    if inline:
        for tag in inline.group(1).split(","):
            tag = tag.strip().strip("'\"")
            if tag in _YAML_TAG_KEYWORDS:
                key, pts = _YAML_TAG_KEYWORDS[tag]
                scores[key] = scores.get(key, 0) + pts

    return scores


def split_into_sections(body):
    """Split markdown body into (section_type, text) pairs."""
    sections = []
    current_type = "body"
    current_lines = []

    for line in body.split("\n"):
        # Check for heading transitions
        if line.startswith("#"):
            # Flush current section
            if current_lines:
                sections.append((current_type, "\n".join(current_lines)))
                current_lines = []

            if _TRAINING_HEADINGS.match(line):
                current_type = "training"
            elif _COMPAT_HEADINGS.match(line):
                current_type = "compatibility"
            elif _REF_HEADINGS.match(line):
                current_type = "references"
            else:
                current_type = "body"

        current_lines.append(line)

    if current_lines:
        sections.append((current_type, "\n".join(current_lines)))

    return sections


def is_table_line(line):
    """Check if a line looks like a markdown table row."""
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def check_context(text, match_start, match_end=None):
    """Check the context around a match for negation/comparison/export/training.

    Returns a multiplier:
      0.0  negated
      0.25 export / runtime-deploy
      0.3  comparative
      2.5  unambiguous training-disclosure (chip literal + duration phrase nearby)
      1.5  training-context phrase nearby
      1.0  normal
    """
    window_start = max(0, match_start - 80)
    if match_end is None:
        match_end = match_start
    window_end = min(len(text), match_end + 80)
    preceding = text[window_start:match_start]
    context = text[window_start:window_end]

    if _NEGATION_RE.search(preceding):
        return 0.0
    if _COMPARISON_RE.search(preceding):
        return 0.3
    if _EXPORT_RE.search(preceding):
        return 0.25
    # Strong: a chip literal sits inside an explicit duration disclosure
    # ("2.788M H800 GPU hours", "2048 Ascend 910 processors trained for X").
    # Use a wider window for this check — these phrases can run long.
    wide = text[max(0, match_start - 160):min(len(text), match_end + 160)]
    if HARDWARE_DURATION_RE.search(wide) and HARDWARE_LITERAL_RE.search(wide):
        return 2.5
    if _TRAINING_RE.search(context):
        return 1.5
    return 1.0


def scan_section(text, section_type):
    """Scan a section of text and return weighted scores."""
    scores = {}
    section_weight = SECTION_WEIGHTS.get(section_type, 1.0)
    matched = False

    # Determine per-line table weight
    lines = text.split("\n")
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append((offset, is_table_line(line)))
        offset += len(line) + 1

    def get_line_weight(pos):
        """Return table-adjusted weight for a character position."""
        for i, (lo, is_tbl) in enumerate(line_offsets):
            next_lo = line_offsets[i + 1][0] if i + 1 < len(line_offsets) else len(text)
            if lo <= pos < next_lo:
                return SECTION_WEIGHTS["table"] if is_tbl else 1.0
        return 1.0

    explicit_snippets = []

    for provider, signals in HARDWARE_SIGNALS.items():
        for level, base_weight in [("strong", 5), ("medium", 3), ("weak", 1)]:
            for pattern in signals.get(level, []):
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    ctx_mult = check_context(text, m.start(), m.end())
                    if ctx_mult == 0.0:
                        continue
                    tbl_weight = get_line_weight(m.start())
                    final = base_weight * section_weight * ctx_mult * tbl_weight
                    scores[provider] = scores.get(provider, 0) + final
                    matched = True

                    start = max(0, m.start() - 100)
                    end = min(len(text), m.end() + 100)
                    snippet = text[start:end].replace("\n", " ").strip()
                    explicit_snippets.append({
                        "provider": provider,
                        "snippet": f"...{snippet}..."
                    })

    return scores, matched, explicit_snippets


# ── Model card analysis ──────────────────────────────────────────────


def analyze_modelcard(modelcard_text, model_id=""):
    """Analyze a single model card and return chip-provider scores."""
    if not modelcard_text or not modelcard_text.strip():
        return {
            "chip_provider": "unknown", "chip_provider_score": 0,
            "chip_provider_confidence": 0.0, "chip_providers_all": {},
            "matched_sections": [],
        }

    yaml_text, body = parse_yaml_frontmatter(modelcard_text)
    total_scores = {}
    matched_sections = []
    chip_snippets = []

    is_derivative, base_model, runtime_library = detect_derivative(yaml_text, model_id, body)

    yaml_scores = extract_yaml_signals(yaml_text)
    for key, sc in yaml_scores.items():
        total_scores[key] = total_scores.get(key, 0) + sc
    if yaml_scores:
        matched_sections.append("yaml_frontmatter")

    yaml_scan_scores, yaml_matched, yaml_snips = scan_section(yaml_text, "yaml_frontmatter")
    for key, sc in yaml_scan_scores.items():
        total_scores[key] = total_scores.get(key, 0) + sc
    if yaml_matched and "yaml_frontmatter" not in matched_sections:
        matched_sections.append("yaml_frontmatter")
    chip_snippets.extend(yaml_snips)

    sections = split_into_sections(body)
    for section_type, section_text in sections:
        sec_scores, sec_matched, sec_snips = scan_section(section_text, section_type)
        # Derivative models: discount all chip scores from body text since the
        # card describes the conversion/inference process, not training.
        if is_derivative:
            for key in list(sec_scores):
                sec_scores[key] = sec_scores[key] * 0.25
        for key, sc in sec_scores.items():
            total_scores[key] = total_scores.get(key, 0) + sc
        if sec_matched and section_type not in matched_sections:
            matched_sections.append(section_type)
        chip_snippets.extend(sec_snips)

    chip_scores = {k: round(v, 1) for k, v in total_scores.items() if k in CHIP_PROVIDERS and v > 0}

    sorted_chips = sorted(chip_scores.items(), key=lambda x: -x[1])
    if sorted_chips and sorted_chips[0][1] >= MIN_SCORE_THRESHOLD:
        top_chip_name, top_chip_sc = sorted_chips[0]
        chip_conf = min(1.0, round(top_chip_sc / CONFIDENCE_DIVISOR, 2))
        chip_conf = apply_training_disclosure_cap(chip_conf, chip_snippets)
    else:
        top_chip_name, top_chip_sc, chip_conf = "unknown", 0, 0.0

    # Build section-labeled text for LLM consumption.
    # Ordering: surface training-bearing content first so the LLM's limited window
    # isn't eaten by YAML license boilerplate. Priority: training > compatibility >
    # body > references; YAML last (trimmed).
    _PRIORITY = {"training": 0, "compatibility": 2, "body": 3, "references": 4}
    labeled_sections = []
    ordered = sorted(
        enumerate(sections),
        key=lambda it: (_PRIORITY.get(it[1][0], 3), it[0]),
    )
    for _, (section_type, section_text) in ordered:
        trimmed = section_text.strip()
        if not trimmed:
            continue
        label = section_type.upper().replace("_", " ")
        labeled_sections.append(f"=== [{label}] ===\n{trimmed}")
    yaml_trimmed = yaml_text.strip()
    if yaml_trimmed:
        yaml_tail = yaml_trimmed[:800]
        labeled_sections.append(f"=== [YAML FRONTMATTER] ===\n{yaml_tail}")
    section_labeled_text = "\n\n".join(labeled_sections)

    return {
        "chip_provider": top_chip_name,
        "chip_provider_score": top_chip_sc,
        "chip_provider_confidence": chip_conf,
        "chip_providers_all": dict(sorted_chips),
        "matched_sections": matched_sections,
        "chip_snippets": chip_snippets,
        "section_labeled_text": section_labeled_text,
        "is_derivative": is_derivative,
        "base_model": base_model,
        "runtime_library": runtime_library,
    }


# ── Main pipeline ────────────────────────────────────────────────────

def _analyze_model(model):
    """Top-level worker function (must be picklable for ProcessPoolExecutor)."""
    model_id = model.get("id", "")
    result = analyze_modelcard(model.get("modelcard", ""), model_id=model_id)
    result["id"] = model_id
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel worker processes (default: 4)")
    args = parser.parse_args()

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(_analyze_model, data),
            total=len(data),
            desc="Analyzing model cards",
        ))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\nResults saved to {output_path}")
    print(f"{'Model':50s} {'Chip':12s} {'Conf':6s}  All chips")
    print("-" * 100)
    for r in results:
        chip = r.get("chip_provider", "unknown")
        chip_conf = r.get("chip_provider_confidence", 0)
        all_chips = r.get("chip_providers_all", {})
        print(f"  {r['id']:48s} {chip:12s} {chip_conf:<6.2f}  {all_chips}")


if __name__ == "__main__":
    main()
