import argparse
import concurrent.futures
import json
import os
import re

from tqdm import tqdm

from signals import (
    HARDWARE_SIGNALS, FRAMEWORK_SIGNALS,
    CHIP_PROVIDERS, FRAMEWORKS, MIN_SCORE_THRESHOLD, CONFIDENCE_DIVISOR,
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

# Heading patterns → section type
_TRAINING_HEADINGS = re.compile(
    r'#+\s*(?:training|hardware|infrastructure|compute|setup)',
    re.IGNORECASE,
)
_COMPAT_HEADINGS = re.compile(
    r'#+\s*(?:compatib|supported?\s+hardware|requirements?|installation|deploy|usage|how\s+to\s+(?:use|run))',
    re.IGNORECASE,
)
_REF_HEADINGS = re.compile(
    r'#+\s*(?:citation|references?|acknowledg|bibtex|license|paper)',
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
    r'(?:trained?\s+(?:on|with|using)|training\s+(?:on|with|hardware|infrastructure|setup|cluster)|'
    r'fine[- ]?tun(?:ed|ing)\s+(?:on|with|using)|pre[- ]?train(?:ed|ing)\s+(?:on|with)|'
    r'(?:\d+\s*[x×]\s*)?(?:A100|H100|V100|TPU\s*v\d|P100|T4|MI\d{3}|Gaudi)\s*.*?(?:hours?|days?|weeks?)|'
    r'compute\s+(?:budget|resources?|infrastructure)|'
    r'(?:hours?|days?|weeks?)\s+(?:of\s+)?(?:training|compute)|'
    r'training\s+(?:was\s+)?(?:done|performed|conducted|run)\s+(?:on|using|with))',
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

# ── YAML frontmatter keywords → signals ──────────────────────────────

_YAML_LIBRARY_MAP = {
    "pytorch": ("pytorch", 8),
    "torch": ("pytorch", 8),
    "timm": ("pytorch", 8),
    "transformers": ("pytorch", 8),
    "tensorflow": ("tensorflow", 8),
    "tf": ("tensorflow", 8),
    "jax": ("jax", 8),
    "flax": ("jax", 6),
    "paddlepaddle": ("paddlepaddle", 8),
    "paddle": ("paddlepaddle", 8),
    "sentence-transformers": ("pytorch", 8),
    "onnx": ("onnx", 6),
    "mlx": ("pytorch", 3),       # inference runtime; MLX models are PyTorch conversions
    "coreml": ("pytorch", 3),    # inference runtime; CoreML models are PyTorch conversions
    "openvino": ("pytorch", 3),  # inference runtime; OpenVINO models are PyTorch conversions
}

# Framework → implied default chip provider (for YAML authority chip capping)
_FRAMEWORK_CHIP_MAP = {
    "pytorch": "nvidia",
    "tensorflow": "nvidia",
    "jax": "google_tpu",
    "paddlepaddle": "nvidia",
    "mxnet": "nvidia",
}

_YAML_TAG_KEYWORDS = {
    # Chip providers
    "tpu": ("google_tpu", 6),
    "cuda": ("nvidia", 5),
    "gpu": ("nvidia", 2),
    "rocm": ("amd", 6),
    "gaudi": ("intel", 6),
    "habana": ("intel", 6),
    "inferentia": ("aws", 6),
    "trainium": ("aws", 6),
    # Frameworks
    "pytorch": ("pytorch", 5),
    "tensorflow": ("tensorflow", 5),
    "jax": ("jax", 5),
    "flax": ("jax", 4),
    "onnx": ("onnx", 4),
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


def detect_derivative(yaml_text, model_id):
    """Detect if a model is a derivative (quantized/converted) model.

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

    is_derivative = bool(runtime_library) or (name_is_derivative and base_model is not None)

    return is_derivative, base_model, runtime_library


def extract_yaml_signals(yaml_text):
    """Score structured YAML fields (library_name, tags).

    Returns (scores_dict, yaml_framework_or_None).
    """
    scores = {}
    yaml_lower = yaml_text.lower()
    yaml_framework = None

    # library_name
    m = re.search(r'library_name:\s*(\S+)', yaml_lower)
    if m:
        lib = m.group(1).strip()
        if lib in _YAML_LIBRARY_MAP:
            key, pts = _YAML_LIBRARY_MAP[lib]
            scores[key] = scores.get(key, 0) + pts
            if key in FRAMEWORKS:
                yaml_framework = key

    # tags (could be YAML list or inline)
    for tag_match in re.finditer(r'(?:^|\n)\s*-\s+(\S+)', yaml_text):
        tag = tag_match.group(1).strip().lower()
        if tag in _YAML_TAG_KEYWORDS:
            key, pts = _YAML_TAG_KEYWORDS[tag]
            scores[key] = scores.get(key, 0) + pts

    # Also scan tags: [...] inline format
    inline = re.search(r'tags:\s*\[([^\]]+)\]', yaml_lower)
    if inline:
        for tag in inline.group(1).split(","):
            tag = tag.strip().strip("'\"")
            if tag in _YAML_TAG_KEYWORDS:
                key, pts = _YAML_TAG_KEYWORDS[tag]
                scores[key] = scores.get(key, 0) + pts

    return scores, yaml_framework


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


def check_context(text, match_start):
    """Check the context around a match for negation/comparison/export/training.

    Returns a multiplier: 0.0 (negated), 0.25 (export), 0.3 (comparative),
    2.0 (training), 1.0 (normal).
    """
    window_start = max(0, match_start - 80)
    window_end = min(len(text), match_start + 80)
    preceding = text[window_start:match_start]
    context = text[window_start:window_end]

    if _NEGATION_RE.search(preceding):
        return 0.0
    if _COMPARISON_RE.search(preceding):
        return 0.3
    if _EXPORT_RE.search(preceding):
        return 0.25
    if _TRAINING_RE.search(context):
        return 2.0
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

    all_signals = [
        (HARDWARE_SIGNALS, "hardware"),
        (FRAMEWORK_SIGNALS, "framework"),
    ]

    for signal_dict, signal_type in all_signals:
        for provider, signals in signal_dict.items():
            for level, base_weight in [("strong", 5), ("medium", 3), ("weak", 1)]:
                for pattern in signals.get(level, []):
                    for m in re.finditer(pattern, text, re.IGNORECASE):
                        ctx_mult = check_context(text, m.start())
                        if ctx_mult == 0.0:
                            continue
                        tbl_weight = get_line_weight(m.start())
                        final = base_weight * section_weight * ctx_mult * tbl_weight
                        scores[provider] = scores.get(provider, 0) + final
                        matched = True
                        
                        if signal_type == "hardware":
                            # Extract ~100 chars around the match
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
    """Analyze a single model card and return chip/framework scores."""
    if not modelcard_text or not modelcard_text.strip():
        return {
            "chip_provider": "unknown", "chip_provider_score": 0,
            "chip_provider_confidence": 0.0, "chip_providers_all": {},
            "framework": "unknown", "framework_score": 0,
            "framework_confidence": 0.0, "frameworks_all": {},
            "matched_sections": [],
        }

    yaml_text, body = parse_yaml_frontmatter(modelcard_text)
    total_scores = {}
    matched_sections = []
    chip_snippets = []

    # 0. Detect derivative/converted model
    is_derivative, base_model, runtime_library = detect_derivative(yaml_text, model_id)

    # 1. YAML frontmatter structured signals
    yaml_scores, yaml_framework = extract_yaml_signals(yaml_text)
    for key, sc in yaml_scores.items():
        total_scores[key] = total_scores.get(key, 0) + sc
    if yaml_scores:
        matched_sections.append("yaml_frontmatter")

    # 2. Also regex-scan the YAML text itself (catches things like "tpu" in tags lists)
    yaml_scan_scores, yaml_matched, yaml_snips = scan_section(yaml_text, "yaml_frontmatter")
    for key, sc in yaml_scan_scores.items():
        total_scores[key] = total_scores.get(key, 0) + sc
    if yaml_matched and "yaml_frontmatter" not in matched_sections:
        matched_sections.append("yaml_frontmatter")
    chip_snippets.extend(yaml_snips)

    # 3. Split body into sections and scan each
    sections = split_into_sections(body)
    for section_type, section_text in sections:
        sec_scores, sec_matched, sec_snips = scan_section(section_text, section_type)
        # Derivative models: discount all chip scores from body text since the
        # card describes the conversion/inference process, not training.
        if is_derivative:
            for key in list(sec_scores):
                if key in CHIP_PROVIDERS:
                    sec_scores[key] = sec_scores[key] * 0.25
        for key, sc in sec_scores.items():
            total_scores[key] = total_scores.get(key, 0) + sc
        if sec_matched and section_type not in matched_sections:
            matched_sections.append(section_type)
        chip_snippets.extend(sec_snips)

    # 4. Split into chips and frameworks, round scores
    chip_scores = {k: round(v, 1) for k, v in total_scores.items() if k in CHIP_PROVIDERS and v > 0}
    fw_scores = {k: round(v, 1) for k, v in total_scores.items() if k in FRAMEWORKS and v > 0}

    # 5. YAML library_name authority: cap competing frameworks
    #    Anchor on the YAML-declared base score (not inflated by body text).
    #    Also ensure the YAML-declared framework gets at least cap+1 so it
    #    always wins over capped competitors.
    if yaml_framework and yaml_framework in fw_scores:
        yaml_base = yaml_scores.get(yaml_framework, 0)
        cap = max(yaml_base * 2, 10)
        for fw_name in list(fw_scores):
            if fw_name != yaml_framework:
                fw_scores[fw_name] = min(fw_scores[fw_name], cap)
        # Ensure YAML framework wins over capped competitors
        fw_scores[yaml_framework] = max(fw_scores[yaml_framework], cap + 1)

    # We NO LONGER cap competing chip providers based on framework. 
    # Explicit chip scores are allowed to accumulate their native tallies.

    # Determine top chip provider
    sorted_chips = sorted(chip_scores.items(), key=lambda x: -x[1])
    if sorted_chips and sorted_chips[0][1] >= MIN_SCORE_THRESHOLD:
        top_chip_name, top_chip_sc = sorted_chips[0]
        chip_conf = min(1.0, round(top_chip_sc / CONFIDENCE_DIVISOR, 2))
    else:
        top_chip_name, top_chip_sc, chip_conf = "unknown", 0, 0.0

    # Determine top framework
    sorted_fw = sorted(fw_scores.items(), key=lambda x: -x[1])
    if sorted_fw and sorted_fw[0][1] >= MIN_SCORE_THRESHOLD:
        top_fw_name, top_fw_sc = sorted_fw[0]
        fw_conf = min(1.0, round(top_fw_sc / CONFIDENCE_DIVISOR, 2))
    else:
        top_fw_name, top_fw_sc, fw_conf = "unknown", 0, 0.0

    # Build section-labeled text for LLM consumption
    labeled_sections = []
    if yaml_text.strip():
        labeled_sections.append(f"=== [YAML FRONTMATTER] ===\n{yaml_text.strip()}")
    for section_type, section_text in sections:
        label = section_type.upper().replace("_", " ")
        trimmed = section_text.strip()
        if trimmed:
            labeled_sections.append(f"=== [{label}] ===\n{trimmed}")
    section_labeled_text = "\n\n".join(labeled_sections)

    return {
        "chip_provider": top_chip_name,
        "chip_provider_score": top_chip_sc,
        "chip_provider_confidence": chip_conf,
        "chip_providers_all": dict(sorted_chips),
        "framework": top_fw_name,
        "framework_score": top_fw_sc,
        "framework_confidence": fw_conf,
        "frameworks_all": dict(sorted_fw),
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
    parser.add_argument("--workers", type=int, default=4,
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
