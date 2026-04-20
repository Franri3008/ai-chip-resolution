import re

from llm_client import complete_async

VALID_CHIPS = {
    "nvidia", "amd", "intel", "google_tpu", "apple", "aws", "qualcomm",
}

CONFIDENCE_MAP = {
    "high": 0.7,
    "medium": 0.5,
    "low": 0.0,  # low → force unknown
}

# Specific chip names that must appear verbatim in training_evidence to commit to a chip.
_CHIP_LITERAL_RE = re.compile(
    r'\b(?:A100|H100|V100|H200|P100|T4|L40[Ss]?|A6000|A10[Gg]?|DGX|'
    r'MI\d{3}[Xx]?|Instinct|Gaudi|Habana|Xeon|'
    r'TPU(?:\s*v\d+)?|Trainium|Inferentia|'
    r'M[1-4](?:\s*(?:Pro|Max|Ultra))?|'
    r'Snapdragon|Hexagon|Nvidia|CUDA|ROCm|XLA|MLX|CoreML|OpenVINO|'
    r'trained\s+on|training\s+(?:hardware|infrastructure|utilized)|'
    r'GPU\s+hours?|fine[- ]?tun(?:ed|ing)\s+on|pre[- ]?train(?:ed|ing)\s+on)\b',
    re.IGNORECASE,
)


def _build_prompt(model_name, yaml_library, framework, section_labeled_text, modelcard_excerpt, extra_context):
    card_text = section_labeled_text[:5000] if section_labeled_text else modelcard_excerpt[:3000]
    return (
        f'Determine the hardware chip provider used to TRAIN the HuggingFace model "{model_name}".\n\n'
        f"Known facts:\n"
        f"- YAML library_name: {yaml_library or 'not specified'}\n"
        f"- Detected framework: {framework or 'unknown'}\n"
        f"{extra_context}\n\n"
        f"=== MODEL CARD (with section labels) ===\n{card_text}\n\n"
        f"CRITICAL DISTINCTIONS:\n"
        f"- library_name values 'mlx', 'coreml', 'openvino', 'onnx' indicate an inference/conversion "
        f"runtime, NOT training hardware. These models are derivatives of another model.\n"
        f"- [COMPATIBILITY] or [BODY] sections saying 'compatible with TPU/JAX' or "
        f"'export to ONNX' = inference/export support, NOT training hardware\n"
        f"- [TRAINING] sections saying 'trained on 8xA100' or 'TPU v4 pod' = actual training hardware\n"
        f"- Generic mentions of 'GPU' or 'CUDA' without training context are weak signals\n\n"
        f"HARD RULE — return `unknown` by default:\n"
        f"If you cannot quote a sentence from the card that names a SPECIFIC chip "
        f"(A100/H100/V100/H200/T4/L40/DGX, MI250/MI300/Gaudi/Habana, TPU vN, Trainium/Inferentia, "
        f"M1/M2/M3/M4, Snapdragon/Hexagon) OR uses an explicit training phrase "
        f"(\"trained on\", \"fine-tuned on\", \"pre-trained on\", \"training hardware\", \"GPU hours\"), "
        f"you MUST answer:\n"
        f"  training_evidence: none found\n"
        f"  conclusion: unknown\n"
        f"  confidence: low\n"
        f"Do NOT infer a chip from framework, library_name, dependencies, or generic GPU/CUDA mentions. "
        f"Do NOT guess nvidia as a default. If in doubt, answer unknown.\n\n"
        f"Reply in EXACTLY this format (3 lines):\n"
        f"training_evidence: <direct verbatim quote from the model card text above, or 'none found'>\n"
        f"conclusion: <one of: {', '.join(sorted(VALID_CHIPS))}, unknown>\n"
        f"confidence: <high, medium, or low>\n"
    )


def _norm(text):
    return re.sub(r'\s+', ' ', text.strip().lower()) if text else ""


def _parse_answer(answer, card_text=""):
    conclusion = None
    confidence_label = "low"
    training_evidence = ""

    for line in answer.split("\n"):
        raw = line.strip()
        lower = raw.lower()
        if lower.startswith("training_evidence:"):
            training_evidence = raw.split(":", 1)[1].strip().strip("'\"")
        elif lower.startswith("conclusion:"):
            value = lower.split(":", 1)[1].strip()
            if value in VALID_CHIPS:
                conclusion = value
            elif value == "unknown":
                conclusion = None
        elif lower.startswith("confidence:"):
            confidence_label = lower.split(":", 1)[1].strip()

    if conclusion is None:
        single = answer.strip().lower()
        if single in VALID_CHIPS:
            conclusion = single

    confidence = CONFIDENCE_MAP.get(confidence_label, 0.0)

    # Enforcement: require a valid training_evidence quote to commit to any chip.
    ev_norm = _norm(training_evidence)
    card_norm = _norm(card_text)
    if conclusion is not None:
        if (not ev_norm
                or ev_norm in {"none found", "none", "n/a", "na", "not specified"}
                or not _CHIP_LITERAL_RE.search(training_evidence)
                or (card_norm and ev_norm not in card_norm)):
            conclusion = None
            confidence = 0.0

    if confidence_label == "low":
        conclusion = None
        confidence = 0.0

    return conclusion, confidence


async def ask_llm_chip(
    model_name,
    yaml_library=None,
    framework=None,
    modelcard_excerpt="",
    section_labeled_text="",
    extra_context="",
):
    """Ask the configured LLM to determine the training chip provider.

    Returns (chip_provider_str_or_None, confidence_float, cost_float).
    """
    prompt = _build_prompt(
        model_name, yaml_library, framework,
        section_labeled_text, modelcard_excerpt, extra_context,
    )
    answer, cost = await complete_async([{"role": "user", "content": prompt}])
    card_text = section_labeled_text or modelcard_excerpt or ""
    conclusion, confidence = _parse_answer(answer, card_text=card_text)
    return conclusion, confidence, cost
