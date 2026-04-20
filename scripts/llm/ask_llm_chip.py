from llm_client import complete_async

VALID_CHIPS = {
    "nvidia", "amd", "intel", "google_tpu", "apple", "aws", "qualcomm",
}

CONFIDENCE_MAP = {
    "high": 0.8,
    "medium": 0.6,
    "low": 0.4,
}


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
        f"- A library_name like 'sentence-transformers' or 'transformers' implies PyTorch/NVIDIA "
        f"unless the TRAINING section explicitly states otherwise\n"
        f"- Generic mentions of 'GPU' or 'CUDA' without training context are weak signals\n\n"
        f"Reply in EXACTLY this format (3 lines):\n"
        f"training_evidence: <direct quote from a TRAINING section, or 'none found'>\n"
        f"conclusion: <one of: {', '.join(sorted(VALID_CHIPS))}, unknown>\n"
        f"confidence: <high, medium, or low>\n"
    )


def _parse_answer(answer):
    conclusion = None
    confidence = 0.6

    for line in answer.split("\n"):
        line_lower = line.strip().lower()
        if line_lower.startswith("conclusion:"):
            value = line_lower.split(":", 1)[1].strip()
            if value in VALID_CHIPS:
                conclusion = value
        elif line_lower.startswith("confidence:"):
            value = line_lower.split(":", 1)[1].strip()
            confidence = CONFIDENCE_MAP.get(value, 0.6)

    if conclusion is None:
        single = answer.strip().lower()
        if single in VALID_CHIPS:
            conclusion = single

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
    conclusion, confidence = _parse_answer(answer)
    return conclusion, confidence, cost
