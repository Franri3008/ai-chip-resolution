import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "llm"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "classifiers"))

from ask_llm_chip import _parse_answer


def test_llm_none_found_forces_unknown():
    answer = (
        "training_evidence: none found\n"
        "conclusion: nvidia\n"
        "confidence: high\n"
    )
    conclusion, confidence = _parse_answer(answer, card_text="irrelevant card text")
    assert conclusion is None
    assert confidence == 0.0


def test_llm_low_confidence_forces_unknown():
    card = "We trained on 8 H100 GPUs for 100 hours."
    answer = (
        f"training_evidence: {card}\n"
        "conclusion: nvidia\n"
        "confidence: low\n"
    )
    conclusion, confidence = _parse_answer(answer, card_text=card)
    assert conclusion is None
    assert confidence == 0.0


def test_llm_evidence_not_in_card_forces_unknown():
    """LLM fabricates a quote that isn't actually in the card — reject."""
    card = "The model card mentions nothing about training hardware."
    answer = (
        "training_evidence: trained on H100 GPUs\n"
        "conclusion: nvidia\n"
        "confidence: high\n"
    )
    conclusion, confidence = _parse_answer(answer, card_text=card)
    assert conclusion is None
    assert confidence == 0.0


def test_llm_valid_evidence_commits_to_chip():
    card = "We trained on 8 H100 GPUs for 100 hours on Meta's cluster."
    answer = (
        "training_evidence: We trained on 8 H100 GPUs for 100 hours\n"
        "conclusion: nvidia\n"
        "confidence: high\n"
    )
    conclusion, confidence = _parse_answer(answer, card_text=card)
    assert conclusion == "nvidia"
    assert confidence == 0.7


def test_llm_evidence_without_chip_literal_forces_unknown():
    """Evidence is a substring of the card but doesn't name a chip or training phrase → reject."""
    card = "The model is a transformer with 8B parameters."
    answer = (
        "training_evidence: The model is a transformer with 8B parameters\n"
        "conclusion: nvidia\n"
        "confidence: high\n"
    )
    conclusion, confidence = _parse_answer(answer, card_text=card)
    assert conclusion is None
    assert confidence == 0.0


def test_llm_medium_confidence_caps_at_0_5():
    card = "Model was fine-tuned on A100 80GB for 50 hours."
    answer = (
        "training_evidence: fine-tuned on A100 80GB for 50 hours\n"
        "conclusion: nvidia\n"
        "confidence: medium\n"
    )
    conclusion, confidence = _parse_answer(answer, card_text=card)
    assert conclusion == "nvidia"
    assert confidence == 0.5
