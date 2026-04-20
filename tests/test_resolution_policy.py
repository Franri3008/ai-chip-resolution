import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from main import resolve_initial_conclusion


def _unknown_analysis():
    return {
        "chip_provider": "unknown",
        "chip_provider_confidence": 0.0,
        "chip_snippets": [],
        "framework": "unknown",
        "framework_confidence": 0.0,
        "matched_sections": [],
        "detection_files": [],
    }


def test_low_trust_github_repo_is_ignored_for_chip():
    original = main.LOW_TRUST_GITHUB_CHIP_REPOS
    main.LOW_TRUST_GITHUB_CHIP_REPOS = {"example/broad-library"}
    try:
        mc = {"main_github": "https://github.com/example/broad-library", "main_arxiv": None, "main_arxiv_confidence": 0.0}
        mca = _unknown_analysis()
        gha = {
            "chip_provider": "nvidia",
            "chip_provider_confidence": 0.45,
            "chip_snippets": [
                {"snippet": "...Generate training data... torch.cuda.empty_cache()...", "file": "research/train_utils.py"},
            ],
            "framework": "pytorch",
            "framework_confidence": 1.0,
            "detection_files": ["research/train_utils.py"],
        }
        axa = _unknown_analysis()

        resolved = resolve_initial_conclusion(mc, mca, gha, axa)

        assert resolved["chip_provider"] == "unknown"
    finally:
        main.LOW_TRUST_GITHUB_CHIP_REPOS = original


def test_runtime_only_arxiv_benchmark_is_ignored():
    mc = {"main_github": None, "main_arxiv": "https://arxiv.org/abs/1234.5678", "main_arxiv_confidence": 0.86}
    mca = _unknown_analysis()
    gha = _unknown_analysis()
    axa = {
        "chip_provider": "nvidia",
        "chip_provider_confidence": 0.56,
        "chip_snippets": [
            {"snippet": "...average query latency ... one Tesla V100 GPU per query for neural re-rankers...", "section": "body"},
        ],
        "framework": "unknown",
        "framework_confidence": 0.0,
    }

    resolved = resolve_initial_conclusion(mc, mca, gha, axa)

    assert resolved["chip_provider"] == "unknown"


def test_explicit_training_paper_evidence_is_kept():
    mc = {"main_github": None, "main_arxiv": "https://arxiv.org/abs/1234.5678", "main_arxiv_confidence": 0.88}
    mca = _unknown_analysis()
    gha = _unknown_analysis()
    axa = {
        "chip_provider": "nvidia",
        "chip_provider_confidence": 0.48,
        "chip_snippets": [
            {"snippet": "...the model was trained on 8 V100 GPUs for approximately 90 hours...", "section": "body"},
        ],
        "framework": "unknown",
        "framework_confidence": 0.0,
    }

    resolved = resolve_initial_conclusion(mc, mca, gha, axa)

    assert resolved["chip_provider"] == "nvidia"
    assert resolved["chip_provider_source"] == "arxiv_paper"


def test_low_conf_modelcard_runtime_usage_is_ignored():
    mc = {"main_github": None, "main_arxiv": None, "main_arxiv_confidence": 0.0}
    mca = {
        "chip_provider": "nvidia",
        "chip_provider_confidence": 0.3,
        "chip_snippets": [
            {"snippet": "...pipeline.to(torch.device(\"cuda\"))..."},
        ],
        "framework": "pytorch",
        "framework_confidence": 0.4,
        "matched_sections": ["body"],
    }
    gha = _unknown_analysis()
    axa = _unknown_analysis()

    resolved = resolve_initial_conclusion(mc, mca, gha, axa)

    assert resolved["chip_provider"] == "unknown"


def test_explicit_modelcard_training_tpu_is_kept():
    mc = {"main_github": None, "main_arxiv": None, "main_arxiv_confidence": 0.0}
    mca = {
        "chip_provider": "google_tpu",
        "chip_provider_confidence": 1.0,
        "chip_snippets": [
            {"snippet": "...We trained our model on a TPU v3-8 during 100k steps..."},
        ],
        "framework": "pytorch",
        "framework_confidence": 0.85,
        "matched_sections": ["training"],
    }
    gha = _unknown_analysis()
    axa = _unknown_analysis()

    resolved = resolve_initial_conclusion(mc, mca, gha, axa)

    assert resolved["chip_provider"] == "google_tpu"
    assert resolved["chip_provider_source"] == "modelcard"


def test_thin_margin_does_not_flip_to_arxiv():
    mc = {"main_github": None, "main_arxiv": "https://arxiv.org/abs/1234.5678", "main_arxiv_confidence": 0.8}
    mca = {
        "chip_provider": "google_tpu",
        "chip_provider_confidence": 0.48,
        "chip_snippets": [
            {"snippet": "...we trained our model on a TPU v4-8 for 200k steps..."},
        ],
        "framework": "unknown",
        "framework_confidence": 0.0,
        "matched_sections": ["training"],
    }
    gha = _unknown_analysis()
    axa = {
        "chip_provider": "nvidia",
        "chip_provider_confidence": 0.52,
        "chip_snippets": [
            {"snippet": "...we trained on 8 A100 GPUs for 100 hours..."},
        ],
        "framework": "unknown",
        "framework_confidence": 0.0,
    }

    resolved = resolve_initial_conclusion(mc, mca, gha, axa)

    # ax margin over mc is 0.04 < 0.15 → arxiv does not win primary
    # Falls through to modelcard default
    assert resolved["chip_provider"] == "google_tpu"
    assert resolved["chip_provider_source"] == "modelcard"
    # Disagreeing arxiv signal (>= 0.3) triggers conflict penalty
    assert resolved["source_conflict"] is True
    assert resolved["chip_provider_confidence"] <= 0.55


def test_source_conflict_lowers_confidence():
    mc = {"main_github": "https://github.com/example/repo", "main_arxiv": None, "main_arxiv_confidence": 0.0}
    mca = {
        "chip_provider": "google_tpu",
        "chip_provider_confidence": 0.9,
        "chip_snippets": [
            {"snippet": "...we trained on TPU v4 pods for 300k steps..."},
        ],
        "framework": "unknown",
        "framework_confidence": 0.0,
        "matched_sections": ["training"],
    }
    gha = {
        "chip_provider": "nvidia",
        "chip_provider_confidence": 0.6,
        "chip_snippets": [
            {"snippet": "...we trained on 8 A100 GPUs for 100 hours..."},
        ],
        "framework": "unknown",
        "framework_confidence": 0.0,
        "detection_files": ["scripts/train.py"],
    }
    axa = _unknown_analysis()

    resolved = resolve_initial_conclusion(mc, mca, gha, axa)

    # gh=0.6 vs mc=0.9: gh primary requires gh_conf - mc_conf >= 0.15, fails.
    # mc wins as default. Conflict (gh=nvidia 0.6) caps confidence at 0.55.
    assert resolved["chip_provider"] == "google_tpu"
    assert resolved["source_conflict"] is True
    assert resolved["chip_provider_confidence"] <= 0.55


def test_low_floor_arxiv_still_returned_via_fallback():
    mc = {"main_github": None, "main_arxiv": "https://arxiv.org/abs/1234.5678", "main_arxiv_confidence": 0.8}
    mca = _unknown_analysis()
    gha = _unknown_analysis()
    axa = {
        "chip_provider": "nvidia",
        "chip_provider_confidence": 0.35,
        "chip_snippets": [
            {"snippet": "...we trained on 4 V100 GPUs for approximately 20 hours..."},
        ],
        "framework": "unknown",
        "framework_confidence": 0.0,
    }

    resolved = resolve_initial_conclusion(mc, mca, gha, axa)

    # ax < MIN_PREFER_ARXIV (0.5) → skips primary branch
    # Falls through: mc unknown, gh unknown, ax != unknown → ax wins at raw 0.35
    assert resolved["chip_provider"] == "nvidia"
    assert resolved["chip_provider_source"] == "arxiv_paper"
    assert resolved["chip_provider_confidence"] == 0.35
    assert resolved["source_conflict"] is False


def test_training_disclosure_cap_limits_heuristic_confidence():
    mc = {"main_github": None, "main_arxiv": None, "main_arxiv_confidence": 0.0}
    mca = {
        "chip_provider": "nvidia",
        "chip_provider_confidence": 0.8,
        "chip_snippets": [
            {"snippet": "...model supports CUDA acceleration via TensorRT backend..."},
        ],
        "framework": "pytorch",
        "framework_confidence": 0.9,
        "matched_sections": ["body"],
    }
    gha = _unknown_analysis()
    axa = _unknown_analysis()

    resolved = resolve_initial_conclusion(mc, mca, gha, axa)

    # No snippet contains explicit training-disclosure language
    # → confidence capped at 0.6 before aggregation
    assert resolved["chip_provider"] == "nvidia"
    assert resolved["chip_provider_confidence"] <= 0.6


def test_disclosure_cap_requires_colocation_not_just_any_snippet():
    """A 'we trained on X' snippet without a chip name must NOT lift the cap for a
    separate runtime-only H100 snippet. Co-location is required in the same snippet."""
    from signals import apply_training_disclosure_cap

    snippets = [
        # Hardware mention — but in a compatibility/support context only.
        {"snippet": "The model fits into a single 80GB GPU (like NVIDIA H100 or AMD MI300X)."},
        # Disclosure phrase — but mentions dataset, not chip.
        {"snippet": "We trained on a curated corpus of 2T tokens."},
    ]
    # Start from an above-cap confidence; should be clamped to 0.6.
    assert apply_training_disclosure_cap(0.8, snippets) == 0.6


def test_hardware_duration_snippet_lifts_cap():
    """Kokoro-style single snippet with hardware + duration keyword lifts the cap."""
    from signals import apply_training_disclosure_cap

    snippets = [
        {"snippet": "Training Cost: About $1000 for 1000 hours of A100 80GB vRAM."},
    ]
    # Co-location of A100 + hours → cap should not fire.
    assert apply_training_disclosure_cap(0.8, snippets) == 0.8


def test_colocated_training_disclosure_lifts_cap():
    """A single snippet naming both a training phrase and a chip literal lifts the cap."""
    from signals import apply_training_disclosure_cap

    snippets = [
        {"snippet": "We trained the model on 8 H100 GPUs for 120 hours."},
    ]
    assert apply_training_disclosure_cap(0.8, snippets) == 0.8


def test_conditional_fine_tuned_does_not_lift_cap():
    """\"can be fine-tuned on H100\" is a suggestion about user fine-tuning, not a
    disclosure about the training of the model itself. The cap must still fire."""
    from signals import apply_training_disclosure_cap, has_explicit_training_chip_evidence

    snippets = [
        {"snippet": "The larger gpt-oss-120b can be fine-tuned on a single H100 node."},
    ]
    assert has_explicit_training_chip_evidence(snippets) is False
    assert apply_training_disclosure_cap(0.8, snippets) == 0.6


def test_capped_confidence_routes_to_llm_trigger():
    """resolve_initial_conclusion passes through ~0.6; main trigger logic (not tested
    here) promotes exactly-capped values to needs_llm=True. Verify the cap value
    propagates unchanged so the trigger has something to match on."""
    mc = {"main_github": None, "main_arxiv": None, "main_arxiv_confidence": 0.0}
    mca = {
        "chip_provider": "nvidia",
        "chip_provider_confidence": 0.68,
        "chip_snippets": [
            {"snippet": "...fit into a single 80GB GPU (like NVIDIA H100 or AMD MI300X)..."},
        ],
        "framework": "pytorch",
        "framework_confidence": 0.9,
        "matched_sections": ["body"],
    }
    gha = _unknown_analysis()
    axa = _unknown_analysis()

    resolved = resolve_initial_conclusion(mc, mca, gha, axa)

    assert resolved["chip_provider"] == "nvidia"
    # Cap fired; value lands at TRAINING_DISCLOSURE_CAP.
    assert abs(resolved["chip_provider_confidence"] - 0.6) < 0.01
