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
