"""Regression tests for Chinese-chip detection (huawei_ascend / cambricon).

These cover the heuristic-only path:
- Ascend disclosures should be picked up from common phrasings (MindSpore, CANN,
  Atlas, 国产算力, "trained on Ascend").
- The DeepSeek-V3-style card (H800 training + recommended Ascend inference
  section) must NOT mistake the Ascend mentions for training disclosure.
- NVIDIA-China SKUs (H800/A800) remain `nvidia`.
"""

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "classifiers"))

from from_modelcard import analyze_modelcard  # noqa: E402


def _card(s):
    """Strip uniform leading whitespace so heading lines start with '#'."""
    return textwrap.dedent(s).strip()


def test_pangu_alpha_modelcard_resolves_huawei_ascend():
    card = _card("""
    ## PanGu-α Introduction

    PanGu-α is the first large-scale Chinese pre-trained language model with
    200 billion parameters trained on 2048 Ascend processors using an automatic
    hybrid parallel training strategy. The whole training process is done on
    the "Peng Cheng Cloud Brain II" computing platform with the domestic deep
    learning framework called MindSpore.

    The model is trained based on the domestic full-stack software and hardware
    ecosystem (MindSpore + CANN + Atlas910 + ModelArts).
    """)
    result = analyze_modelcard(card, "Hanlard/Pangu_alpha")
    assert result["chip_provider"] == "huawei_ascend"
    assert result["chip_provider_confidence"] >= 0.6


def test_telechat_modelcard_with_mindspore_link_resolves_ascend():
    card = _card("""
    ---
    license: apache-2.0
    ---
    # 星辰语义大模型-TeleChat2

    🏔 [MindSpore](https://gitee.com/mindspore/mindformers/tree/dev/research/telechat)

    - TeleChat2 是首个完全国产算力训练并开源的千亿参数模型。
    - 完全基于国产算力训练。
    """)
    result = analyze_modelcard(card, "Tele-AI/TeleChat2-115B")
    assert result["chip_provider"] == "huawei_ascend"


def test_deepseek_v3_style_card_keeps_nvidia_despite_ascend_inference_section():
    """Inference-only Ascend mentions under a recommended-inference heading
    must not outweigh an explicit H800 GPU-hours training disclosure.
    Mirrors the layout of the real deepseek-ai/DeepSeek-V3 card."""
    card = _card("""
    ## 1. Introduction

    DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training,
    which makes it remarkably economical. We pre-trained on H800 clusters with
    2048 H800 GPUs over the course of two months.

    ## 2. Pre-Training

    Training was performed on H800 GPUs. The full pre-training run consumed
    2.788 million H800 GPU hours across our cluster.

    ## 6. How to Run Locally

    DeepSeek-V3 can be deployed using NVIDIA GPUs (with vLLM or TRT-LLM) or
    Huawei Ascend NPUs.

    ### 6.7 Recommended Inference Functionality with Huawei Ascend NPUs

    The MindIE framework from the Huawei Ascend community has successfully
    adapted the BF16 version of DeepSeek-V3. For step-by-step guidance on
    Ascend NPUs, please follow the instructions linked here.
    """)
    result = analyze_modelcard(card, "deepseek-ai/DeepSeek-V3")
    assert result["chip_provider"] == "nvidia", (
        f"expected nvidia from H800 training disclosure, got "
        f"{result['chip_provider']} with scores {result['chip_providers_all']}"
    )


def test_cambricon_disclosure_resolves_cambricon():
    card = _card("""
    ---
    library_name: pytorch
    ---
    # MyModel

    ## Training

    We trained the model on a Cambricon MLU 590 cluster using the cnnl backend
    via torch_mlu. Training took 200 NPU hours.
    """)
    result = analyze_modelcard(card, "research/my-model")
    assert result["chip_provider"] == "cambricon"


def test_h800_keeps_nvidia_label_not_chinese_chip():
    """H800 is an NVIDIA SKU (export-restricted variant). Must remain nvidia
    even though it's only used inside China."""
    card = _card("""
    # Some Chinese Model

    ## Training

    We trained for 2.788M H800 GPU hours on a cluster of 2048 H800s in our
    Beijing datacenter.
    """)
    result = analyze_modelcard(card, "research/test-h800")
    assert result["chip_provider"] == "nvidia"
