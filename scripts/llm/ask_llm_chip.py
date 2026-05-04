import os
import re
import sys

from llm_client import complete_async

# Allow calling from arbitrary entry points: ensure classifiers/ is importable.
_CLASSIFIERS = os.path.join(os.path.dirname(__file__), "..", "classifiers")
if _CLASSIFIERS not in sys.path:
    sys.path.insert(0, _CLASSIFIERS)
from signals import snippet_is_training_disclosure, snippet_is_training_context  # noqa: E402

VALID_CHIPS = {
    "nvidia", "amd", "intel", "google_tpu", "apple", "aws", "qualcomm",
    "huawei_ascend", "cambricon",
    "baidu_kunlun", "moore_threads", "iluvatar", "hygon", "metax",
}

CONFIDENCE_MAP = {
    "high": 0.8,
    "medium": 0.6,
    "low": 0.4,
}

# Quote validity: must contain a chip name or an explicit training-language token.
_CHIP_OR_TRAINING_RE = re.compile(
    r'\b(?:A100|H100|H800|A800|V100|H200|P100|T4|L40[Ss]?|A6000|A10[Gg]?|DGX|'
    r'MI\d{3}[Xx]?|Instinct|Gaudi|Habana|Xeon|'
    r'TPU(?:\s*v\d+)?|Trainium|Inferentia|Neuron|'
    r'M[1-4](?:\s*(?:Pro|Max|Ultra))?|MLX|CoreML|OpenVINO|'
    # Chinese accelerators
    r'Ascend(?:\s*\d{3}[A-Da-d]?)?|910[ABCDabcd]|Atlas\s*\d{3}|MindSpore|CANN|HCCL|'
    r'DaVinci|Cambricon|MLU\s*\d{3}|cnML|cnnl|BANGPy|'
    r'Kunlun(?:xin)?|P800|R[23]00|XPU|昆仑|'
    r'MUSA|MTT\s*S\d{3,4}|Moore[-_ ]?Threads|S4000|mthreads|'
    r'Iluvatar|BI[-_ ]?V?1\d{2}|CoreX|天数智芯|'
    r'Hygon|DCU|DTK|海光|MetaX|Muxi|MXMACA|沐曦|'
    r'trained\s+(?:on|using|with)|training\s+(?:hardware|infrastructure|utilized|cost[s]?|compute)|'
    r'GPU\s+hours?|NPU\s+hours?|fine[- ]?tun(?:ed|ing)\s+(?:on|using|with)|'
    r'pre[- ]?train(?:ed|ing)\s+(?:on|using|with)|compute\s+cluster|'
    # Chinese disclosure phrasing common in TeleChat / Pangu cards
    r'国产算力|完全基于国产|domestic\s+(?:chinese\s+)?computing|昇腾|'
    # Training-launcher patterns — distributed-training invocations are legit
    # training evidence even without a specific chip name. The LLM is instructed
    # to map CUDA/torchrun -> nvidia, jax/xla -> tpu, rocm -> amd, mindspore/hccl -> ascend.
    r'torchrun|CUDA_VISIBLE_DEVICES|ASCEND_RT_VISIBLE_DEVICES|DistributedDataParallel|nccl|hccl|'
    r'accelerate\s+launch|deepspeed|mp\.spawn|--nproc[-_]?per[-_]?node|--num[-_]?gpus?|--num[-_]?npus?|'
    r'jax\.distributed|TPUStrategy|torch_xla|model\.cuda\(\)|model\.npu\(\)|'
    r'torch\.cuda|torch_npu|torch_mlu|mindspore\.set_context|'
    r'XPU_VISIBLE_DEVICES|MUSA_VISIBLE_DEVICES|DCU_VISIBLE_DEVICES|METAX_VISIBLE_DEVICES|'
    r'torch_musa|torch_dcu|xpurt|ixrt|mxmaca|paddle\.set_device)\b',
    re.IGNORECASE,
)

# Hypothetical / user-instruction language that should NOT be treated as disclosure.
_CONDITIONAL_RE = re.compile(
    r'\b(?:can|could|may|might|should|would|recommend(?:ed)?|if\s+you|users?\s+can|'
    r'allow[s]?\s+(?:you|users)|able\s+to|capable\s+of)\b[^.]{0,80}?'
    r'\b(?:be\s+)?(?:trained?|fine[- ]?tuned|pre[- ]?trained|run|deployed)\b',
    re.IGNORECASE,
)


def _filter_disclosure_snippets(snippets):
    """Surface snippets that contain a non-hypothetical training phrase. The LLM
    decides whether each quote is about *this* model's training vs a dataset or
    a user-fine-tuning suggestion. Strict chip-literal co-location would miss
    the common 'trained on N GPUs' / 'our cluster of X' phrasings."""
    if not snippets:
        return []
    kept = []
    for s in snippets:
        if snippet_is_training_disclosure(s) or snippet_is_training_context(s):
            kept.append(s)
    return kept


def _format_snippets(snippets, label, max_chars):
    if not snippets:
        return ""
    lines = []
    budget = max_chars
    seen = set()
    for s in snippets:
        text = s.get("snippet", "") if isinstance(s, dict) else str(s)
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        src = ""
        if isinstance(s, dict):
            src = s.get("file", "") or s.get("section", "")
        piece = f"- {text}" + (f"  [{src}]" if src else "")
        if len(piece) > budget:
            piece = piece[:budget]
        lines.append(piece)
        budget -= len(piece) + 1
        if budget <= 0:
            break
    if not lines:
        return ""
    return f"\n=== {label} ===\n" + "\n".join(lines)


def _build_prompt(model_name, yaml_library, section_labeled_text,
                  modelcard_excerpt, extra_context,
                  github_snippets=None, arxiv_snippets=None):
    # vLLM's 4096-token context means the full prompt must stay under ~12000 chars
    # (prompt skeleton ~800, instruction block ~1800, leaving ~9400 for card text).
    # Split the evidence budget: card gets the biggest slice, github + arxiv each get
    # a smaller slice. Training disclosures often live in github scripts (CUDA_VISIBLE_
    # _DEVICES, torchrun) or arXiv papers (H100 clusters), not in modelcards.
    card_text = section_labeled_text[:3500] if section_labeled_text else modelcard_excerpt[:2500]
    # Pre-filter snippets through the strict co-location check; raw top-K
    # snippets include runtime-only chip mentions that the LLM mis-reads.
    gh_filtered = _filter_disclosure_snippets(github_snippets)
    ax_filtered = _filter_disclosure_snippets(arxiv_snippets)
    gh_block = _format_snippets(gh_filtered, "EXTERNAL TRAINING DISCLOSURE (GITHUB)", 1500)
    ax_block = _format_snippets(ax_filtered, "EXTERNAL TRAINING DISCLOSURE (ARXIV)", 1200)
    return (
        f'You are classifying the hardware the model "{model_name}" was TRAINED on '
        f'(not the hardware a user runs it on).\n\n'
        f"Known facts:\n"
        f"- YAML library_name: {yaml_library or 'not specified'}\n"
        f"{extra_context}\n\n"
        f"=== MODEL CARD ===\n{card_text}"
        f"{gh_block}"
        f"{ax_block}\n\n"
        f"COUNTS AS TRAINING EVIDENCE (commit to a chip):\n"
        f"- A sentence or table cell naming a specific accelerator AND training context: "
        f"\"trained on 8xA100\", \"H100 GPU hours\", \"TPU v4-128 pod\", \"MI250X cluster\", "
        f"\"fine-tuned on H100\", \"trained on Ascend NPU\", \"2048 Ascend 910 processors\", "
        f"\"Atlas 800T A2 cluster\" — IF it refers to *this* model's own training.\n"
        f"- A cost/compute table row labelled \"Training Cost\", \"GPU hours\", \"NPU hours\", or "
        f"\"Training Factors\" that names a chip (e.g. \"A100 80GB GPU hours | 1000\", "
        f"\"6K Ascend NPUs\").\n"
        f"- An explicit \"Hardware and Software\" / \"Training Infrastructure\" / "
        f"\"国产化适配\" / \"国产算力\" section that names a chip.\n"
        f"- Training-script infrastructure that implies the chip family:\n"
        f"    * `torchrun --nproc-per-node=N`, `CUDA_VISIBLE_DEVICES=0,1,…`, `deepspeed train.py`, "
        f"`accelerate launch`, `model.cuda()`, `torch.nn.parallel.DistributedDataParallel`, "
        f"`nccl` backend — these are NVIDIA CUDA training and count as `nvidia`.\n"
        f"    * `jax.distributed`, `flax`, `optax`, `torch_xla`, `TPUStrategy` — these are "
        f"Google TPU training and count as `google_tpu`.\n"
        f"    * `rocm`, `hipify`, `MI250`/`MI300` — AMD training → `amd`.\n"
        f"    * `mindspore`, `mindspore.set_context(device_target='Ascend')`, `HCCL` backend, "
        f"`ASCEND_RT_VISIBLE_DEVICES`, `torch_npu`, `npu-smi`, `model.npu()` — Huawei Ascend "
        f"training → `huawei_ascend`. MindSpore is Huawei's framework and runs only on Ascend "
        f"in production.\n"
        f"    * `torch_mlu`, `cnnl`, `cndrv`, `BANGPy`, `MLUDevice`, `model.mlu()` — Cambricon "
        f"training → `cambricon`.\n"
        f"    * `paddle.set_device('xpu')`, `xpurt`, `XPU_VISIBLE_DEVICES`, `kunlunxin`, "
        f"\"trained on Kunlun P800\" / \"昆仑芯\" — Baidu Kunlun training → `baidu_kunlun`. "
        f"Note: bare `XPU` alone is ambiguous (Intel oneAPI also uses it); require "
        f"Kunlun/Baidu/Paddle/P800 context.\n"
        f"    * `torch_musa`, `MUSA_VISIBLE_DEVICES`, `mthreads`, `vllm-musa`, "
        f"\"MTT S4000\" / \"Moore Threads\" — Moore Threads training → `moore_threads`.\n"
        f"    * `ixrt`, `corex`, `Iluvatar`, \"BI-V100\" / \"BI-V150\" / \"天数智芯\" — "
        f"Iluvatar training → `iluvatar`.\n"
        f"    * `hy-smi`, `Hygon DCU`, `DTK`, `海光` (only when paired with DCU/DTK; bare "
        f"`DCU` alone is too generic) — Hygon training → `hygon`.\n"
        f"    * `mxmaca`, `mx-smi`, `METAX_VISIBLE_DEVICES`, `MetaX C500`, `Muxi`, "
        f"`沐曦` — MetaX training → `metax`.\n"
        f"  Phrases like \"完全基于国产算力训练\" / \"trained entirely on domestic Chinese computing "
        f"power\" combined with a MindSpore/Ascend mention count as `huawei_ascend`. \"国产算力\" "
        f"alone (no chip-vendor name) is ambiguous between Chinese vendors → return `unknown`.\n"
        f"  Note: H800 and A800 are NVIDIA SKUs (export-restricted variants of H100/A100); they "
        f"count as `nvidia`, even when used to train Chinese models.\n"
        f"  Only count launcher signals if they sit in this repo's training scripts, not in "
        f"a reference/optional inference snippet.\n\n"
        f"DOES NOT COUNT — return `unknown`:\n"
        f"- Inference / deployment / runtime mentions: \"runs on CUDA\", \"supports H100\", "
        f"\"compatible with TPU\", \"device_map='cuda:0'\", \"works on M2 Mac\".\n"
        f"- Hypothetical user fine-tuning: \"can be fine-tuned on H100\", "
        f"\"users may train on A100\", \"recommended: 4xH100\".\n"
        f"- A mention of a sibling/larger model's training hardware, unless this model "
        f"shares the same disclosure.\n"
        f"- Just a framework hint (\"transformers\", \"pytorch\") with no chip named.\n"
        f"- Generic \"GPU\" or \"CUDA\" without a specific chip or training quote.\n\n"
        f"DECIDING:\n"
        f"- If the MODEL CARD is silent AND the EXTERNAL DISCLOSURE blocks are "
        f"empty/absent, answer `unknown`.\n"
        f"- If any block contains a training-script import of `torch.distributed`, "
        f"`accelerate`, `deepspeed`, or a launcher (`torchrun`, `CUDA_VISIBLE_DEVICES`), "
        f"commit to `nvidia` at medium confidence — these frameworks run on NVIDIA "
        f"CUDA in practice. Similarly `jax.distributed`/`torch_xla` → `google_tpu`, "
        f"`mindspore.set_context(device_target='Ascend')`/`HCCL`/`ASCEND_RT_VISIBLE_DEVICES` "
        f"→ `huawei_ascend`, `torch_mlu`/`cnnl`/`MLUDevice` → `cambricon`, "
        f"`paddle.set_device('xpu')`/`xpurt`/Kunlun-P800 → `baidu_kunlun`, "
        f"`torch_musa`/MTT-S4000 → `moore_threads`, `ixrt`/CoreX/BI-V100 → `iluvatar`, "
        f"Hygon-DCU/DTK/`hy-smi` → `hygon`, MXMACA/`mx-smi`/MetaX-C500 → `metax`.\n"
        f"- Never default to `nvidia` purely from the framework hint — you need an "
        f"actual training-hardware or training-launcher signal.\n"
        f"- A `mindspore` YAML library_name or a MindSpore reference inside a TRAINING "
        f"section is a strong indicator of `huawei_ascend`. But a MindSpore link or "
        f"`ms-swift`/`ModelScope` reference inside an inference / deployment section is "
        f"NOT a training disclosure and should be ignored — return `unknown` if no "
        f"training-context Ascend/MindSpore quote exists.\n"
        f"- Do NOT promote to `huawei_ascend` based on the model being Chinese-authored "
        f"alone. Many Chinese labs (Alibaba/Qwen, DeepSeek, ZhipuAI/GLM) train on NVIDIA "
        f"H800/A800. Without a concrete Ascend / MindSpore / 国产算力 training quote, "
        f"return `unknown` rather than guessing.\n\n"
        f"You may quote from the MODEL CARD or from any EXTERNAL TRAINING DISCLOSURE "
        f"block shown above. The external blocks have been pre-filtered to contain only "
        f"co-located training+hardware language, so if one names a specific chip, trust "
        f"it as disclosure of this model's training hardware.\n\n"
        f"Reply in EXACTLY this format (3 lines, nothing else):\n"
        f"training_evidence: <short verbatim quote copy-pasted from any section above; "
        f"or 'none found' if no qualifying sentence/cell exists>\n"
        f"conclusion: <one of: {', '.join(sorted(VALID_CHIPS))}, unknown>\n"
        f"confidence: <high | medium | low>\n"
    )


def _norm(text):
    if not text:
        return ""
    # Lower, collapse whitespace, drop markdown table pipes / asterisks / backticks.
    t = text.strip().lower()
    t = re.sub(r'[`*|_>#]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t


_STOPWORDS = {
    "a", "an", "and", "the", "of", "on", "in", "to", "for", "with", "was",
    "is", "are", "we", "our", "by", "at", "be", "this", "that", "or", "from",
    "as", "it", "its", "per",
}


def _token_overlap(evidence, card_text):
    """Fraction of non-stopword content tokens from evidence that appear in card."""
    ev = _norm(evidence)
    card = _norm(card_text)
    if not ev or not card:
        return 0.0
    ev_tokens = [t for t in re.findall(r'\w+', ev) if t not in _STOPWORDS]
    if not ev_tokens:
        return 0.0
    card_tokens = set(re.findall(r'\w+', card))
    hits = sum(1 for t in ev_tokens if t in card_tokens)
    return hits / len(ev_tokens)


def _evidence_in_card(evidence, card_text):
    """Accept if (a) evidence is a substring of the normalized card, (b) any long
    run of the evidence appears verbatim, or (c) ≥80% of the evidence's content
    tokens appear in the card. (c) handles LLM quotes stitched from table cells
    (e.g. "in A100 80GB GPU hours | 1000" — real card has "... | 500 | 500 | 1000")."""
    ev = _norm(evidence)
    card = _norm(card_text)
    if not ev or not card:
        return False
    if ev in card:
        return True
    if len(ev) >= 25:
        for start in range(0, max(1, len(ev) - 25), 5):
            if ev[start:start + 30] in card:
                return True
    return _token_overlap(evidence, card_text) >= 0.8


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
            value = lower.split(":", 1)[1].strip().strip(".").strip()
            if value in VALID_CHIPS:
                conclusion = value
            elif value == "unknown":
                conclusion = None
        elif lower.startswith("confidence:"):
            confidence_label = lower.split(":", 1)[1].strip().strip(".").strip()

    if conclusion is None:
        single = answer.strip().lower()
        if single in VALID_CHIPS:
            conclusion = single

    confidence = CONFIDENCE_MAP.get(confidence_label, 0.0)

    if conclusion is None:
        return None, 0.0

    ev_lower = training_evidence.lower()
    no_ev = (
        not training_evidence
        or ev_lower in {"none found", "none", "n/a", "na", "not specified", "unknown", "not found"}
    )

    # Reject answers that commit to a chip without a concrete evidence quote.
    if no_ev:
        return None, 0.0

    # Reject quotes that contain only conditional / hypothetical phrasing about
    # users training ("can be fine-tuned on H100") — that's not a disclosure.
    if _CONDITIONAL_RE.search(training_evidence) and not re.search(
        r'\b(?:training\s+cost|gpu\s+hours?|training\s+utilized|training\s+factors?)\b',
        training_evidence, re.IGNORECASE,
    ):
        return None, 0.0

    # Reject quotes that don't contain any chip name or training phrase.
    if not _CHIP_OR_TRAINING_RE.search(training_evidence):
        return None, 0.0

    # Reject quotes the LLM fabricated (not in the card).
    if card_text and not _evidence_in_card(training_evidence, card_text):
        return None, 0.0

    return conclusion, confidence


async def ask_llm_chip(
    model_name,
    yaml_library=None,
    modelcard_excerpt="",
    section_labeled_text="",
    extra_context="",
    github_snippets=None,
    arxiv_snippets=None,
):
    """Ask the configured LLM to determine the training chip provider.

    github_snippets / arxiv_snippets are the chip-centered snippets surfaced by
    the respective classifiers (each item: {"snippet": str, "file"/"section": str}).

    Returns (chip_provider_str_or_None, confidence_float, cost_float).
    """
    prompt = _build_prompt(
        model_name, yaml_library,
        section_labeled_text, modelcard_excerpt, extra_context,
        github_snippets=github_snippets, arxiv_snippets=arxiv_snippets,
    )
    # Evidence for substring verification spans card + all snippets we passed in.
    corpus_parts = [section_labeled_text or modelcard_excerpt or ""]
    for s in (github_snippets or []) + (arxiv_snippets or []):
        corpus_parts.append(s.get("snippet", "") if isinstance(s, dict) else str(s))
    card_text = "\n".join(corpus_parts)
    answer, cost = await complete_async([{"role": "user", "content": prompt}])
    conclusion, confidence = _parse_answer(answer, card_text=card_text)
    return conclusion, confidence, cost
