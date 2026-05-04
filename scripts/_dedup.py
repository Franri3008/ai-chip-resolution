"""Group HuggingFace model_ids into families.

A family is a stem like `Qwen/Qwen3` that collapses size, precision/quant,
fine-tune, and date-stamp variants into one key — so `Qwen/Qwen3-7B-Instruct`,
`Qwen/Qwen3-14B-Instruct-AWQ`, and `Qwen/Qwen3-0.5B-Base` all share the stem
`Qwen/Qwen3`. Distinct *products* are preserved: `Qwen/Qwen3-VL`,
`Qwen/Qwen3-Coder`, `Qwen/Qwen3-Embedding`, `Qwen/Qwen3-Reranker` keep their
own stems because they are different lines, not just size/precision rebuilds.

Stem rules — applied in order, each repeatedly until no further match:
  1. precision / quantization suffixes (-FP8, -GGUF, -AWQ, -Q4_K_M, -INT4, -8bit, …)
  2. variant tags                      (-Instruct, -Chat, -Base, -it, -ft, -SFT, -DPO,
                                        -Reasoning, -Distill, -Mini, -Lite, -Pro, …)
  3. date stamps                       (-2501, -20240701)
  4. parameter sizes                   (-7B, -0.5B, -110M, -E2B, -A3B …)
  5. trailing audio sample-rate        (-12Hz, -48kHz)
"""
from __future__ import annotations

import re

_PRECISION_RE = re.compile(
    r"[-_](?:"
    r"FP8|FP16|BF16|INT[48]|Q[1-8](?:_[KMSk0-9]+)*|"
    r"GGUF|MLX|AWQ|GPTQ|EXL2|exl[23]|"
    r"[48]bit|[48]_bit|fp8|fp16|bf16|int[48]"
    r")\b",
    re.IGNORECASE,
)

_VARIANT_RE = re.compile(
    r"[-_](?:"
    r"Instruct|Chat|Base|it|ft|SFT|DPO|ORPO|RLHF|RLAIF|GRPO|"
    r"Reasoning|Thinking|Reasoner|Foundation|Pretrained|Pretraining|"
    r"Preview|Distill|Distilled|Mini|Lite|Pro|Ultra|Max|"
    r"Original|Final|Latest|Stable|Beta|Alpha|RC\d*|"
    r"Hindi|English|Multilingual|Multi|"
    r"FineTuned|finetuned|Adapter|LoRA"
    r")(?=$|[-_])",
    re.IGNORECASE,
)

_DATE_RE = re.compile(r"[-_](?:20\d{2}|2[0-9])\d{2,8}(?=$|[-_])")

_SIZE_RE = re.compile(r"[-_][A-Z]?\d+(?:\.\d+)?[BbMmKk](?=$|[-_])")

_HZ_RE = re.compile(r"[-_]\d+(?:k?Hz|kbps)\b", re.IGNORECASE)


def _strip_repeated(rx: re.Pattern, s: str) -> str:
    while True:
        new = rx.sub("", s)
        if new == s:
            return s
        s = new


def family_stem(model_id: str) -> str:
    """Return a normalized family key for a HuggingFace model_id."""
    if "/" not in model_id:
        return model_id
    org, name = model_id.split("/", 1)

    s = name
    s = _strip_repeated(_PRECISION_RE, s)
    s = _strip_repeated(_VARIANT_RE, s)
    s = _strip_repeated(_DATE_RE, s)
    s = _strip_repeated(_SIZE_RE, s)
    s = _strip_repeated(_HZ_RE, s)
    s = re.sub(r"[-_.\s]+$", "", s)
    return f"{org}/{s}"


def deduplicate_rows(rows, id_field: str = "id"):
    """Yield one representative row per family stem.

    Assumes input is sorted by popularity descending, so the first row seen
    for each stem is the most-popular representative.

    Returns (deduped_rows, n_in, n_out). `deduped_rows` is a list.
    """
    seen: set[str] = set()
    out: list = []
    n_in = 0
    for row in rows:
        n_in += 1
        stem = family_stem(row[id_field])
        if stem in seen:
            continue
        seen.add(stem)
        out.append(row)
    return out, n_in, len(out)
