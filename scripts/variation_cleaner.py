"""Group HF model_ids into families and pick one representative per family.

Goal: when sampling top-N popular models, we want one row per *model family*
rather than per checkpoint. A family is e.g. `Qwen/Qwen3` — we don't want
`Qwen/Qwen3-7B-Instruct` AND `Qwen/Qwen3-14B-Instruct` AND
`Qwen/Qwen3-7B-Instruct-FP8` to all show up as separate rows; one of them
(typically the most-downloaded) is enough.

Family stem rules — strip in order:
  1. precision / quantization suffixes  (-FP8, -GGUF, -MLX, -AWQ, -Q4_K_M, -INT4, -8bit, -bf16, …)
  2. standard variant tags              (-Instruct, -Chat, -Base, -it, -ft, -SFT, -DPO,
                                         -RLHF, -Reasoning, -Thinking, -Reasoner,
                                         -Foundation, -Pretrained, -Preview, -Distill, …)
  3. date stamps                        (-2501, -20240701)
  4. parameter sizes                    (-7B, -0.5B, -110M, -E2B, -A3B — including MoE
                                         active-parameter form)
  5. trailing audio/video sample-rate   (-12Hz, -24Hz, -48kHz)

We deliberately PRESERVE product variants — `Qwen/Qwen3-VL`, `Qwen/Qwen3-Embedding`,
`Qwen/Qwen3-Coder`, `Qwen/Qwen3-Reranker` stay distinct from `Qwen/Qwen3` because
they're different products (vision-language, embeddings, code, reranker).

Usage:
    python scripts/variation_cleaner.py
        # reads database/models.csv
        # writes database/models_dedup.csv (most-downloaded representative per stem)
        # plus database/models_dedup_groups.json with the full grouping for inspection
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB = ROOT / "database"

# Allow huge fields (the modelcard column can be ~MB).
csv.field_size_limit(sys.maxsize)

# ── Stem rules ──────────────────────────────────────────────────────────────
# Each rule applies repeatedly until no further match (so multi-suffix names
# like `Qwen3-0.5B-Instruct-AWQ` collapse one suffix at a time).

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

# Date stamps:  -2501, -20240701, -250105 (no day)
_DATE_RE = re.compile(r"[-_](?:20\d{2}|2[0-9])\d{2,8}(?=$|[-_])")

# Parameter size: -7B, -1.5B, -110M, -E2B, -A3B (MoE active-param), -0.5b
# Allow optional letter prefix (E for "expert", A for "active").
_SIZE_RE = re.compile(
    r"[-_][A-Z]?\d+(?:\.\d+)?[BbMmKk](?=$|[-_])"
)

# Audio sample-rate suffix: -12Hz, -48kHz
_HZ_RE = re.compile(r"[-_]\d+(?:k?Hz|kbps)\b", re.IGNORECASE)


def _strip_repeated(rx: re.Pattern, s: str) -> str:
    """Apply rx.sub('', s) until it stops shrinking."""
    while True:
        new = rx.sub("", s)
        if new == s:
            return s
        s = new


def family_stem(model_id: str) -> str:
    """Return a normalized family key for a HuggingFace model_id.

    Same-family checkpoints collapse to the same stem; distinct products keep
    their own stems.
    """
    if "/" not in model_id:
        return model_id
    org, name = model_id.split("/", 1)

    s = name
    s = _strip_repeated(_PRECISION_RE, s)
    s = _strip_repeated(_VARIANT_RE, s)
    s = _strip_repeated(_DATE_RE, s)
    s = _strip_repeated(_SIZE_RE, s)
    s = _strip_repeated(_HZ_RE, s)
    # Final cleanup: trim trailing separators.
    s = re.sub(r"[-_.\s]+$", "", s)
    return f"{org}/{s}"


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    src = DB / "models.csv"
    if not src.exists():
        print(f"missing {src} — run scripts/ingest/get_models.py first", file=sys.stderr)
        sys.exit(1)

    rows = []
    with src.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            try:
                downloads = int(r.get("downloads") or 0)
            except ValueError:
                downloads = 0
            r["_downloads_int"] = downloads
            rows.append(r)

    # Group by stem; within each group keep the highest-download row.
    groups: dict[str, list[dict]] = {}
    for r in rows:
        stem = family_stem(r["id"])
        groups.setdefault(stem, []).append(r)

    representatives = []
    for stem, members in groups.items():
        members.sort(key=lambda r: -r["_downloads_int"])
        rep = members[0]
        representatives.append(rep)

    # Sort representatives by downloads (matching the source ordering).
    representatives.sort(key=lambda r: -r["_downloads_int"])

    # Write deduplicated CSV.
    out = DB / "models_dedup.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in representatives:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Also write the full groupings so the user can audit.
    summary_path = DB / "models_dedup_groups.json"
    summary = {
        stem: {
            "representative": members[0]["id"],
            "members": [m["id"] for m in sorted(members, key=lambda r: -r["_downloads_int"])],
            "downloads_top": members[0]["_downloads_int"],
            "n_members": len(members),
        }
        for stem, members in groups.items()
        if len(members) > 1
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"Read {len(rows)} models → {len(representatives)} family representatives "
        f"({len(rows) - len(representatives)} variants collapsed)."
    )
    print(f"  wrote {out}")
    print(f"  wrote {summary_path}  ({len(summary)} families with >1 variant)")


if __name__ == "__main__":
    main()
