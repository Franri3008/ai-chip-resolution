"""Bundle the current pipeline output into a self-contained snapshot for
recovery / sharing. Writes a `database/snapshot_<label>/` directory with:

    results.json                  ← final per-model classifications
    modelcards.json               ← cached HF cards for every processed model
    modelcard_chip_analysis.json  ← per-card chip score breakdown
    github_chip_analysis.json     ← per-github score breakdown
    arxiv_chip_analysis.json      ← per-paper score breakdown
    SUMMARY.txt                   ← chip / source / year distributions
    RUN_COMMAND.txt               ← the run command needed to reproduce

Usage:
    python scripts/save_snapshot.py                # default label = today's date
    python scripts/save_snapshot.py --label v1     # custom label
"""

import argparse
import datetime as dt
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB = ROOT / "database"

ARTIFACTS = [
    "results.json",
    "modelcards.json",
    "modelcard_chip_analysis.json",
    "github_chip_analysis.json",
    "arxiv_chip_analysis.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default=dt.date.today().isoformat(),
        help="Snapshot directory suffix (default: YYYY-MM-DD).",
    )
    parser.add_argument(
        "--cmd",
        default="python main.py --quarters 2022-2026 --top 50 "
        "--source-csv database/models_dedup.csv "
        "--workers 16 --llm --provider LOCAL",
        help="Run command to record in RUN_COMMAND.txt for reproduction.",
    )
    args = parser.parse_args()

    out = DB / f"snapshot_{args.label}"
    out.mkdir(parents=True, exist_ok=True)

    missing = []
    for f in ARTIFACTS:
        src = DB / f
        if not src.exists():
            missing.append(f)
            continue
        shutil.copy2(src, out / f)

    if missing:
        print(f"warning: missing artifacts: {missing}", file=sys.stderr)

    # Build SUMMARY.txt
    results_path = out / "results.json"
    if not results_path.exists():
        print("no results.json — nothing to summarise", file=sys.stderr)
        sys.exit(1)

    with results_path.open(encoding="utf-8") as f:
        results = json.load(f)

    chip_counts = Counter(r["conclusion"]["chip_provider"] for r in results)
    src_counts = Counter(r["conclusion"].get("chip_provider_source", "-") for r in results)
    year_counts = Counter(r.get("year") for r in results)
    fw_counts = Counter(r["conclusion"].get("framework", "unknown") for r in results)

    lines = [
        f"# Snapshot {args.label}",
        f"# generated {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Total models: {len(results)}",
        "",
        "## Chip distribution",
        *(f"  {chip:20s}  {n}" for chip, n in chip_counts.most_common()),
        "",
        "## Decision source distribution",
        *(f"  {src or '(none)':20s}  {n}" for src, n in src_counts.most_common()),
        "",
        "## Framework distribution",
        *(f"  {fw:20s}  {n}" for fw, n in fw_counts.most_common()),
        "",
        "## Year distribution",
        *(f"  {y or '(none)':<6}  {n}" for y, n in sorted(year_counts.items(), key=lambda x: x[0] or 0)),
        "",
    ]
    (out / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "RUN_COMMAND.txt").write_text(args.cmd + "\n", encoding="utf-8")

    print(f"Snapshot saved to {out}")
    print((out / "SUMMARY.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
