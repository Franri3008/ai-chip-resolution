#!/usr/bin/env bash
# Sequentially runs the classifier on:
#   (i)  top 50 most-downloaded models per month, 2022-2026
#   (ii) top 10000 most-downloaded models ever
# Each run snapshots its outputs into database/runs/ with a unique suffix
# so the artifacts from the two runs do not overwrite each other.
#
# Heuristic-only (no --llm). Re-run a single classifier with --llm later
# if you want the LLM fallback for the unknowns.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=/opt/anaconda3/bin/python
RUNS_DIR=database/runs
LOGS_DIR=logs
mkdir -p "$RUNS_DIR" "$LOGS_DIR"

snapshot() {
    local suffix="$1"
    for f in modelcards.json results.json modelcard_chip_analysis.json \
             github_chip_analysis.json arxiv_chip_analysis.json; do
        if [[ -f "database/$f" ]]; then
            cp "database/$f" "$RUNS_DIR/${f%.json}_${suffix}.json"
        fi
    done
    echo "  Snapshot saved to $RUNS_DIR/*_${suffix}.json"
}

banner() {
    echo
    echo "================================================================"
    echo "$@"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
}

# ───────── Run (i): top 50 per month, 2022-2026 ─────────
banner "RUN 1/2  --top 50 --years 2022-2026  (up to 3,000 models)"
START1=$(date +%s)
$PY main.py --top 50 --years 2022-2026 --workers 16 \
    2>&1 | tee "$LOGS_DIR/run1_top50_per_month.log"
RC1=${PIPESTATUS[0]}
END1=$(date +%s)
echo "Run 1 exit code: $RC1   elapsed: $(( (END1-START1)/60 )) min"
snapshot "top50_per_month"

# ───────── Run (ii): top 10k ever ─────────
banner "RUN 2/2  --top 10000  (top 10,000 most-downloaded ever)"
START2=$(date +%s)
$PY main.py --top 10000 --workers 16 \
    2>&1 | tee "$LOGS_DIR/run2_top10k.log"
RC2=${PIPESTATUS[0]}
END2=$(date +%s)
echo "Run 2 exit code: $RC2   elapsed: $(( (END2-START2)/60 )) min"
snapshot "top10k"

banner "ALL RUNS COMPLETE"
echo "Run 1 (50/month, 2022-2026) : RC=$RC1   elapsed=$(( (END1-START1)/60 )) min"
echo "Run 2 (top 10k ever)        : RC=$RC2   elapsed=$(( (END2-START2)/60 )) min"
ls -la "$RUNS_DIR/"
