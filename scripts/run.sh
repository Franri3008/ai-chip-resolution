#!/usr/bin/env bash
# Run main.py in one of three preset modes, snapshotting outputs to
# database/runs/ with a mode-specific suffix.
#
# Usage:
#   ./scripts/run.sh monthly [main.py args...]   # default: --top 50 --years 2022-2026
#   ./scripts/run.sh alltime [main.py args...]   # default: --top 10000
#   ./scripts/run.sh both    [monthly args -- alltime args]
#       (no args → defaults for both; '--' separates monthly from alltime args)
#
# Examples:
#   ./scripts/run.sh monthly --top 100 --years 2024 --workers 8
#   ./scripts/run.sh alltime --top 5000
#   ./scripts/run.sh both
#   ./scripts/run.sh both --top 100 --years 2024 -- --top 5000

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=${PY:-python}
RUNS_DIR=database/runs
LOGS_DIR=logs
mkdir -p "$RUNS_DIR" "$LOGS_DIR"

DEFAULT_MONTHLY="--top 50 --years 2022-2026 --workers 16"
DEFAULT_ALLTIME="--top 10000 --workers 16"

ARTIFACTS=(modelcards.json results.json modelcard_chip_analysis.json
           github_chip_analysis.json arxiv_chip_analysis.json)

snapshot() {
    local suffix=$1
    for f in "${ARTIFACTS[@]}"; do
        [[ -f database/$f ]] && cp "database/$f" "$RUNS_DIR/${f%.json}_${suffix}.json"
    done
    echo "  Snapshot → $RUNS_DIR/*_${suffix}.json"
}

extract_top() {
    # Echo the value passed to --top in the given args; "N" if not present.
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --top) echo "$2"; return;;
            --top=*) echo "${1#*=}"; return;;
        esac
        shift
    done
    echo "N"
}

run_main() {
    local label=$1 suffix=$2 logname=$3
    shift 3
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  $label"
    echo "  args: $*"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════"
    local t0=$(date +%s)
    $PY main.py "$@" 2>&1 | tee "$LOGS_DIR/$logname.log"
    local rc=${PIPESTATUS[0]}
    echo "  rc=$rc  elapsed=$(( ($(date +%s)-t0)/60 )) min"
    snapshot "$suffix"
    return "$rc"
}

monthly_run() {
    local args=("$@")
    [[ ${#args[@]} -eq 0 ]] && args=($DEFAULT_MONTHLY)
    local top=$(extract_top "${args[@]}")
    run_main "MONTHLY" "top${top}_per_month" "run_monthly" "${args[@]}"
}

alltime_run() {
    local args=("$@")
    [[ ${#args[@]} -eq 0 ]] && args=($DEFAULT_ALLTIME)
    local top=$(extract_top "${args[@]}")
    run_main "ALL-TIME" "top${top}" "run_alltime" "${args[@]}"
}

usage() {
    cat <<EOF
Usage: $0 {monthly|alltime|both} [args...]

  monthly [args]                 Top N per month (default: $DEFAULT_MONTHLY)
  alltime [args]                 Top N all-time  (default: $DEFAULT_ALLTIME)
  both [m_args... -- a_args...]  Run both. '--' separates monthly from alltime
                                 args. With no args, both modes use their
                                 defaults.

Examples:
  $0 monthly --top 100 --years 2024 --workers 8
  $0 alltime --top 5000
  $0 both
  $0 both --top 100 --years 2024 -- --top 5000

Snapshots land in $RUNS_DIR/ with a suffix derived from the run mode and --top.
EOF
}

split_at_separator() {
    # Populate global arrays MONTHLY_ARGS and ALLTIME_ARGS by splitting "$@"
    # at the first '--'. Sets SEPARATOR_FOUND=1 if '--' was present.
    MONTHLY_ARGS=()
    ALLTIME_ARGS=()
    SEPARATOR_FOUND=0
    for arg in "$@"; do
        if [[ $SEPARATOR_FOUND -eq 0 && $arg == "--" ]]; then
            SEPARATOR_FOUND=1
            continue
        fi
        if [[ $SEPARATOR_FOUND -eq 0 ]]; then
            MONTHLY_ARGS+=("$arg")
        else
            ALLTIME_ARGS+=("$arg")
        fi
    done
}

case "${1:-}" in
    monthly) shift; monthly_run "$@";;
    alltime) shift; alltime_run "$@";;
    both)
        shift
        split_at_separator "$@"
        if [[ $# -gt 0 && $SEPARATOR_FOUND -eq 0 ]]; then
            echo "Error: 'both' with args needs '--' to separate monthly from alltime." >&2
            echo "Example: $0 both --top 100 --years 2024 -- --top 5000" >&2
            exit 1
        fi
        monthly_run "${MONTHLY_ARGS[@]}"
        alltime_run "${ALLTIME_ARGS[@]}"
        ;;
    -h|--help|help|"") usage; exit 0;;
    *) echo "Unknown mode: $1" >&2; usage; exit 1;;
esac
