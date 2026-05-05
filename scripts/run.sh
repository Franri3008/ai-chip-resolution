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
# both-mode is optimised: it resolves the monthly + alltime model-id lists
# from models.csv, runs the pipeline ONCE over their union, and filters the
# snapshots per mode. Models that appear in both lists are processed only
# once instead of twice.
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
    $PY scripts/_snapshot.py --suffix "$suffix" --runs-dir "$RUNS_DIR"
}

filter_snapshot() {
    # filter_snapshot <suffix> <ids_file> — keep only records whose id is in
    # ids_file. Used after a unified both-mode run to produce per-mode views.
    local suffix=$1 ids_file=$2
    $PY scripts/_snapshot.py --suffix "$suffix" --ids-file "$ids_file" --runs-dir "$RUNS_DIR"
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

both_run_union() {
    # Resolve monthly + alltime model-id lists from models.csv, run main.py
    # once over the union, then filter snapshots per mode. This avoids
    # processing the (large) overlap twice.
    local monthly_args=("${MONTHLY_ARGS[@]}")
    local alltime_args=("${ALLTIME_ARGS[@]}")
    [[ ${#monthly_args[@]} -eq 0 ]] && monthly_args=($DEFAULT_MONTHLY)
    [[ ${#alltime_args[@]} -eq 0 ]] && alltime_args=($DEFAULT_ALLTIME)

    local monthly_top alltime_top
    monthly_top=$(extract_top "${monthly_args[@]}")
    alltime_top=$(extract_top "${alltime_args[@]}")
    local monthly_suffix="top${monthly_top}_per_month"
    local alltime_suffix="top${alltime_top}"

    # The pipeline auto-fetches models.csv only when --update-models is set
    # AND --ids-file is unset. Our union path uses --ids-file (which suppresses
    # the auto-fetch), so honour --update-models manually first.
    if [[ "${monthly_args[*]} ${alltime_args[*]}" == *--update-models* ]]; then
        echo "Refreshing database/models.csv (--update-models requested)…"
        $PY scripts/ingest/get_models.py
    fi

    local monthly_ids alltime_ids union_ids
    monthly_ids=$(mktemp -t aichip_monthly_ids.XXXXXX)
    alltime_ids=$(mktemp -t aichip_alltime_ids.XXXXXX)
    union_ids=$(mktemp -t aichip_union_ids.XXXXXX)

    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  BOTH (union) — resolving monthly + alltime ID lists"
    echo "════════════════════════════════════════════════════════════════"
    $PY scripts/ingest/get_modelcard.py "${monthly_args[@]}" --list-ids "$monthly_ids"
    $PY scripts/ingest/get_modelcard.py "${alltime_args[@]}" --list-ids "$alltime_ids"

    sort -u "$monthly_ids" "$alltime_ids" > "$union_ids"
    local n_monthly n_alltime n_union
    n_monthly=$(wc -l < "$monthly_ids" | tr -d ' ')
    n_alltime=$(wc -l < "$alltime_ids" | tr -d ' ')
    n_union=$(wc -l < "$union_ids"   | tr -d ' ')
    echo "  monthly=$n_monthly  alltime=$n_alltime  union=$n_union  "\
"saved=$((n_monthly + n_alltime - n_union)) duplicate-runs"

    # Run the pipeline once over the union. Pass monthly's args verbatim;
    # main.py silently ignores --top/--years/--quarters/--source-csv when
    # --ids-file is set, so only the cross-cutting flags (--workers, --llm,
    # --provider, --llm-concurrency, --deduplicate) take effect here.
    run_main "BOTH (union of monthly+alltime, $n_union models)" \
             "_union" "run_both_union" \
             "${monthly_args[@]}" --ids-file "$union_ids"
    local rc=$?

    # Replace the auto-snapshot from run_main (which used the "_union" suffix)
    # with per-mode filtered snapshots — those match what monthly_run and
    # alltime_run would have produced individually.
    echo
    echo "  Filter-snapshotting per mode…"
    filter_snapshot "$monthly_suffix" "$monthly_ids"
    filter_snapshot "$alltime_suffix" "$alltime_ids"
    # Drop the redundant "_union" snapshot — keeping it would just duplicate
    # database/*.json on disk.
    for f in "${ARTIFACTS[@]}"; do
        rm -f "$RUNS_DIR/${f%.json}__union.json"
    done

    rm -f "$monthly_ids" "$alltime_ids" "$union_ids"
    return "$rc"
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
        both_run_union
        ;;
    -h|--help|help|"") usage; exit 0;;
    *) echo "Unknown mode: $1" >&2; usage; exit 1;;
esac
