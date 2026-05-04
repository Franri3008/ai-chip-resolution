"""Score a results.json against tests/ground_truth*.csv with per-slice + per-outcome metrics.

Designed to be runnable standalone after `main.py --ids-file …`. Reports:

    correct        predicted == expected
    missed         predicted == 'unknown', expected != 'unknown'   (recall miss; OK per policy)
    false_positive predicted != 'unknown', expected == 'unknown'   (precision miss; bad)
    confused       both known but differ                           (worst)

Slices are read from the `method` column of the CSVs (top / random / chinese_slice /
chinese_slice_nvidia_control / etc.). A model present in multiple CSVs is bucketed
by the *last* slice that mentions it, so curated overrides win.

Usage:
    python tests/eval/eval_score.py [results.json]                # default: database/results.json
    python tests/eval/eval_score.py results_baseline.json results_after.json  # diff mode
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"

_GT_PROVIDER_MAP = {
    "nvidia": "nvidia", "google": "google_tpu", "apple": "apple", "amd": "amd",
    "intel": "intel", "aws": "aws", "qualcomm": "qualcomm",
    "huawei": "huawei_ascend", "huawei_ascend": "huawei_ascend", "ascend": "huawei_ascend",
    "cambricon": "cambricon", "unknown": "unknown",
}


def load_gt():
    """Return {id: {"expected": chip, "slice": method, "note": ...}}."""
    gt = {}
    files = []
    main_csv = TESTS / "ground_truth.csv"
    if main_csv.exists():
        files.append(main_csv)
    files.extend(sorted(TESTS.glob("ground_truth_*.csv")))
    for path in files:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mid = row["id"].strip()
                provider = row["provider"].strip().lower()
                expected = _GT_PROVIDER_MAP.get(provider, provider)
                slice_name = (row.get("method") or "").strip() or "unscoped"
                gt[mid] = {"expected": expected, "slice": slice_name}
    return gt


def categorize(pred, expected):
    if pred == expected:
        return "correct"
    if pred == "unknown" and expected != "unknown":
        return "missed"
    if pred != "unknown" and expected == "unknown":
        return "false_positive"
    return "confused"


def score(results_path):
    gt = load_gt()
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    by_slice = defaultdict(lambda: defaultdict(int))
    rows = []
    for r in results:
        mid = r["id"]
        if mid not in gt:
            continue
        expected = gt[mid]["expected"]
        slice_name = gt[mid]["slice"]
        pred = r["conclusion"]["chip_provider"]
        outcome = categorize(pred, expected)
        by_slice[slice_name][outcome] += 1
        by_slice[slice_name]["_total"] += 1
        by_slice["__ALL__"][outcome] += 1
        by_slice["__ALL__"]["_total"] += 1
        rows.append({
            "id": mid, "slice": slice_name, "expected": expected,
            "pred": pred, "outcome": outcome,
            "src": r["conclusion"].get("chip_provider_source") or "—",
            "conf": r["conclusion"].get("chip_provider_confidence", 0.0),
        })
    return rows, by_slice


def fmt_pct(n, d):
    return f"{n}/{d} ({100*n/d:.1f}%)" if d else f"{n}/0 (—)"


def print_report(results_path, rows, by_slice):
    print(f"\n=== {results_path} ===")
    slice_order = sorted(s for s in by_slice if s != "__ALL__")
    slice_order.append("__ALL__")
    print(f"\n{'SLICE':40s} {'CORRECT':>14s} {'MISSED':>14s} {'FP':>14s} {'CONFUSED':>14s} {'TOTAL':>6s}")
    for s in slice_order:
        tot = by_slice[s]["_total"]
        print(f"{s:40s} "
              f"{fmt_pct(by_slice[s]['correct'], tot):>14s} "
              f"{fmt_pct(by_slice[s]['missed'], tot):>14s} "
              f"{fmt_pct(by_slice[s]['false_positive'], tot):>14s} "
              f"{fmt_pct(by_slice[s]['confused'], tot):>14s} "
              f"{tot:>6d}")
    print()
    bad = [r for r in rows if r["outcome"] in ("false_positive", "confused")]
    if bad:
        print("Failures (false_positive / confused):")
        for r in bad:
            print(f"  [{r['outcome']:14s}] {r['id']:55s} expected={r['expected']:15s} got={r['pred']:15s} ({r['src']})")
    misses = [r for r in rows if r["outcome"] == "missed"]
    if misses:
        print(f"\nMissed ({len(misses)} rows — predicted 'unknown' when GT had a label):")
        for r in misses:
            print(f"  {r['id']:55s} expected={r['expected']:15s} ({r['src']})")


def diff_reports(rows_a, rows_b, label_a, label_b):
    by_id_a = {r["id"]: r for r in rows_a}
    by_id_b = {r["id"]: r for r in rows_b}
    flips = []
    for mid in sorted(set(by_id_a) | set(by_id_b)):
        ra = by_id_a.get(mid); rb = by_id_b.get(mid)
        if not ra or not rb: continue
        if ra["outcome"] != rb["outcome"] or ra["pred"] != rb["pred"]:
            flips.append((mid, ra, rb))
    if not flips:
        print(f"\n=== DIFF {label_a} → {label_b}: no per-row changes ===")
        return
    print(f"\n=== DIFF {label_a} → {label_b}: {len(flips)} row(s) changed ===")
    for mid, ra, rb in flips:
        arrow = f"{ra['pred']:15s} ({ra['outcome']:14s}) → {rb['pred']:15s} ({rb['outcome']:14s})"
        print(f"  {mid:55s} expected={ra['expected']:15s} {arrow}")


def main():
    paths = [Path(p) for p in sys.argv[1:]] or [ROOT / "database" / "results.json"]
    reports = []
    for p in paths:
        if not p.exists():
            print(f"Missing: {p}", file=sys.stderr); sys.exit(1)
        rows, by_slice = score(p)
        print_report(p, rows, by_slice)
        reports.append((p, rows))
    if len(reports) == 2:
        diff_reports(reports[0][1], reports[1][1], reports[0][0].name, reports[1][0].name)


if __name__ == "__main__":
    main()
