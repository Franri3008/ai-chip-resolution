"""One-shot script: add 'correct' field to an existing results.json using ground_truth.csv."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
GT_PATH = ROOT / "tests" / "ground_truth.csv"
RESULTS_PATH = ROOT / "database" / "results.json"

# Ground truth provider names → pipeline chip_provider names
_GT_PROVIDER_MAP = {
    "nvidia": "nvidia",
    "google": "google_tpu",
    "apple": "apple",
    "amd": "amd",
    "intel": "intel",
    "aws": "aws",
    "qualcomm": "qualcomm",
    "unknown": "unknown",
}


def main():
    # Load ground truth
    gt = {}
    with open(GT_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row["id"].strip()
            provider = row["provider"].strip().lower()
            gt[model_id] = _GT_PROVIDER_MAP.get(provider, provider)

    # Load results
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    correct = 0
    incorrect = 0
    not_in_gt = 0
    mismatches = []

    for r in results:
        model_id = r["id"]
        conclusion = r["conclusion"]
        predicted = conclusion["chip_provider"]

        if model_id not in gt:
            conclusion["correct"] = -1
            not_in_gt += 1
            continue

        expected = gt[model_id]

        conclusion["expected_provider"] = expected
        if predicted == expected:
            conclusion["correct"] = 1
            correct += 1
        else:
            conclusion["correct"] = 0
            incorrect += 1
            mismatches.append((model_id, expected, predicted, conclusion.get("chip_provider_source")))

    # Write back
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    evaluated = correct + incorrect

    print(f"Ground Truth Evaluation")
    print(f"{'='*60}")
    print(f"  Models in ground truth:     {evaluated}")
    print(f"  Not in ground truth:        {not_in_gt}")
    if evaluated > 0:
        pct = correct / evaluated * 100
        print(f"  Correct:                    {correct}/{evaluated} ({pct:.1f}%)")
        print(f"  Incorrect:                  {incorrect}/{evaluated} ({100-pct:.1f}%)")
    else:
        print(f"  Correct:                    0/0 (N/A)")

    if mismatches:
        print(f"\n  Mismatches:")
        for model_id, expected, predicted, src in mismatches:
            print(f"    {model_id:48s}  expected={expected:12s}  got={predicted:12s}  (via {src or '?'})")

    print(f"\nUpdated {RESULTS_PATH}")


if __name__ == "__main__":
    main()
