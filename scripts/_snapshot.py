"""Snapshot database/*.json artifacts into database/runs/ with a suffix.

Two modes:
  * --copy: copy each artifact verbatim (used by single-mode runs).
  * --ids-file: filter list/dict artifacts to the records whose `id` is in
    the file. Used by run.sh both-mode after the pipeline runs once over
    the union of monthly+alltime IDs — each mode's snapshot keeps only its
    own subset so dashboards see the same per-mode views as before.
"""
import argparse
import json
import os
import pathlib

ARTIFACTS = (
    "modelcards.json",
    "results.json",
    "modelcard_chip_analysis.json",
    "github_chip_analysis.json",
    "arxiv_chip_analysis.json",
)


def _load_ids(path):
    with open(path, encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }


def _filter(data, ids):
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict) and r.get("id") in ids]
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k in ids}
    return data


def _size(x):
    return len(x) if hasattr(x, "__len__") else "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", required=True,
                   help="Suffix appended to each artifact filename in runs/.")
    p.add_argument("--ids-file", default=None,
                   help="When set, filter list/dict artifacts to records "
                        "whose `id` is in this file. Otherwise: copy verbatim.")
    p.add_argument("--db-dir", default="database")
    p.add_argument("--runs-dir", default="database/runs")
    args = p.parse_args()

    pathlib.Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    ids = _load_ids(args.ids_file) if args.ids_file else None

    for fname in ARTIFACTS:
        src = os.path.join(args.db_dir, fname)
        if not os.path.exists(src):
            continue
        dst = os.path.join(args.runs_dir, f"{fname[:-5]}_{args.suffix}.json")
        if ids is None:
            with open(src, "rb") as fin, open(dst, "wb") as fout:
                fout.write(fin.read())
            print(f"  Snapshot → {dst}")
            continue
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        filtered = _filter(data, ids)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        print(f"  Snapshot ({_size(filtered)}/{_size(data)}) → {dst}")


if __name__ == "__main__":
    main()
