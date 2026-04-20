import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

csv.field_size_limit(sys.maxsize)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1";

from huggingface_hub import ModelCard, login
from huggingface_hub import logging as hf_logging
from tqdm import tqdm

hf_logging.set_verbosity_error();

token_path = os.path.join(os.path.dirname(__file__), "..", "..", "keys", ".hf_token")
with open(token_path) as f:
    hf_token = f.read().strip();
login(token=hf_token);

def _parse_years(years_str: str) -> set:
    """Parse a years string into a set of ints.

    Accepts: single year "2023", comma list "2022,2023", range "2022-2024",
    or a mix "2021,2023-2025" → {2021, 2023, 2024, 2025}.
    """
    years = set()
    for part in years_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            years.update(range(int(lo), int(hi) + 1))
        else:
            years.add(int(part))
    return years


def _row_year(row: dict) -> int | None:
    """Return the creation year for a CSV row (created_at, fallback last_modified)."""
    for field in ("created_at", "last_modified"):
        val = row.get(field, "").strip()
        if val and val != "None":
            try:
                return int(val[:4])
            except (ValueError, IndexError):
                pass
    return None


parser = argparse.ArgumentParser()
parser.add_argument("--top", type=int, default=None,
                    help="Number of models to fetch (default: all)")
parser.add_argument("--years", type=str, default=None,
                    help="Filter by creation year(s). Examples: 2023  |  2022,2023  |  2022-2024")
parser.add_argument("--workers", type=int, default=8,
                    help="Concurrent HF model-card fetches (default: 8). "
                         "HF reads are I/O-bound, so this scales well on wide machines.")
args = parser.parse_args()

csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "models.csv")
out_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "modelcards.json")

target_years = _parse_years(args.years) if args.years else None

id_to_year: dict[str, int] = {}

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    if target_years:
        # Top N per year: scan CSV in download-rank order, fill each year's
        # bucket until it reaches --top, then combine.
        per_year: dict[int, list[str]] = {yr: [] for yr in target_years}
        for row in reader:
            yr = _row_year(row)
            if yr in per_year:
                bucket = per_year[yr]
                if args.top is None or len(bucket) < args.top:
                    bucket.append(row["id"])
                    id_to_year[row["id"]] = yr
        model_ids = [mid for yr in sorted(per_year) for mid in per_year[yr]]
        for yr in sorted(per_year):
            print(f"  {yr}: {len(per_year[yr])} models"
                  + (f" (top {args.top})" if args.top else ""))
        print(f"Total: {len(model_ids)} models across {len(target_years)} year(s)")
    else:
        if args.top:
            rows = [row for _, row in zip(range(args.top), reader)]
        else:
            rows = list(reader)
        model_ids = [row["id"] for row in rows]
        for row in rows:
            yr = _row_year(row)
            if yr is not None:
                id_to_year[row["id"]] = yr

def _fetch(model_id):
    try:
        card = ModelCard.load(model_id)
        if card.content and card.content.strip():
            rec = {"id": model_id, "modelcard": card.content}
            yr = id_to_year.get(model_id)
            if yr is not None:
                rec["year"] = yr
            return rec
    except Exception:
        pass
    return None


results = []
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futures = [ex.submit(_fetch, mid) for mid in model_ids]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching model cards"):
        out = fut.result()
        if out is not None:
            results.append(out)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
