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


def _row_year_month(row: dict) -> tuple:
    """Return (year, month) from CSV row or (None, None) if unparseable.
    Expects ISO-8601-ish timestamps like 2024-07-15T....Z."""
    for field in ("created_at", "last_modified"):
        val = row.get(field, "").strip()
        if val and val != "None":
            try:
                year = int(val[:4])
                month = int(val[5:7])
                if 1 <= month <= 12:
                    return year, month
            except (ValueError, IndexError):
                pass
    return None, None


parser = argparse.ArgumentParser()
parser.add_argument("--top", type=int, default=None,
                    help="When --years is set: top N models per MONTH within the year range "
                         "(so 2022-2023 = 24 monthly buckets × N). Otherwise: total cap.")
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
id_to_month: dict[str, int] = {}

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    if target_years:
        # Top N per (year, month): scan CSV in download-rank order and fill
        # each month's bucket until it reaches --top. A 2022-2023 range is
        # 24 monthly buckets, so `--top 50 --years 2022-2023` yields up to 1200.
        per_month: dict[tuple, list[str]] = {
            (yr, mo): [] for yr in target_years for mo in range(1, 13)
        }
        for row in reader:
            yr, mo = _row_year_month(row)
            if yr is None or mo is None:
                continue
            key = (yr, mo)
            if key not in per_month:
                continue
            bucket = per_month[key]
            if args.top is None or len(bucket) < args.top:
                bucket.append(row["id"])
                id_to_year[row["id"]] = yr
                id_to_month[row["id"]] = mo
        model_ids = [mid for key in sorted(per_month) for mid in per_month[key]]
        # Per-year summary (aggregate across months for a compact log).
        per_year_totals: dict[int, int] = {yr: 0 for yr in target_years}
        non_empty_months = 0
        for (yr, mo), bucket in per_month.items():
            per_year_totals[yr] += len(bucket)
            if bucket:
                non_empty_months += 1
        for yr in sorted(per_year_totals):
            print(f"  {yr}: {per_year_totals[yr]} models"
                  + (f" (top {args.top}/month × 12)" if args.top else ""))
        print(f"Total: {len(model_ids)} models across "
              f"{non_empty_months}/{12*len(target_years)} monthly buckets")
    else:
        if args.top:
            rows = [row for _, row in zip(range(args.top), reader)]
        else:
            rows = list(reader)
        model_ids = [row["id"] for row in rows]
        for row in rows:
            yr, mo = _row_year_month(row)
            if yr is not None:
                id_to_year[row["id"]] = yr
            if mo is not None:
                id_to_month[row["id"]] = mo

def _fetch(model_id):
    try:
        card = ModelCard.load(model_id)
        if card.content and card.content.strip():
            rec = {"id": model_id, "modelcard": card.content}
            yr = id_to_year.get(model_id)
            if yr is not None:
                rec["year"] = yr
            mo = id_to_month.get(model_id)
            if mo is not None:
                rec["month"] = mo
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
