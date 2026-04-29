"""Extract the top 5000 models (by downloads) created in the last 6 months."""

import csv
import heapq
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

csv.field_size_limit(sys.maxsize)

HERE = Path(__file__).resolve().parent
SRC = HERE / "models.csv"
DST = HERE / "short.csv"

TOP_N = 5000
MONTHS_BACK = 6
CUTOFF = datetime.now(timezone.utc) - timedelta(days=MONTHS_BACK * 30)


def parse_created_at(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> None:
    heap: list[tuple[int, int, list[str]]] = []
    counter = 0
    scanned = 0
    kept_after_filter = 0

    with SRC.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        downloads_idx = header.index("downloads")
        created_idx = header.index("created_at")

        for row in reader:
            scanned += 1
            created = parse_created_at(row[created_idx])
            if created is None or created < CUTOFF:
                continue
            try:
                downloads = int(row[downloads_idx])
            except (ValueError, IndexError):
                continue
            kept_after_filter += 1
            counter += 1
            entry = (downloads, counter, row)
            if len(heap) < TOP_N:
                heapq.heappush(heap, entry)
            elif downloads > heap[0][0]:
                heapq.heapreplace(heap, entry)

    top = sorted(heap, key=lambda e: e[0], reverse=True)

    with DST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for _, _, row in top:
            writer.writerow(row)

    print(f"scanned={scanned} in_window={kept_after_filter} written={len(top)} -> {DST}")


if __name__ == "__main__":
    main()
