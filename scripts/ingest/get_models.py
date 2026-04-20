import csv
import json
import logging
import os
import time

from huggingface_hub import HfApi

logging.disable(logging.WARNING)

token_path = os.path.join(os.path.dirname(__file__), "..", "..", "keys", ".hf_token")
with open(token_path) as f:
    hf_token = f.read().strip()
api = HfApi(token=hf_token)

start = time.time()

models = api.list_models(
    full=True,
    cardData=True,
    sort="downloads",
)

fields = ["id", "downloads", "likes", "pipeline_tag", "created_at", "last_modified", "tags", "card_data"]

out_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "models.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for m in models:
        try:
            card = json.dumps(vars(m.cardData)) if m.cardData else ""
        except Exception:
            card = ""
        created = getattr(m, "created_at", None) or getattr(m, "createdAt", None)
        writer.writerow({
            "id": m.id,
            "downloads": m.downloads,
            "likes": m.likes,
            "pipeline_tag": m.pipeline_tag,
            "created_at": str(created) if created else "",
            "last_modified": str(m.lastModified) if hasattr(m, "lastModified") else str(getattr(m, "last_modified", "")),
            "tags": "|".join(m.tags or []),
            "card_data": card,
        })

print(f"Models extracted.\nTotal time: {(time.time() - start)/60} minutes")