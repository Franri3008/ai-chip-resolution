import argparse
import csv
import json
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1";

from huggingface_hub import ModelCard, login
from huggingface_hub import logging as hf_logging
from tqdm import tqdm

hf_logging.set_verbosity_error();

token_path = os.path.join(os.path.dirname(__file__), "..", "..", "keys", ".hf_token")
with open(token_path) as f:
    hf_token = f.read().strip();
login(token=hf_token);

parser = argparse.ArgumentParser()
parser.add_argument("--top", type=int, default=None, help="Number of models to fetch (default: all)");
args = parser.parse_args();

csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "models.csv")
out_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "modelcards.json")

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if args.top:
        model_ids = [row["id"] for _, row in zip(range(args.top), reader)]
    else:
        model_ids = [row["id"] for row in reader]

results = [];
for model_id in tqdm(model_ids, desc="Fetching model cards"):
    try:
        card = ModelCard.load(model_id)
        if card.content and card.content.strip():
            results.append({"id": model_id, "modelcard": card.content})
    except Exception:
        pass

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
