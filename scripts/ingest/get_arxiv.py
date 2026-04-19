"""Extract arXiv paper URLs from model cards and write them back to modelcards.json."""

import json
import re
from pathlib import Path

data_path = Path(__file__).parent.parent.parent / "database" / "modelcards.json"

# Matches arxiv.org/abs/{id} and arxiv.org/pdf/{id}[.pdf] with optional version suffix
arxiv_re = re.compile(
    r'https?://arxiv\.org/(?:abs|pdf)/([\w.]+?)(?:v\d+)?(?:\.pdf)?(?=[\s\])\"\'>,:;]|$)',
    re.IGNORECASE,
)


def extract_arxiv_links(text):
    """Return deduplicated list of normalized arxiv abs URLs from text."""
    ids = list(dict.fromkeys(arxiv_re.findall(text)))
    return [f"https://arxiv.org/abs/{aid}" for aid in ids]


def main():
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    for model in data:
        card = model.get("modelcard", "")
        links = extract_arxiv_links(card)
        model["arxiv_links"] = links
        if links:
            total += 1

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Extracted arXiv links: {total}/{len(data)} models have at least one paper")


if __name__ == "__main__":
    main()
