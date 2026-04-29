"""Extract arXiv paper URLs from model cards and write them back to modelcards.json."""

import json
import re
from pathlib import Path

data_path = Path(__file__).parent.parent.parent / "database" / "modelcards.json"

# Matches https://arxiv.org/abs/{id} and pdf/{id}[.pdf] with optional version
# suffix. The end-anchor accepts whitespace, common URL terminators AND `}` so
# BibTeX `url={https://arxiv.org/abs/X}` blocks parse correctly.
arxiv_re = re.compile(
    r'https?://arxiv\.org/(?:abs|pdf)/([\w.]+?)(?:v\d+)?(?:\.pdf)?(?=[\s\])\"\'>,:;}]|$)',
    re.IGNORECASE,
)

# BibTeX eprint pattern: `eprint = {2407.20750}` or `eprint={2407.20750v1}`.
# Many cards include only the eprint, no full URL. Recover those too.
arxiv_eprint_re = re.compile(
    r'eprint\s*=\s*\{?\s*(\d{4}\.\d{4,5})(?:v\d+)?\s*\}?',
    re.IGNORECASE,
)


def extract_arxiv_links(text):
    """Return deduplicated list of normalized arxiv abs URLs from text."""
    ids = list(arxiv_re.findall(text))
    ids.extend(arxiv_eprint_re.findall(text))
    seen = set()
    out = []
    for aid in ids:
        if aid not in seen:
            seen.add(aid)
            out.append(aid)
    return [f"https://arxiv.org/abs/{aid}" for aid in out]


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
