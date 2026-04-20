"""Score and select the primary arXiv paper for each model.

Heuristic scoring pass + LLM fallback for low-confidence picks.
Writes main_arxiv, main_arxiv_confidence, main_arxiv_source to modelcards.json.
"""

import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm"))
from ask_llm_arxiv import ask_llm_arxiv
from llm_client import llm_enabled, set_concurrency

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "modelcards.json")

# Context patterns — signals that the paper is THIS model's paper
OUR_PAPER_RE = re.compile(
    r'(our\s+paper|we\s+introduce|we\s+propose|we\s+present|we\s+release|'
    r'this\s+(?:paper|work|model)\s+(?:is|was)|described\s+in\s+our|'
    r'paper\s*:|technical\s+report)',
    re.IGNORECASE,
)

# Context patterns — signals that the paper is about a DIFFERENT model
THIRD_PARTY_RE = re.compile(
    r'(based\s+on|originally\s+(?:proposed|introduced|described)\s+in|'
    r'following|as\s+(?:described|proposed)\s+in|inspired\s+by|'
    r'builds\s+(?:on|upon)|extends)',
    re.IGNORECASE,
)

BIBTEX_RE = re.compile(r'```bibtex')
TABLE_ROW_RE = re.compile(r'\|.*\|')

LLM_CONFIDENCE_THRESHOLD = 0.60


def relative_position(card, pos):
    total = len(card)
    if total == 0:
        return 0.5
    return pos / total


def is_in_bibtex(card, pos):
    for m in BIBTEX_RE.finditer(card):
        bib_end = card.find('```', m.start() + 3)
        if bib_end == -1:
            bib_end = len(card)
        if m.start() <= pos <= bib_end:
            return True
    return False


def is_in_table(card, pos):
    line_start = card.rfind('\n', 0, pos) + 1
    line_end = card.find('\n', pos)
    if line_end == -1:
        line_end = len(card)
    line = card[line_start:line_end]
    return bool(TABLE_ROW_RE.match(line.strip()))


def has_context(card, pos, pattern, window=200):
    start = max(0, pos - window)
    end = min(len(card), pos + 60 + window)
    snippet = card[start:end]
    return bool(pattern.search(snippet))


def score_arxiv_link(arxiv_url, model_id, card):
    """Score an arXiv link based on its context in the model card.

    Returns (score, reasons).
    """
    score = 0
    reasons = []

    pos = card.find(arxiv_url)
    if pos == -1:
        # Try matching by arxiv ID alone
        m = re.search(r'arxiv\.org/(?:abs|pdf)/([\w.]+)', arxiv_url)
        if m:
            pos = card.find(m.group(1))
    if pos == -1:
        pos = len(card) // 2

    # "Our paper" context — strong signal this is the model's paper
    if has_context(card, pos, OUR_PAPER_RE):
        score += 4
        reasons.append("our paper context")

    # Model name appears near the link
    model_name = model_id.split("/")[-1].lower()
    model_name_stripped = re.sub(r'[-_]?(v?\d+[\d.]*[bBmM]?|base|large|small|tiny|mini|instruct|uncased|cased)$', '', model_name, flags=re.IGNORECASE)
    window_start = max(0, pos - 300)
    window_end = min(len(card), pos + 300)
    nearby_text = card[window_start:window_end].lower()
    if model_name_stripped and model_name_stripped in nearby_text:
        score += 3
        reasons.append("model name near link")

    # Position in card
    rel_pos = relative_position(card, pos)
    if rel_pos < 0.15:
        score += 2
        reasons.append("very early in card")
    elif rel_pos < 0.35:
        score += 1
        reasons.append("early in card")

    # Negative: inside BibTeX block (just a citation, not an intro reference)
    if is_in_bibtex(card, pos):
        score -= 3
        reasons.append("inside bibtex")

    # Negative: inside table row
    if is_in_table(card, pos):
        score -= 1
        reasons.append("inside table row")

    # Negative: third-party reference context
    if has_context(card, pos, THIRD_PARTY_RE):
        score -= 2
        reasons.append("third-party reference context")

    return score, reasons


def compute_confidence(best_score, second_score, num_links):
    if num_links == 0:
        return 0.0
    if num_links == 1:
        return min(1.0, 0.7 + max(0, best_score) * 0.03)
    gap = best_score - second_score
    base = 0.5
    base += min(0.3, gap * 0.06)
    base += min(0.15, max(0, best_score) * 0.015)
    return round(min(1.0, max(0.1, base)), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for LLM fallback calls (default: 4)")
    parser.add_argument("--llm-concurrency", type=int, default=None,
                        help="Max concurrent in-flight LLM requests. Defaults to --workers.")
    args = parser.parse_args()

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    # Pass 1: Heuristic scoring (fast, no network calls)
    for model in data:
        links = model.get("arxiv_links", [])
        card = model.get("modelcard", "")
        model_id = model.get("id", "")

        if not links:
            model["main_arxiv"] = None
            model["main_arxiv_confidence"] = 0.0
            model["main_arxiv_source"] = None
            continue

        if len(links) == 1:
            s, r = score_arxiv_link(links[0], model_id, card)
            conf = compute_confidence(s, -10, 1)
            model["main_arxiv"] = links[0]
            model["main_arxiv_confidence"] = conf
            model["main_arxiv_source"] = "heuristic"
            continue

        scored = []
        for link in links:
            s, r = score_arxiv_link(link, model_id, card)
            scored.append((s, r, link))
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_reasons, best_link = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -10

        conf = compute_confidence(best_score, second_score, len(links))
        model["main_arxiv"] = best_link
        model["main_arxiv_confidence"] = conf
        model["main_arxiv_source"] = "heuristic"

    # Pass 2: LLM fallback for low-confidence picks (parallelized)
    model_by_id = {m["id"]: m for m in data}
    llm_queue = [
        (m["id"], m.get("modelcard", ""), m.get("arxiv_links", []))
        for m in data
        if m.get("main_arxiv_confidence", 1.0) < LLM_CONFIDENCE_THRESHOLD
        and m.get("arxiv_links", [])
    ]

    total_llm_cost = 0.0
    if not llm_enabled():
        if llm_queue:
            print(f"  LLM disabled (--llm not set). Skipping {len(llm_queue)} arXiv candidate-selection call(s).")
    else:
        set_concurrency(args.llm_concurrency or args.workers)

        async def _run_all_llm():
            async def _one(item):
                model_id, card, links = item
                try:
                    llm_result, cost = await ask_llm_arxiv(card, model_id, links)
                    return model_id, llm_result, cost, None
                except Exception as e:
                    return model_id, None, 0.0, str(e)
            return await asyncio.gather(*[_one(item) for item in llm_queue])

        for model_id, llm_result, cost, err in asyncio.run(_run_all_llm()):
            total_llm_cost += cost
            if err:
                print(f"  LLM failed for {model_id}: {err}")
            elif llm_result:
                model_by_id[model_id]["main_arxiv"] = llm_result
                model_by_id[model_id]["main_arxiv_source"] = "llm"
                print(f"  LLM override for {model_id}: {llm_result} (Cost: ${cost:.6f})")

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Done. Results:")
    for model in data:
        conf = model.get("main_arxiv_confidence", 0)
        link = model.get("main_arxiv", "N/A")
        if link:
            print(f"  {model['id']:50s} -> {str(link):50s}  (confidence: {conf})")

    if total_llm_cost > 0:
        print(f"\nTotal LLM Cost: ${total_llm_cost:.6f}")


if __name__ == "__main__":
    main()
