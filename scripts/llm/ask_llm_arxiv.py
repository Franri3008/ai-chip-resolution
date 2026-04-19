"""LLM fallback for selecting the primary arXiv paper for a HuggingFace model."""

import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

from openai import OpenAI

_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _extract_arxiv_id(url):
    """Extract bare arxiv ID from an abs/pdf URL."""
    m = re.search(r'arxiv\.org/(?:abs|pdf)/([\w.]+?)(?:v\d+)?(?:\.pdf)?$', url)
    return m.group(1) if m else None


def fetch_arxiv_metadata(arxiv_id):
    """Fetch title, authors, abstract from the arXiv API.

    Returns dict or None on failure.
    """
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        root = ET.fromstring(resp.read())
        entry = root.find("atom:entry", _NS)
        if entry is None:
            return None
        title_el = entry.find("atom:title", _NS)
        summary_el = entry.find("atom:summary", _NS)
        authors = [a.find("atom:name", _NS).text for a in entry.findall("atom:author", _NS)]
        categories = [c.get("term") for c in entry.findall("atom:category", _NS)]
        return {
            "title": title_el.text.strip().replace("\n", " ") if title_el is not None else "(no title)",
            "abstract": summary_el.text.strip()[:500] if summary_el is not None else "(no abstract)",
            "authors": authors[:10],
            "categories": categories[:5],
        }
    except Exception:
        return None


def ask_llm_arxiv(modelcard_text, model_name, candidate_links, llm_model="gpt-4o-mini"):
    """Ask LLM to select the primary arXiv paper for a model.

    Returns (selected_url_or_None, cost_float).
    """
    token_path = os.path.join(os.path.dirname(__file__), "..", "..", "keys", ".openrouter_token")
    api_key = open(token_path).read().strip()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Fetch metadata for each candidate (max 5)
    links_with_meta = []
    for link in candidate_links[:5]:
        arxiv_id = _extract_arxiv_id(link)
        meta = fetch_arxiv_metadata(arxiv_id) if arxiv_id else None
        if meta:
            authors_str = ", ".join(meta["authors"][:5])
            cats_str = ", ".join(meta["categories"]) if meta["categories"] else "unknown"
            meta_str = (
                f'  Title: {meta["title"]}\n'
                f'  Authors: {authors_str}\n'
                f'  Categories: {cats_str}\n'
                f'  Abstract: {meta["abstract"][:300]}...'
            )
        else:
            meta_str = "  (metadata unavailable)"
        links_with_meta.append(f"- {link}\n{meta_str}")
        time.sleep(0.5)  # respect arxiv API rate limits

    links_str = "\n".join(links_with_meta)

    # Extract context snippets around each link in the model card
    snippets = []
    for link in candidate_links:
        # Also match by arxiv ID alone (appears in BibTeX etc.)
        arxiv_id = _extract_arxiv_id(link)
        search_terms = [link]
        if arxiv_id:
            search_terms.append(arxiv_id)
        for term in search_terms:
            start_idx = 0
            while True:
                idx = modelcard_text.find(term, start_idx)
                if idx == -1:
                    break
                start = max(0, idx - 100)
                end = min(len(modelcard_text), idx + len(term) + 100)
                snippet = modelcard_text[start:end].replace('\n', ' ').strip()
                snippets.append(f"...{snippet}...")
                start_idx = idx + len(term)

    seen = set()
    unique_snippets = [s for s in snippets if not (s in seen or seen.add(s))]
    snippets_str = "\n".join(f"  - {s}" for s in unique_snippets[:10]) if unique_snippets else "  (none found)"

    card_excerpt = modelcard_text[:4000]

    prompt = (
        f'For the HuggingFace model "{model_name}", identify which arXiv paper '
        f"describes the TRAINING or original development of this specific model.\n\n"
        f"=== MODEL CARD (excerpt) ===\n{card_excerpt}\n\n"
        f"=== CANDIDATE PAPERS (with metadata) ===\n{links_str}\n\n"
        f"=== CONTEXT (where papers appear in model card) ===\n{snippets_str}\n\n"
        f"RULES:\n"
        f"- Select the paper that describes how THIS model was trained or created\n"
        f"- Papers introduced with 'our paper', 'we introduce', 'we propose' are strong signals\n"
        f"- Do NOT select papers that merely describe a DIFFERENT model this one was fine-tuned from\n"
        f"- Do NOT select papers that are just cited as related work or background\n"
        f"- Do NOT select dataset papers unless the model card specifically says 'our paper'\n"
        f"- If a paper's title closely matches the model name, it's likely the right one\n"
        f"- If NO paper describes the training of this specific model, reply None\n\n"
        f'Reply with ONLY the URL of the best paper, or "None" '
        f"if none is appropriate."
    )

    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
    cost = response.usage.model_dump().get("cost", 0.0)

    if answer in candidate_links:
        return answer, cost
    return None, cost
