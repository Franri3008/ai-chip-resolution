"""LLM fallback for selecting the primary arXiv paper for a HuggingFace model."""

import re
import time
import urllib.request
import xml.etree.ElementTree as ET

from llm_client import complete_async

_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _extract_arxiv_id(url):
    m = re.search(r'arxiv\.org/(?:abs|pdf)/([\w.]+?)(?:v\d+)?(?:\.pdf)?$', url)
    return m.group(1) if m else None


def fetch_arxiv_metadata(arxiv_id):
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


def _build_prompt(modelcard_text, model_name, candidate_links):
    links_with_meta = []
    for link in candidate_links[:5]:
        arxiv_id = _extract_arxiv_id(link)
        meta = fetch_arxiv_metadata(arxiv_id) if arxiv_id else None
        if meta:
            meta_str = (
                f'  Title: {meta["title"]}\n'
                f'  Authors: {", ".join(meta["authors"][:5])}\n'
                f'  Categories: {", ".join(meta["categories"]) or "unknown"}\n'
                f'  Abstract: {meta["abstract"][:300]}...'
            )
        else:
            meta_str = "  (metadata unavailable)"
        links_with_meta.append(f"- {link}\n{meta_str}")
        time.sleep(0.5)

    snippets = []
    for link in candidate_links:
        arxiv_id = _extract_arxiv_id(link)
        for term in ([link] + ([arxiv_id] if arxiv_id else [])):
            start_idx = 0
            while True:
                idx = modelcard_text.find(term, start_idx)
                if idx == -1:
                    break
                start = max(0, idx - 100)
                end = min(len(modelcard_text), idx + len(term) + 100)
                snippets.append(f"...{modelcard_text[start:end].replace(chr(10), ' ').strip()}...")
                start_idx = idx + len(term)

    seen = set()
    unique_snippets = [s for s in snippets if not (s in seen or seen.add(s))]

    return (
        f'For the HuggingFace model "{model_name}", identify which arXiv paper '
        f"describes the TRAINING or original development of this specific model.\n\n"
        f"=== MODEL CARD (excerpt) ===\n{modelcard_text[:4000]}\n\n"
        f"=== CANDIDATE PAPERS (with metadata) ===\n{chr(10).join(links_with_meta)}\n\n"
        f"=== CONTEXT (where papers appear in model card) ===\n"
        f"{chr(10).join(f'  - {s}' for s in unique_snippets[:10]) or '  (none found)'}\n\n"
        f"RULES:\n"
        f"- Select the paper that describes how THIS model was trained or created\n"
        f"- Papers introduced with 'our paper', 'we introduce', 'we propose' are strong signals\n"
        f"- Do NOT select papers that are just cited as related work or background\n"
        f"- If NO paper describes the training of this specific model, reply None\n\n"
        f'Reply with ONLY the URL of the best paper, or "None" if none is appropriate.'
    )


async def ask_llm_arxiv(modelcard_text, model_name, candidate_links):
    """Ask the configured LLM to select the primary arXiv paper for a model.

    Returns (selected_url_or_None, cost_float).
    """
    prompt = _build_prompt(modelcard_text, model_name, candidate_links)
    answer, cost = await complete_async([{"role": "user", "content": prompt}])
    if answer in candidate_links:
        return answer, cost
    return None, cost
