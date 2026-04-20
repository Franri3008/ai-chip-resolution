import argparse
import os
import re

import requests

from llm_client import complete_async

_GH_TOKEN = ""
_gh_token_path = os.path.join(os.path.dirname(__file__), "..", "..", "keys", ".gh_token")
if os.path.exists(_gh_token_path):
    with open(_gh_token_path) as f:
        _GH_TOKEN = f.read().strip()

_GH_HEADERS = {"Accept": "application/vnd.github.v3+json"}
if _GH_TOKEN:
    _GH_HEADERS["Authorization"] = f"token {_GH_TOKEN}"


def fetch_repo_metadata(url):
    match = re.match(r'https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', url.rstrip('/.'))
    if not match:
        return None
    owner, repo = match.group(1), match.group(2)
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=_GH_HEADERS, timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "description": data.get("description") or "(no description)",
            "topics": data.get("topics", []),
            "language": data.get("language") or "unknown",
            "fork": data.get("fork", False),
            "size_kb": data.get("size", 0),
        }
    except requests.RequestException:
        return None


def _build_prompt(modelcard_text, model_name, candidate_links):
    links_with_meta = []
    for link in candidate_links[:5]:
        meta = fetch_repo_metadata(link)
        if meta:
            topics_str = ", ".join(meta["topics"][:5]) if meta["topics"] else "none"
            meta_str = (
                f'  Description: {meta["description"]}\n'
                f'  Topics: [{topics_str}]\n'
                f'  Language: {meta["language"]}, Fork: {meta["fork"]}, Size: {meta["size_kb"]}KB'
            )
        else:
            meta_str = "  (metadata unavailable)"
        links_with_meta.append(f"- {link}\n{meta_str}")

    snippets = []
    for link in candidate_links:
        start_idx = 0
        while True:
            idx = modelcard_text.find(link, start_idx)
            if idx == -1:
                break
            start = max(0, idx - 100)
            end = min(len(modelcard_text), idx + len(link) + 100)
            snippets.append(f"...{modelcard_text[start:end].replace(chr(10), ' ').strip()}...")
            start_idx = idx + len(link)

    seen = set()
    unique_snippets = [s for s in snippets if not (s in seen or seen.add(s))]

    return (
        f'For the HuggingFace model "{model_name}", identify which GitHub repository '
        f"contains the actual source code, model architecture, or training scripts "
        f"for this specific model or model family.\n\n"
        f"=== MODEL CARD (excerpt) ===\n{modelcard_text[:4000]}\n\n"
        f"=== CANDIDATE REPOSITORIES (with metadata) ===\n{chr(10).join(links_with_meta)}\n\n"
        f"=== CONTEXT (where links appear in model card) ===\n"
        f"{chr(10).join(f'  - {s}' for s in unique_snippets) or '  (none found)'}\n\n"
        f"RULES:\n"
        f"- Do NOT select inference runtimes or serving tools "
        f"(text-embeddings-inference, vLLM, TGI, llama.cpp)\n"
        f"- Do NOT select dataset repositories\n"
        f"- Do NOT select generic framework repos (huggingface/transformers, "
        f"huggingface/diffusers) UNLESS this model IS the framework's own reference model\n"
        f"- A forked or tiny (<50KB) repo is suspicious — verify it's relevant\n\n"
        f'Reply with ONLY the URL of the best repository, or "None" if none is appropriate.'
    )


async def ask_llm_github(modelcard_text, model_name, candidate_links):
    """Ask the configured LLM to identify the main GitHub repo for a model.

    Returns (selected_url_or_None, cost_float).
    """
    prompt = _build_prompt(modelcard_text, model_name, candidate_links)
    answer, cost = await complete_async([{"role": "user", "content": prompt}])
    if answer in candidate_links:
        return answer, cost
    return None, cost


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(description="Ask an LLM to identify the main GitHub repo for a model")
    parser.add_argument("--modelcard", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--links", required=True, help="Comma-separated candidate GitHub links")
    args = parser.parse_args()

    with open(args.modelcard, encoding="utf-8") as f:
        modelcard_text = f.read()

    candidate_links = [link.strip() for link in args.links.split(",") if link.strip()]
    result, cost = asyncio.run(ask_llm_github(modelcard_text, args.model_name, candidate_links))
    print(f"Result: {result}")
    print(f"Cost: ${cost:.6f}")
