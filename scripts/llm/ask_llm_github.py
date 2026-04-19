import argparse
import os
import re
import sys

import requests
from openai import OpenAI

# GitHub API setup (reuse token from parent project)
_GH_TOKEN = ""
_gh_token_path = os.path.join(os.path.dirname(__file__), "..", "..", "keys", ".gh_token")
if os.path.exists(_gh_token_path):
    with open(_gh_token_path) as f:
        _GH_TOKEN = f.read().strip()

_GH_HEADERS = {"Accept": "application/vnd.github.v3+json"}
if _GH_TOKEN:
    _GH_HEADERS["Authorization"] = f"token {_GH_TOKEN}"


def fetch_repo_metadata(url):
    """Fetch lightweight metadata for a GitHub repo URL.

    Returns dict with description, topics, language, fork, size or None on failure.
    """
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


def ask_llm_github(modelcard_text, model_name, candidate_links, llm_model="gpt-4o-mini"):
    token_path = os.path.join(os.path.dirname(__file__), "..", "..", "keys", ".openrouter_token")
    api_key = open(token_path).read().strip()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Fetch repo metadata for each candidate (max 5)
    links_with_meta = []
    for link in candidate_links[:5]:
        meta = fetch_repo_metadata(link)
        meta_str = ""
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

    links_str = "\n".join(links_with_meta)

    # Extract link context snippets from model card
    snippets = []
    for link in candidate_links:
        start_idx = 0
        while True:
            idx = modelcard_text.find(link, start_idx)
            if idx == -1:
                break
            start = max(0, idx - 100)
            end = min(len(modelcard_text), idx + len(link) + 100)
            snippet = modelcard_text[start:end].replace('\n', ' ').strip()
            snippets.append(f"...{snippet}...")
            start_idx = idx + len(link)

    seen = set()
    unique_snippets = [s for s in snippets if not (s in seen or seen.add(s))]
    snippets_str = "\n".join(f"  - {s}" for s in unique_snippets) if unique_snippets else "  (none found)"

    # Truncated model card for context
    card_excerpt = modelcard_text[:4000]

    prompt = (
        f'For the HuggingFace model "{model_name}", identify which GitHub repository '
        f"contains the actual source code, model architecture, or training scripts "
        f"for this specific model or model family.\n\n"
        f"=== MODEL CARD (excerpt) ===\n{card_excerpt}\n\n"
        f"=== CANDIDATE REPOSITORIES (with metadata) ===\n{links_str}\n\n"
        f"=== CONTEXT (where links appear in model card) ===\n{snippets_str}\n\n"
        f"RULES:\n"
        f"- Do NOT select inference runtimes or serving tools "
        f"(text-embeddings-inference, vLLM, TGI, llama.cpp, text-generation-inference)\n"
        f"- Do NOT select dataset repositories\n"
        f"- Do NOT select generic framework repos (huggingface/transformers, "
        f"huggingface/diffusers, sentence-transformers) UNLESS this model IS "
        f"the framework's own reference/default model\n"
        f"- A repo that is a broad library (many models) is NOT the right choice "
        f"unless this specific model's training code lives there\n"
        f"- Prefer repos whose description or topics mention this model by name\n"
        f"- A forked or tiny (<50KB) repo is suspicious — verify it's relevant\n\n"
        f'Reply with ONLY the URL of the best repository, or "None" '
        f"if none of the links is appropriate."
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask an LLM to identify the main GitHub repo for a model")
    parser.add_argument("--modelcard", required=True, help="Path to a file containing the model card text")
    parser.add_argument("--model-name", required=True, help="HuggingFace model name/ID")
    parser.add_argument("--links", required=True, help="Comma-separated candidate GitHub links")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenRouter model to use (default: gpt-4o-mini)")
    args = parser.parse_args()

    with open(args.modelcard, encoding="utf-8") as f:
        modelcard_text = f.read()

    candidate_links = [link.strip() for link in args.links.split(",") if link.strip()]

    result, cost = ask_llm_github(modelcard_text, args.model_name, candidate_links, args.llm_model)
    print(f"Result: {result}")
    print(f"Cost: ${cost:.6f}")
