import argparse
import asyncio
import json
import os
import re
import sys
from difflib import SequenceMatcher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm"))
from ask_llm_github import ask_llm_github
from llm_client import llm_enabled, set_concurrency

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "modelcards.json")

GENERIC_REPOS = {
    "huggingface/transformers",
    "huggingface/text-embeddings-inference",
    "huggingface/huggingface-llama-recipes",
    "huggingface/trl",
    "daoailab/flashattention",
}

_DATASET_REPO_RE = re.compile(
    r'dataset|data[-_]?set|corpus|benchmark|eval[-_]?data',
    re.IGNORECASE,
)

ORG_ALIASES = {
    "openai": {"openai"},
    "qwen": {"qwenlm"},
    "baai": {"flagopen"},
    "facebookai": {"pytorch", "facebookresearch", "facebook"},
    "facebook": {"pytorch", "facebookresearch", "facebookai"},
    "meta-llama": {"meta-llama", "facebookresearch"},
    "amazon": {"amazon-science"},
    "autogluon": {"amazon-science"},
    "hexgrad": {"hexgrad"},
    "pyannote": {"pyannote"},
    "coqui": {"coqui-ai"},
    "google-bert": {"google-research"},
    "google": {"google-research", "google-research-datasets"},
    "laion": {"laion-ai"},
    "sentence-transformers": {"ukplab"},
    "colbert-ir": {"stanford-futuredata"},
    "omni-research": {"bytedance"},
    "distilbert": {"huggingface"},
    "cross-encoder": {"ukplab"},
    "timm": {"huggingface"},
    "usyd-community": {"vitaetransformer"},
    "nomic-ai": {"nomic-ai"},
}

INTRO_KEYWORDS_RE = re.compile(
    r'(this repository|first released in|GitHub[:\s]|code[- ]?base|source code|repo:|Repository:|Original:)',
    re.IGNORECASE,
);

PIP_INSTALL_RE = re.compile(r'pip install[^\n]*github\.com', re.IGNORECASE);

TABLE_ROW_RE = re.compile(r'\|.*\|');
BIBTEX_RE = re.compile(r'```bibtex');


def normalize(text):
    return re.sub(r'[^a-z0-9]', '', text.lower());


def fuzzy_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio();


def strip_version(name):
    return re.sub(r'[-_]?(v?\d+[\d.]*[bBmM]?|base|large|small|tiny|mini|instruct|uncased|cased|fused|bolt)$', '', name, flags=re.IGNORECASE);


def model_name_tokens(model_id):
    parts = model_id.split("/");
    org = parts[0] if len(parts) > 1 else "";
    name = parts[-1];
    name_stripped = strip_version(name);
    name_stripped = strip_version(name_stripped);
    return normalize(org), normalize(name), normalize(name_stripped);


def repo_tokens(url):
    match = re.match(r'https?://github\.com/([^/]+)/([^/]+)', url);
    if not match:
        return "", ""
    return normalize(match.group(1)), normalize(match.group(2));


def is_in_table(card, link_pos):
    line_start = card.rfind('\n', 0, link_pos) + 1;
    line_end = card.find('\n', link_pos);
    if line_end == -1:
        line_end = len(card);
    line = card[line_start:line_end];
    return bool(TABLE_ROW_RE.match(line.strip()));


def is_in_bibtex(card, link_pos):
    bibtex_starts = [m.start() for m in BIBTEX_RE.finditer(card)];
    for bib_start in bibtex_starts:
        bib_end = card.find('```', bib_start + 3);
        if bib_end == -1:
            bib_end = len(card);
        if bib_start <= link_pos <= bib_end:
            return True
    return False


def relative_position(card, link_pos):
    total = len(card);
    if total == 0:
        return 0.5
    return link_pos / total;


def has_intro_context(card, link_pos, window=200):
    start = max(0, link_pos - window);
    end = min(len(card), link_pos + len("https://github.com/x/y") + window);
    snippet = card[start:end];
    return bool(INTRO_KEYWORDS_RE.search(snippet));


def score_link(link, model_id, card):
    score = 0;
    reasons = [];

    link_clean = link.rstrip('/.');
    gh_org, gh_repo = repo_tokens(link_clean);
    hf_org, hf_name, hf_name_stripped = model_name_tokens(model_id);

    owner_repo = f"{gh_org}/{gh_repo}";
    if owner_repo in {normalize(g) for g in GENERIC_REPOS}:
        if gh_org != hf_org:
            score -= 5;
            reasons.append("generic framework repo");

    link_pos = card.find(link);
    if link_pos == -1:
        link_pos = len(card) // 2;

    pip_matches = list(PIP_INSTALL_RE.finditer(card));
    link_only_in_pip = False;
    if pip_matches:
        for pm in pip_matches:
            if link in card[pm.start():pm.end() + 100]:
                all_occurrences = [m.start() for m in re.finditer(re.escape(link), card)];
                pip_region_starts = {pm2.start() for pm2 in pip_matches};
                link_only_in_pip = all(any(abs(occ - ps) < 200 for ps in pip_region_starts) for occ in all_occurrences);
                break
    if link_only_in_pip:
        score -= 3;
        reasons.append("only in pip install context");

    if hf_name_stripped and gh_repo:
        if hf_name_stripped in gh_repo or gh_repo in hf_name_stripped:
            score += 4;
            reasons.append("model name in repo name");
        elif fuzzy_ratio(hf_name_stripped, gh_repo) > 0.6:
            score += 2;
            reasons.append("fuzzy model name match");

    if hf_org == gh_org:
        score += 3;
        reasons.append("org exact match");
    else:
        aliases = ORG_ALIASES.get(hf_org, set());
        if gh_org in aliases:
            score += 3;
            reasons.append("org alias match");

    rel_pos = relative_position(card, link_pos);
    if rel_pos < 0.15:
        score += 2;
        reasons.append("very early in card");
    elif rel_pos < 0.35:
        score += 1;
        reasons.append("early in card");

    if has_intro_context(card, link_pos):
        score += 3;
        reasons.append("intro keyword context");

    if is_in_table(card, link_pos):
        score -= 2;
        reasons.append("inside table row");

    if is_in_bibtex(card, link_pos):
        score -= 3;
        reasons.append("inside bibtex");

    return score, reasons;


def compute_confidence(best_score, second_score, num_links):
    if num_links == 0:
        return 0.0
    if num_links == 1:
        return min(1.0, 0.7 + max(0, best_score) * 0.03)

    gap = best_score - second_score;
    base = 0.5;
    base += min(0.3, gap * 0.06);
    base += min(0.15, max(0, best_score) * 0.015);
    return round(min(1.0, max(0.1, base)), 2);


LLM_CONFIDENCE_THRESHOLD = 0.60;


def _score_model_heuristic(model):
    """Heuristic scoring for one model (no LLM). Mutates model in place."""
    links = model.get("github_links", []);
    card = model.get("modelcard", "");
    model_id = model.get("id", "");

    if not links:
        model["main_github"] = None;
        model["main_github_confidence"] = 0.0;
        return

    if len(links) == 1:
        s, r = score_link(links[0], model_id, card);
        conf = compute_confidence(s, -10, 1);
        _, gh_repo_name = repo_tokens(links[0]);
        if _DATASET_REPO_RE.search(gh_repo_name):
            conf = min(conf, 0.5);
        model["main_github"] = links[0];
        model["main_github_confidence"] = conf;
        return

    scored = [];
    for link in links:
        s, r = score_link(link, model_id, card);
        scored.append((s, r, link));
    scored.sort(key=lambda x: x[0], reverse=True);

    best_score, best_reasons, best_link = scored[0];
    second_score = scored[1][0] if len(scored) > 1 else -10;

    conf = compute_confidence(best_score, second_score, len(links));
    _, gh_repo_name = repo_tokens(best_link);
    if _DATASET_REPO_RE.search(gh_repo_name):
        conf = min(conf, 0.5);
    model["main_github"] = best_link;
    model["main_github_confidence"] = conf;


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for LLM fallback calls (default: 4)")
    args = parser.parse_args()

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f);

    # Pass 1: heuristic scoring (fast, no network calls)
    for model in data:
        _score_model_heuristic(model);

    # Pass 2: LLM fallback for low-confidence picks (parallelized)
    model_by_id = {m["id"]: m for m in data};
    llm_queue = [
        (m["id"], m.get("modelcard", ""), m.get("github_links", []))
        for m in data
        if m.get("main_github_confidence", 1.0) < LLM_CONFIDENCE_THRESHOLD
        and m.get("github_links", [])
    ];

    total_llm_cost = 0.0;
    if not llm_enabled():
        if llm_queue:
            print(f"  LLM disabled (--llm not set). Skipping {len(llm_queue)} GitHub candidate-selection call(s).");
    else:
        set_concurrency(args.workers)

        async def _run_all_llm():
            async def _one(item):
                model_id, card, links = item
                try:
                    llm_result, cost = await ask_llm_github(card, model_id, links)
                    return model_id, llm_result, cost, None
                except Exception as e:
                    return model_id, None, 0.0, str(e)
            return await asyncio.gather(*[_one(item) for item in llm_queue])

        for model_id, llm_result, cost, err in asyncio.run(_run_all_llm()):
            total_llm_cost += cost;
            if err:
                print(f"  LLM failed for {model_id}: {err}");
            elif llm_result:
                model_by_id[model_id]["main_github"] = llm_result;
                model_by_id[model_id]["main_github_source"] = "llm";
                print(f"  LLM override for {model_id}: {llm_result} (Cost: ${cost:.6f})");

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False);

    print("Done. Results:");
    for model in data:
        conf = model.get("main_github_confidence", 0);
        link = model.get("main_github", "N/A");
        print(f"  {model['id']:50s} -> {str(link):60s}  (confidence: {conf})");

    if total_llm_cost > 0:
        print(f"\nTotal LLM Cost: ${total_llm_cost:.6f}");


if __name__ == "__main__":
    main()

