import os
import re
import sys

from llm_client import complete_async

# Allow calling from arbitrary entry points: ensure classifiers/ is importable.
_CLASSIFIERS = os.path.join(os.path.dirname(__file__), "..", "classifiers")
if _CLASSIFIERS not in sys.path:
    sys.path.insert(0, _CLASSIFIERS)
from signals import snippet_is_training_disclosure, snippet_is_training_context  # noqa: E402

VALID_CHIPS = {
    "nvidia", "amd", "intel", "google_tpu", "apple", "aws", "qualcomm",
}

# Confidence mapping. Tuned for Gemma 4 E4B, which is well-calibrated enough that
# `low` answers still carry real information when accompanied by a concrete quote.
CONFIDENCE_MAP = {
    "high": 0.8,
    "medium": 0.6,
    "low": 0.4,
}

# A valid training-evidence quote must contain either a specific chip/accelerator
# name OR an explicit training-language token. Weaker than "any chip literal" —
# we want the LLM's conclusion to hinge on a concrete clue, not generic "GPU".
_CHIP_OR_TRAINING_RE = re.compile(
    r'\b(?:A100|H100|V100|H200|P100|T4|L40[Ss]?|A6000|A10[Gg]?|DGX|'
    r'MI\d{3}[Xx]?|Instinct|Gaudi|Habana|Xeon|'
    r'TPU(?:\s*v\d+)?|Trainium|Inferentia|Neuron|'
    r'M[1-4](?:\s*(?:Pro|Max|Ultra))?|MLX|CoreML|OpenVINO|'
    r'trained\s+(?:on|using|with)|training\s+(?:hardware|infrastructure|utilized|cost[s]?|compute)|'
    r'GPU\s+hours?|fine[- ]?tun(?:ed|ing)\s+(?:on|using|with)|'
    r'pre[- ]?train(?:ed|ing)\s+(?:on|using|with)|compute\s+cluster)\b',
    re.IGNORECASE,
)

# Hypothetical / user-instruction language that should NOT be treated as disclosure.
_CONDITIONAL_RE = re.compile(
    r'\b(?:can|could|may|might|should|would|recommend(?:ed)?|if\s+you|users?\s+can|'
    r'allow[s]?\s+(?:you|users)|able\s+to|capable\s+of)\b[^.]{0,80}?'
    r'\b(?:be\s+)?(?:trained?|fine[- ]?tuned|pre[- ]?trained|run|deployed)\b',
    re.IGNORECASE,
)


def _filter_disclosure_snippets(snippets):
    """Surface snippets that contain a non-hypothetical training phrase. The LLM
    decides whether each quote is about *this* model's training vs a dataset or
    a user-fine-tuning suggestion. Strict chip-literal co-location would miss
    the common 'trained on N GPUs' / 'our cluster of X' phrasings."""
    if not snippets:
        return []
    kept = []
    for s in snippets:
        if snippet_is_training_disclosure(s) or snippet_is_training_context(s):
            kept.append(s)
    return kept


def _format_snippets(snippets, label, max_chars):
    if not snippets:
        return ""
    lines = []
    budget = max_chars
    seen = set()
    for s in snippets:
        text = s.get("snippet", "") if isinstance(s, dict) else str(s)
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        src = ""
        if isinstance(s, dict):
            src = s.get("file", "") or s.get("section", "")
        piece = f"- {text}" + (f"  [{src}]" if src else "")
        if len(piece) > budget:
            piece = piece[:budget]
        lines.append(piece)
        budget -= len(piece) + 1
        if budget <= 0:
            break
    if not lines:
        return ""
    return f"\n=== {label} ===\n" + "\n".join(lines)


def _build_prompt(model_name, yaml_library, framework, section_labeled_text,
                  modelcard_excerpt, extra_context,
                  github_snippets=None, arxiv_snippets=None):
    # vLLM's 4096-token context means the full prompt must stay under ~12000 chars
    # (prompt skeleton ~800, instruction block ~1800, leaving ~9400 for card text).
    # Split the evidence budget: card gets the biggest slice, github + arxiv each get
    # a smaller slice. Training disclosures often live in github scripts (CUDA_VISIBLE_
    # _DEVICES, torchrun) or arXiv papers (H100 clusters), not in modelcards.
    card_text = section_labeled_text[:3500] if section_labeled_text else modelcard_excerpt[:2500]
    # Only pass GitHub/arXiv snippets that already pass the strict co-location
    # check — raw top-K snippets include `torch.cuda` runtime noise that Gemma
    # reads as training evidence (e.g. regressed gpt-oss-20b).
    gh_filtered = _filter_disclosure_snippets(github_snippets)
    ax_filtered = _filter_disclosure_snippets(arxiv_snippets)
    gh_block = _format_snippets(gh_filtered, "EXTERNAL TRAINING DISCLOSURE (GITHUB)", 1500)
    ax_block = _format_snippets(ax_filtered, "EXTERNAL TRAINING DISCLOSURE (ARXIV)", 1200)
    return (
        f'You are classifying the hardware the model "{model_name}" was TRAINED on '
        f'(not the hardware a user runs it on).\n\n'
        f"Known facts:\n"
        f"- YAML library_name: {yaml_library or 'not specified'}\n"
        f"- Heuristic-detected framework: {framework or 'unknown'}\n"
        f"{extra_context}\n\n"
        f"=== MODEL CARD ===\n{card_text}"
        f"{gh_block}"
        f"{ax_block}\n\n"
        f"COUNTS AS TRAINING EVIDENCE (commit to a chip):\n"
        f"- A sentence or table cell naming a specific accelerator AND training context: "
        f"\"trained on 8xA100\", \"H100 GPU hours\", \"TPU v4-128 pod\", \"MI250X cluster\", "
        f"\"fine-tuned on H100\" — IF it refers to *this* model's own training.\n"
        f"- A cost/compute table row labelled \"Training Cost\", \"GPU hours\", or "
        f"\"Training Factors\" that names a chip (e.g. \"A100 80GB GPU hours | 1000\").\n"
        f"- An explicit \"Hardware and Software\" / \"Training Infrastructure\" section "
        f"that names a chip.\n\n"
        f"DOES NOT COUNT — return `unknown`:\n"
        f"- Inference / deployment / runtime mentions: \"runs on CUDA\", \"supports H100\", "
        f"\"compatible with TPU\", \"device_map='cuda:0'\", \"works on M2 Mac\".\n"
        f"- Hypothetical user fine-tuning: \"can be fine-tuned on H100\", "
        f"\"users may train on A100\", \"recommended: 4xH100\".\n"
        f"- A mention of a sibling/larger model's training hardware, unless this model "
        f"shares the same disclosure.\n"
        f"- Just a framework hint (\"transformers\", \"pytorch\") with no chip named.\n"
        f"- Generic \"GPU\" or \"CUDA\" without a specific chip or training quote.\n\n"
        f"BE CONSERVATIVE: when the card is ambiguous or silent on training hardware, "
        f"answer `unknown`. Never default to `nvidia` just because the framework is pytorch.\n\n"
        f"You may quote from the MODEL CARD or from any EXTERNAL TRAINING DISCLOSURE "
        f"block shown above. The external blocks have been pre-filtered to contain only "
        f"co-located training+hardware language, so if one names a specific chip, trust "
        f"it as disclosure of this model's training hardware.\n\n"
        f"Reply in EXACTLY this format (3 lines, nothing else):\n"
        f"training_evidence: <short verbatim quote copy-pasted from any section above; "
        f"or 'none found' if no qualifying sentence/cell exists>\n"
        f"conclusion: <one of: {', '.join(sorted(VALID_CHIPS))}, unknown>\n"
        f"confidence: <high | medium | low>\n"
    )


def _norm(text):
    if not text:
        return ""
    # Lower, collapse whitespace, drop markdown table pipes / asterisks / backticks.
    t = text.strip().lower()
    t = re.sub(r'[`*|_>#]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t


_STOPWORDS = {
    "a", "an", "and", "the", "of", "on", "in", "to", "for", "with", "was",
    "is", "are", "we", "our", "by", "at", "be", "this", "that", "or", "from",
    "as", "it", "its", "per",
}


def _token_overlap(evidence, card_text):
    """Fraction of non-stopword content tokens from evidence that appear in card."""
    ev = _norm(evidence)
    card = _norm(card_text)
    if not ev or not card:
        return 0.0
    ev_tokens = [t for t in re.findall(r'\w+', ev) if t not in _STOPWORDS]
    if not ev_tokens:
        return 0.0
    card_tokens = set(re.findall(r'\w+', card))
    hits = sum(1 for t in ev_tokens if t in card_tokens)
    return hits / len(ev_tokens)


def _evidence_in_card(evidence, card_text):
    """Accept if (a) evidence is a substring of the normalized card, (b) any long
    run of the evidence appears verbatim, or (c) ≥80% of the evidence's content
    tokens appear in the card. (c) handles LLM quotes stitched from table cells
    (e.g. "in A100 80GB GPU hours | 1000" — real card has "... | 500 | 500 | 1000")."""
    ev = _norm(evidence)
    card = _norm(card_text)
    if not ev or not card:
        return False
    if ev in card:
        return True
    if len(ev) >= 25:
        for start in range(0, max(1, len(ev) - 25), 5):
            if ev[start:start + 30] in card:
                return True
    return _token_overlap(evidence, card_text) >= 0.8


def _parse_answer(answer, card_text=""):
    conclusion = None
    confidence_label = "low"
    training_evidence = ""

    for line in answer.split("\n"):
        raw = line.strip()
        lower = raw.lower()
        if lower.startswith("training_evidence:"):
            training_evidence = raw.split(":", 1)[1].strip().strip("'\"")
        elif lower.startswith("conclusion:"):
            value = lower.split(":", 1)[1].strip().strip(".").strip()
            if value in VALID_CHIPS:
                conclusion = value
            elif value == "unknown":
                conclusion = None
        elif lower.startswith("confidence:"):
            confidence_label = lower.split(":", 1)[1].strip().strip(".").strip()

    if conclusion is None:
        single = answer.strip().lower()
        if single in VALID_CHIPS:
            conclusion = single

    confidence = CONFIDENCE_MAP.get(confidence_label, 0.0)

    if conclusion is None:
        return None, 0.0

    ev_lower = training_evidence.lower()
    no_ev = (
        not training_evidence
        or ev_lower in {"none found", "none", "n/a", "na", "not specified", "unknown", "not found"}
    )

    # Reject answers that commit to a chip without a concrete evidence quote.
    if no_ev:
        return None, 0.0

    # Reject quotes that contain only conditional / hypothetical phrasing about
    # users training ("can be fine-tuned on H100") — that's not a disclosure.
    if _CONDITIONAL_RE.search(training_evidence) and not re.search(
        r'\b(?:training\s+cost|gpu\s+hours?|training\s+utilized|training\s+factors?)\b',
        training_evidence, re.IGNORECASE,
    ):
        return None, 0.0

    # Reject quotes that don't contain any chip name or training phrase.
    if not _CHIP_OR_TRAINING_RE.search(training_evidence):
        return None, 0.0

    # Reject quotes the LLM fabricated (not in the card).
    if card_text and not _evidence_in_card(training_evidence, card_text):
        return None, 0.0

    return conclusion, confidence


async def ask_llm_chip(
    model_name,
    yaml_library=None,
    framework=None,
    modelcard_excerpt="",
    section_labeled_text="",
    extra_context="",
    github_snippets=None,
    arxiv_snippets=None,
):
    """Ask the configured LLM to determine the training chip provider.

    github_snippets / arxiv_snippets are the chip-centered snippets surfaced by
    the respective classifiers (each item: {"snippet": str, "file"/"section": str}).

    Returns (chip_provider_str_or_None, confidence_float, cost_float).
    """
    prompt = _build_prompt(
        model_name, yaml_library, framework,
        section_labeled_text, modelcard_excerpt, extra_context,
        github_snippets=github_snippets, arxiv_snippets=arxiv_snippets,
    )
    # Evidence for substring verification spans card + all snippets we passed in.
    corpus_parts = [section_labeled_text or modelcard_excerpt or ""]
    for s in (github_snippets or []) + (arxiv_snippets or []):
        corpus_parts.append(s.get("snippet", "") if isinstance(s, dict) else str(s))
    card_text = "\n".join(corpus_parts)
    answer, cost = await complete_async([{"role": "user", "content": prompt}])
    conclusion, confidence = _parse_answer(answer, card_text=card_text)
    return conclusion, confidence, cost
