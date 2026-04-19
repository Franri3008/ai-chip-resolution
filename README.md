# Model Hardware Classifier

This project maps Hugging Face models to their likely training hardware provider and ML framework by combining evidence from model cards, GitHub repositories, and arXiv papers.

The pipeline is evidence-driven by design:
- Prefer explicit training disclosures over runtime or deployment mentions.
- Return `unknown` when evidence is weak or ambiguous.
- Avoid model-specific answer tables or paper-specific hardcoded outcomes.

## Requirements

- Python 3.12+
- Hugging Face token
- GitHub token
- OpenRouter token for optional LLM fallbacks

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local development and tests:

```bash
pip install -r requirements-dev.txt
```

## Token Setup

Create token files inside `keys/`:

```bash
mkdir -p keys
echo "hf_..." > keys/.hf_token
echo "ghp_..." > keys/.gh_token
echo "sk-or-..." > keys/.openrouter_token
```

See `keys/README.md` for details. Secret token files are ignored by git.

## Quick Start

Run the full pipeline:

```bash
python main.py
```

Refresh the source model list first:

```bash
python main.py --update-models
```

Limit the run to the top `N` models:

```bash
python main.py --top 50
```

## Pipeline

`main.py` orchestrates the pipeline in this order:

1. `scripts/ingest/get_models.py`
2. `scripts/ingest/get_modelcard.py`
3. `scripts/ingest/get_github.py`
4. `scripts/ingest/get_arxiv.py`
5. `scripts/classifiers/evaluate_github.py`
6. `scripts/classifiers/evaluate_arxiv.py`
7. `scripts/classifiers/from_modelcard.py`
8. `scripts/classifiers/from_githubcode.py`
9. `scripts/classifiers/from_arxiv.py`
10. `main.py` result aggregation and optional evaluation

## Repository Layout

- `main.py`: pipeline entrypoint and final decision policy
- `scripts/ingest/`: data collection from external sources
- `scripts/classifiers/`: heuristic source scoring and classification
- `scripts/llm/`: LLM-assisted fallback logic
- `tests/`: regression tests and ground-truth labels
- `ui/dashboard.html`: local result inspection UI

## Outputs

Generated artifacts are written to `database/` during a run:

- `results.json`: final classifications plus trace metadata
- `modelcard_chip_analysis.json`: model-card scoring output
- `github_chip_analysis.json`: GitHub code scoring output
- `arxiv_chip_analysis.json`: paper scoring output

These files are treated as generated artifacts and are ignored by default.

## Evaluation

Ground truth labels live in `tests/ground_truth.csv`.

To annotate an existing `results.json` without rerunning the pipeline:

```bash
python scripts/add_ground_truth.py
```

Run the resolver regression tests:

```bash
pytest tests/test_resolution_policy.py
```

## Dashboard

Open `ui/dashboard.html` in a browser and drop in a `results.json` file to inspect predictions and accuracy.

## What To Commit

Recommended:
- source code
- tests
- configuration
- documentation

Do not commit:
- API tokens
- local caches
- generated `database/*.json` artifacts
- interpreter or pytest caches

## Current Labels

Chip providers:

`nvidia` · `amd` · `intel` · `google_tpu` · `apple` · `aws` · `qualcomm`

Frameworks:

`pytorch` · `tensorflow` · `jax` · `paddlepaddle` · `mxnet` · `onnx`
