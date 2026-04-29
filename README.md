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
- One LLM provider credential when running with `--llm` (see Token Setup)

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
# Only one of the following is needed when running with --llm:
echo "sk-..." > keys/.openai_token         # for --provider OPENAI (default)
echo "sk-or-..." > keys/.openrouter_token  # for --provider OPENROUTER
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

## LLM Fallback

LLM-assisted classification and candidate selection is **off by default**.
Enable it with `--llm` and pick a provider with `--provider`:

```bash
python main.py --llm                       # OPENAI, gpt-4o-mini (default)
python main.py --llm --provider OPENROUTER # OpenRouter, gpt-4o-mini
python main.py --llm --provider LOCAL      # vLLM at localhost:8000 (Gemma 4 E2B)
```

For `--provider LOCAL`, start a vLLM server before running. **Gemma 4 requires
vllm nightly + transformers v5** (stable `pip install vllm` pins transformers<5
and will fail on `model_type: "gemma4"` weights). `setup.sh` installs the right
combo automatically — see [VM Setup](#vm-setup-gpu-cloud).

```bash
VLLM_USE_DEEP_GEMM=0 vllm serve google/gemma-4-E2B-it \
    --port 8000 --served-model-name gemma4 \
    --gpu-memory-utilization 0.55 --max-num-seqs 32 --max-model-len 4096 \
    --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
    --enable-prefix-caching --quantization fp8
```

Then tell the pipeline which served model name to use:

```bash
LLM_LOCAL_MODEL=gemma4 python main.py --llm --provider LOCAL
```

Override the endpoint with `LLM_LOCAL_BASE_URL` (default `http://localhost:8000/v1`).

When `--llm` is not set the pipeline emits heuristic-only answers and prints a
count of skipped LLM calls per stage.

## VM Setup (GPU Cloud)

End-to-end example for running the full pipeline with local Gemma 4 E2B on a
cloud GPU instance. Tested on AWS `g5.xlarge` (24 GB A10G) and
GCP `n1-standard-8` + T4.

### 1 — Provision the instance

Pick a GPU instance with ≥ 16 GB VRAM:

| Cloud | Instance | GPU | VRAM |
|-------|----------|-----|------|
| AWS | g5.xlarge | A10G | 24 GB |
| GCP | n1-standard-4 + T4 | T4 | 16 GB |
| Lambda | gpu_1x_a10 | A10G | 24 GB |

Use Ubuntu 22.04 LTS. CUDA 12.x drivers must be installed (most GPU images
include them; verify with `nvidia-smi`).

### 2 — Clone and set up

```bash
git clone https://github.com/<your-org>/ai-chip-resolution.git
cd ai-chip-resolution

# Install Python deps + vLLM nightly + transformers v5 + Gemma 4 E2B weights:
bash setup.sh
```

`setup.sh` prompts for your HF and API tokens, then installs **vLLM nightly**
and **transformers v5 from source** (stable `pip install vllm` cannot serve
Gemma 4 — its pin on `transformers<5` rejects `model_type: "gemma4"` weights).
It then downloads `google/gemma-4-E2B-it` via `huggingface-cli`.

Gemma 4 is a gated repo. Your HuggingFace account must have accepted the
license at <https://huggingface.co/google/gemma-4-E2B-it> before setup.sh can
download the weights. Pass `--no-vllm` to setup.sh if you only need cloud
providers (OPENAI / OPENROUTER).

### 3 — Start the vLLM server

Run this in a **separate terminal** (or tmux/screen pane):

```bash
source .venv/bin/activate

VLLM_USE_DEEP_GEMM=0 vllm serve google/gemma-4-E2B-it \
    --port 8000 --served-model-name gemma4 \
    --gpu-memory-utilization 0.55 --max-num-seqs 32 --max-model-len 4096 \
    --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
    --enable-prefix-caching --quantization fp8
```

Notes:
- `VLLM_USE_DEEP_GEMM=0` avoids a DeepGEMM FP8-warmup crash when the
  `deep_gemm` kernels aren't installed.
- `--limit-mm-per-prompt '{"image": 0, "audio": 0}'` disables Gemma 4's
  multimodal inputs (we only need text for classification).
- `--quantization fp8` roughly halves VRAM footprint.

Wait until you see `Application startup complete` before running the pipeline.

### 4 — Run the pipeline

```bash
source .venv/bin/activate

# Heuristic-only run first — fast, no GPU required:
python main.py --top 100

# Full run with Gemma 4 E2B as LLM fallback:
LLM_LOCAL_MODEL=gemma4 \
  python main.py --top 100 --llm --provider LOCAL --workers 8
```

`--workers 8` sets the asyncio concurrency cap for LLM calls; tune it against
`--max-num-seqs` on the vLLM server (keep them equal or less).

### 5 — Inspect results

```bash
# Open the dashboard (any browser on the same machine or via SSH tunnel):
python -m http.server 8080 &   # serve current dir
# then open  http://localhost:8080/ui/dashboard.html  and load database/results.json
```

### Useful env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_LOCAL_BASE_URL` | `http://localhost:8000/v1` | vLLM endpoint |
| `LLM_LOCAL_MODEL` | `google/gemma-4-E2B-it` | model name sent to vLLM (set to the `--served-model-name` alias, e.g. `gemma4`) |
| `VLLM_USE_DEEP_GEMM` | `0` | keep at 0 unless deep_gemm kernels are installed |
| `OPENAI_API_KEY` | — | alternative to `keys/.openai_token` |

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

Ground truth labels live in `tests/ground_truth.csv` and `tests/ground_truth_chinese.csv`.
Scoring runs automatically as part of `main.py`; standalone scoring against an
existing `results.json` is available via `python research/eval_score.py database/results.json`.

Run the resolver regression tests:

```bash
pytest tests/test_resolution_policy.py
```

## Dashboard

Open `ui/dashboard.html` in a browser and drop in a `results.json` file to inspect predictions and accuracy.

## Current Labels

Chip providers:

`nvidia` · `amd` · `intel` · `google_tpu` · `apple` · `aws` · `qualcomm` ·
`huawei_ascend` · `cambricon` · `baidu_kunlun` · `moore_threads` · `iluvatar` ·
`hygon` · `metax`

Frameworks:

`pytorch` · `tensorflow` · `jax` · `paddlepaddle` · `mxnet` · `onnx` · `mindspore`

Notes:
- `huawei_ascend`: Ascend NPUs (910A/B/C, Atlas 200/800/900). Triggers on `ascend`,
  `mindspore`, `cann`, `hccl`, `davinci`, `npu-smi`, `vllm-ascend`, `昇腾`.
- `cambricon`: MLU 370/590. Triggers on `cambricon`, `cnnl`, `cndrv`, `bangpy`,
  `MLUDevice`, `torch_mlu`.
- `baidu_kunlun`: Kunlun XPU (P800, R200/R300). Triggers on `kunlun(xin)`, `xpurt`,
  `paddle.set_device('xpu')`, `XPU_VISIBLE_DEVICES`, `昆仑芯`.
- `moore_threads`: MTT S3000/S4000. Triggers on `musa`, `mthreads`, `torch_musa`,
  `MUSA_VISIBLE_DEVICES`.
- `iluvatar`: BI-V100/V150 (天数智芯). Triggers on `iluvatar`, `ixrt`, `corex`,
  `BI-V100`.
- `hygon`: Hygon DCU (K100/Z100). Triggers on `hygon`, `hy-smi`, `DTK`, `海光`.
- `metax`: MetaX/Muxi C500/C550. Triggers on `metax`, `muxi`, `mxmaca`, `mx-smi`,
  `沐曦`.
- NVIDIA-China SKUs (H800, A800) remain `nvidia` — export-restricted variants of
  H100/A100, not Chinese silicon.

## Focused evaluation against a curated slice

To evaluate against a specific list of model_ids (skipping the top-N popularity sample),
pass an `--ids-file`:

```bash
python main.py --ids-file research/eval_ids.txt --workers 8
python research/eval_score.py database/results.json
```

The scorer reports per-slice accuracy (correct / missed / false_positive / confused) using
`tests/ground_truth.csv` and any `tests/ground_truth_*.csv` slice files. Diff two runs by
passing both paths:

```bash
python research/eval_score.py database/results_baseline.json database/results.json
```
