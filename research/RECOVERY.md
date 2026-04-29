# Recovery / reproduction guide

This run produced the snapshot in `database/snapshot_<date>/`. To reproduce
or re-run:

## 1. Set up env (already done in `.venv`)

```bash
bash setup.sh                 # creates .venv, installs vLLM nightly + Gemma 4
```

## 2. Refresh the source CSV (HuggingFace's full model index — slow)

```bash
.venv/bin/python scripts/ingest/get_models.py
.venv/bin/python scripts/variation_cleaner.py    # writes models_dedup.csv
```

## 3. Boot the local vLLM server (separate terminal)

```bash
VLLM_USE_DEEP_GEMM=0 .venv/bin/vllm serve google/gemma-4-E2B-it \
    --port 8000 --served-model-name gemma4 \
    --gpu-memory-utilization 0.55 --max-num-seqs 32 --max-model-len 4096 \
    --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
    --enable-prefix-caching --quantization fp8
```

## 4. Run the production sweep

```bash
LLM_LOCAL_MODEL=gemma4 .venv/bin/python main.py \
    --quarters 2022-2026 --top 50 \
    --source-csv database/models_dedup.csv \
    --workers 16 --llm --provider LOCAL
```

This samples the top 50 most-downloaded model FAMILIES per quarter (variants
collapsed into a single representative — see `scripts/variation_cleaner.py`),
across 2022 Q1 through 2026 Q4. Up to 50 × 20 = 1000 models depending on
how many quarters had that many qualifying releases.

## 5. Bundle the snapshot

```bash
.venv/bin/python scripts/save_snapshot.py --label final
```

Writes `database/snapshot_final/` containing every artifact + `SUMMARY.txt`
+ `RUN_COMMAND.txt`.

## 6. Inspect the dashboard

```bash
.venv/bin/python -m http.server 8080 --bind 127.0.0.1 &
# Open http://localhost:8080/ui/dashboard.html in a browser, drag in
# database/results.json (or hit a HuggingFace dataset URL the page is
# wired to fetch from).
```

## Eval against the curated 78-row ground-truth slice

```bash
LLM_LOCAL_MODEL=gemma4 .venv/bin/python main.py \
    --ids-file research/eval_ids.txt --workers 16 \
    --llm --provider LOCAL
.venv/bin/python research/eval_score.py database/results.json
```

Last measured eval result on this slice: **72/78 (92.3%)** correct,
**1 false positive**, **97.3%** precision-at-known.
