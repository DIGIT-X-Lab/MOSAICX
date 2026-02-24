# EVAL-002 Optimizer Sequence Report

Date: 2026-02-24
Branch: `feat/dspy-rlm-evolution`
Pipeline: `query`
Model endpoint: `http://127.0.0.1:8000/v1` (`mlx-community/gpt-oss-120b-4bit`)

## Goal

Run real DSPy optimizer sequence (`MIPROv2`, `SIMBA`, `GEPA`) and persist reproducible artifacts with command provenance.

## Commands

```bash
scripts/ensure_vllm_mlx_server.sh
PYTHONPATH=. .venv/bin/python scripts/run_optimizer_sequence.py \
  --pipeline query \
  --trainset tests/datasets/evaluation/query_optimizer_tiny_train.jsonl \
  --valset tests/datasets/evaluation/query_optimizer_tiny_val.jsonl \
  --profile quick \
  --out-dir docs/runs/2026-02-24-query-optimizer-seq-tiny
```

Baseline metrics were computed and written after the run:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
# computes baseline train/val and updates optimizer_sequence_manifest.json
PY
```

## Artifacts

- `docs/runs/2026-02-24-query-optimizer-seq-tiny/optimizer_sequence_manifest.json`
- `docs/runs/2026-02-24-query-optimizer-seq-tiny/baseline_metrics.json`
- `docs/runs/2026-02-24-query-optimizer-seq-tiny/miprov2_optimized.json`
- `docs/runs/2026-02-24-query-optimizer-seq-tiny/simba_optimized.json`
- `docs/runs/2026-02-24-query-optimizer-seq-tiny/gepa_optimized.json`

## Before/After

- Baseline: train `100.0`, val `100.0`
- MIPROv2 optimized: train `100.0`, val `100.0`
- SIMBA optimized: train `100.0`, val `100.0`
- GEPA optimized: train `100.0`, val `100.0`

## Notes

- Run used bounded `quick` profile to keep local 120B execution tractable:
  - `MIPROv2`: `max_iterations=2`, `num_trials=2`, `num_candidates=2`
  - `SIMBA`: `max_iterations=4`, `num_candidates=2`, `max_steps=2`
  - `GEPA`: `max_iterations=4`, `num_candidates=2`
- In this DSPy version, `GEPA` initialization falls back to `BootstrapFewShot` when unsupported args are rejected; fallback was observed during run and artifact was still produced.
- Several `RLM reached max iterations` warnings appeared; DSPy handled them via extract fallback and completed run.

## Automation Hook

Use periodic artifact refresh via:

```bash
scripts/run_eval_optimizer_artifacts.sh query
```

Recommended cron example (daily at 03:00 local):

```cron
0 3 * * * cd /Users/nutellabear/Documents/00-Code/MOSAICX && scripts/run_eval_optimizer_artifacts.sh query >> /tmp/mosaicx_optimizer_cron.log 2>&1
```
