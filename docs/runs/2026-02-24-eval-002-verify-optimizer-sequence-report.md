# EVAL-002 Verify Optimizer Sequence Report

Date: 2026-02-24
Branch: `feat/dspy-rlm-evolution`
Pipeline: `verify`
Model endpoint: `http://127.0.0.1:8000/v1` (`mlx-community/gpt-oss-120b-4bit`)

## Goal

Run real DSPy optimizer sequence (`MIPROv2`, `SIMBA`, `GEPA`) for verify and persist reproducible artifacts.

## Commands

```bash
MOSAICX_OPT_PROFILE=quick scripts/run_eval_optimizer_artifacts.sh verify
```

Resolved output directory from run:

- `docs/runs/2026-02-24T155229Z-verify-optimizer-seq`

## Artifacts

- `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/optimizer_sequence_manifest.json`
- `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/baseline_metrics.json`
- `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/miprov2_optimized.json`
- `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/simba_optimized.json`
- `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/gepa_optimized.json`

## Before/After

- Baseline: train `93.33`, val `80.0`
- MIPROv2: train `93.33`, val `80.0`
- SIMBA: train `93.33`, val `80.0`
- GEPA: train `93.33`, val `80.0`

## Notes

- Quick profile bounds were applied for tractable local execution:
  - `MIPROv2`: `max_iterations=2`, `num_trials=2`, `num_candidates=2`
  - `SIMBA`: `max_iterations=4`, `num_candidates=2`, `max_steps=2`
  - `GEPA`: `max_iterations=4`, `num_candidates=2`
- SIMBA compile on verify pipeline hit DSPy multi-LM incompatibility:
  - error: `ValueError: Multiple LMs are being used in the module. There's no unique LM to return.`
  - behavior: automatic fallback to `BootstrapFewShot` for this strategy run.
  - recorded in manifest as `effective_strategy=SIMBA->BootstrapFewShot` and `compile_error`.
- GEPA remains initialized via current compatibility handling; artifacts were produced.

## Automation Hook

```bash
scripts/run_eval_optimizer_artifacts.sh verify
```
