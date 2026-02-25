# EXTX-04 Run Artifact: Selective BestOfN for Uncertain Extraction Sections

Date: 2026-02-25
Branch: feat/dspy-rlm-evolution
Issue: #62

## Runtime Preconditions

1. `scripts/clear_dspy_cache.sh`
2. `scripts/ensure_vllm_mlx_server.sh`
3. `PYTHONPATH=.` from repository root

## Implementation Summary

- Added selective BestOfN path in `mosaicx/pipelines/extraction.py`:
  - trigger gate: planner routes containing `heavy_extract` or `repair`
  - candidate selector: `_try_bestofn_for_uncertain_sections(...)`
  - deterministic reward components:
    - schema compliance
    - evidence overlap
    - critical-field completeness
    - contradiction penalty
    - null-overuse penalty
- BestOfN diagnostics are now attached in `_planner.bestofn`.
- Structured chain now records BestOfN step when triggered (`bestofn_uncertain`).

## Commands Executed

1. `PYTHONPATH=. .venv/bin/pytest -q tests/test_extraction_pipeline.py tests/test_extract_contract.py tests/test_cli_extract.py tests/test_mcp_server_dspy_config.py tests/test_mcp_query.py tests/test_mcp_verify.py tests/test_public_api.py -k "extract or planner or verify_output or query_start or query_ask or query_close"`
2. `PYTHONPATH=. .venv/bin/mosaicx extract --document tests/datasets/standardize/Sample_Report_Cervical_Spine.pdf --template MRICervicalSpineV3 -o /tmp/mosaicx_extx04_smoke2.json`

## Results

- Command 1: PASS (`58 passed`, `39 deselected`)
- Command 2: PASS (`EXIT:0`)
  - `_planner` diagnostics from output JSON:
    - `selected_structured_path=outlines_primary`
    - `bestofn.triggered=false`
    - `bestofn.reason=skipped_outlines_primary_succeeded`

## Acceptance Check

- BestOfN only triggers for planner-flagged uncertain sections: PASS (unit tests cover trigger/no-trigger behavior).
- Deterministic reward components covered by tests: PASS.
- BestOfN diagnostics and chain order visible in output: PASS.
