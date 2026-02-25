# EXTX-03 Run Artifact: Outlines-First Structured Extraction Chain

Date: 2026-02-25
Branch: feat/dspy-rlm-evolution
Issue: #61

## Runtime Preconditions

1. `scripts/clear_dspy_cache.sh`
2. `scripts/ensure_vllm_mlx_server.sh`
3. `PYTHONPATH=.` from repository root

## Implementation Summary

- Added deterministic structured extraction chain in `mosaicx/pipelines/extraction.py`:
  1. `outlines_primary`
  2. `dspy_typed_direct`
  3. `dspy_json_adapter`
  4. `dspy_two_step_adapter`
  5. `existing_json_fallback`
  6. `existing_outlines_rescue`
- Chain attempts and selected path are persisted in planner diagnostics:
  - `_planner.structured_chain`
  - `_planner.selected_structured_path`
  - `_planner.structured_fallback_used`
- Preserved normalization/coercion guarantees for model-instance outputs in new chain path.

## Commands Executed

1. `PYTHONPATH=. .venv/bin/pytest -q tests/test_extraction_pipeline.py tests/test_extract_contract.py tests/test_cli_extract.py tests/test_mcp_server_dspy_config.py tests/test_mcp_query.py tests/test_mcp_verify.py tests/test_public_api.py -k "extract or planner or verify_output or query_start or query_ask or query_close"`
2. `PYTHONPATH=. .venv/bin/mosaicx extract --document tests/datasets/standardize/Sample_Report_Cervical_Spine.pdf --template MRICervicalSpineV3 -o /tmp/mosaicx_extx03_smoke.json`

## Results

- Command 1: PASS (`54 passed`, `39 deselected`)
- Command 2: PASS (`EXIT:0`)
  - `_planner` diagnostics from output JSON:
    - `planner=react`
    - `react_used=true`
    - `selected_structured_path=outlines_primary`
    - `structured_fallback_used=false`
    - `structured_chain=[{"step":"outlines_primary","ok":true}]`

## Acceptance Check

- Outlines constrained generation is primary for strict schema extraction: PASS.
- Fallback order is deterministic and logged: PASS (`structured_chain` captures ordered attempts).
- Regression tests include malformed/non-JSON recovery paths: PASS (`tests/test_extraction_pipeline.py::TestStructuredExtractionChain`).
