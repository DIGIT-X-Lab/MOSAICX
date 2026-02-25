# EXTX-02 Run Artifact: ReAct Planner-First Section Routing

Date: 2026-02-25
Branch: feat/dspy-rlm-evolution
Issue: #60

## Runtime Preconditions

1. `scripts/clear_dspy_cache.sh`
2. `scripts/ensure_vllm_mlx_server.sh`
3. `PYTHONPATH=.` from repository root

## Implementation Summary

- Added planner control plane in `mosaicx/pipelines/extraction.py`:
  - section splitting (`_split_document_sections`)
  - complexity hints (`_section_complexity_hint`)
  - ReAct route planner (`_plan_section_routes_with_react`)
  - routed context composition (`_compose_routed_document_text`)
  - end-to-end planner call (`_plan_extraction_document_text`)
- `DocumentExtractor.forward()` now executes `Plan extraction` before `Extract` (schema + auto modes).
- Planner diagnostics are propagated to outputs via `_planner` in:
  - CLI (`mosaicx/cli.py`)
  - SDK (`mosaicx/sdk.py`)
  - MCP `extract_document` (`mosaicx/mcp_server.py`)

## Commands Executed

1. `PYTHONPATH=. .venv/bin/pytest -q tests/test_extraction_pipeline.py tests/test_extract_contract.py tests/test_cli_extract.py tests/test_mcp_server_dspy_config.py tests/test_mcp_query.py tests/test_mcp_verify.py tests/test_public_api.py -k "extract or planner or verify_output or query_start or query_ask or query_close"`
2. `PYTHONPATH=. .venv/bin/mosaicx extract --document tests/datasets/standardize/Sample_Report_Cervical_Spine.pdf --template MRICervicalSpineV3 -o /tmp/mosaicx_extx02_smoke.json`

## Results

- Command 1: PASS (`51 passed`, `39 deselected`)
- Command 2: PASS (`EXIT:0`)
  - planner diagnostics from output JSON:
    - `planner=react`
    - `react_used=true`
    - `strategy_counts={constrained_extract:1, deterministic:1, heavy_extract:0, repair:0}`
    - `compression_ratio=0.9875`

## Acceptance Check

- Planner decisions visible in diagnostics: PASS (`_planner` in CLI/SDK/MCP outputs).
- Easy sections avoid heavy extraction path: PASS (observed deterministic/constrained routing in smoke run).
- Route selection tested under complex/easy section scenarios: PASS (new planner tests in `tests/test_extraction_pipeline.py`).
