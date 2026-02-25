# EXTX-01 Run Artifact: Canonical Extract Contract Propagation

Date: 2026-02-25
Branch: feat/dspy-rlm-evolution
Issue: #59

## Runtime Preconditions

1. `scripts/clear_dspy_cache.sh`
2. `scripts/ensure_vllm_mlx_server.sh`
3. `PYTHONPATH=.` test invocation from repo root

## Commands Executed

1. `PYTHONPATH=. .venv/bin/pytest -q tests/test_extract_contract.py tests/test_extraction_pipeline.py tests/test_sdk_envelope.py tests/test_mcp_server_dspy_config.py`
2. `PYTHONPATH=. .venv/bin/pytest -q tests/test_mcp_query.py tests/test_mcp_verify.py tests/test_public_api.py -k "extract or verify_output or query_start or query_ask or query_close"`

## Results

- Command 1: PASS (`24 passed`)
- Command 2: PASS (`27 passed`, `37 deselected`)

## Implementation Scope Verified

- Canonical extraction contract attached in CLI extract path.
- Canonical extraction contract attached in SDK extract path.
- Canonical extraction contract attached in MCP `extract_document` tool.
- Contract semantics covered:
  - `supported`
  - `needs_review`
  - `insufficient_evidence`

## Notes

- Tests use deterministic mocks for extractor outputs to avoid brittle model/runtime dependencies while validating contract propagation logic.
- Local vLLM preflight is still executed before the test runs per project runtime rule.
