# Query + Verify Hardening Progress (Atomic Tracker)

Last updated: 2026-02-23
Branch: `feat/dspy-rlm-evolution`
Owner: Codex
Mode: Execute + verify + push

## Goal
Make `verify --level thorough` and `query` evidence behavior robust, chunk-aware, and consistently interpretable for developers.

## Atomic Tasks
- [x] `Q1` Long-document chunk retrieval tools for query (`search_document_chunks`, `get_document_chunk`, citations metadata)
Status: `done`
Commit: `7ee4753`

- [x] `V1` Strengthen verify thorough chunk search semantics (term-aware matching + ranked chunk evidence)
Status: `done`

- [x] `V2` Parse structured evidence metadata from audit reports into `FieldVerdict` (chunk/span/score/source)
Status: `done`

- [x] `V3` Standardize SDK verify evidence payload (`citations`) for claim/extraction modes
Status: `done`

- [x] `V4` Unify CLI verify evidence display semantics with query (clear typed evidence table + location)
Status: `done`

- [x] `T1` Add/adjust tests for audit parsing, sdk verify payload, and cli verify display
Status: `done`

- [x] `R1` Run focused test suites (`verify`, `sdk_verify`, `cli_verify`, `query smoke`) and fix regressions
Status: `done`
Run command: `.venv/bin/pytest -q tests/test_verify_models.py tests/test_verify_audit.py tests/test_verify_engine.py tests/test_sdk_verify.py tests/test_cli_verify.py tests/test_mcp_verify.py`
Result: `110 passed`
Run command: `.venv/bin/pytest -q tests/test_query_tools.py tests/test_query_engine.py -m 'not integration' tests/test_query_control_plane.py`
Result: `74 passed, 2 deselected`

- [x] `G1` Commit atomic changes and push
Status: `done`
Commit: `fabcba1`

## Notes
- Existing unrelated modified/untracked files are present in the worktree and are intentionally left untouched.
- No placeholder features: each task ships with tests or explicit failure notes.
