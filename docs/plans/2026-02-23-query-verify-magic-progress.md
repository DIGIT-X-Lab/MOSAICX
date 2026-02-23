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

## Phase 2 (Completed)
- [x] `Q2` Replace brittle follow-up suffixing with structured follow-up rewrite from session state (`query_state`, prior turn context, DSPy rewrite + deterministic fallback)
Status: `done (local)`

- [x] `Q3` Add planner-first tabular execution path (execute ReAct plan directly for `count_rows`, `count_distinct`, `list_values`, `aggregate`, `mixed`)
Status: `done (local)`

- [x] `T2` Add regression tests for follow-up rewrite + planner-first execution
Status: `done (local)`

- [x] `R2` Run focused query tests and fix regressions
Status: `done`
Run command: `.venv/bin/pytest -q tests/test_query_engine.py tests/test_query_control_plane.py -m 'not integration'`
Result: `44 passed, 2 deselected`
Run command: `.venv/bin/pytest -q tests/test_query_tools.py -m 'not integration'`
Result: `31 passed`

- [x] `G2` Commit + push phase-2 improvements
Status: `done`
Commit: `4512c35`

## Phase 3 (Completed)
- [x] `Q4` Focus-aware tabular citation ranking/selection (source+column bias to avoid mismatched evidence in count/value questions)
Status: `done (local)`

- [x] `Q5` Protect deterministic tabular answers from unnecessary reconciler rewrites
Status: `done (local)`

- [x] `T3` Add regressions for deterministic overwrite prevention + focused value evidence selection
Status: `done (local)`

- [x] `R3` Run focused query suites after phase-3 changes
Status: `done`
Run command: `.venv/bin/pytest -q tests/test_query_engine.py -m 'not integration'`
Result: `44 passed, 2 deselected`
Run command: `.venv/bin/pytest -q tests/test_query_control_plane.py tests/test_query_tools.py -m 'not integration'`
Result: `33 passed`

- [x] `G3` Commit + push phase-3 improvements
Status: `done`
Commit: `33762c0`

## Phase 4 (In Progress)
- [x] `Q6` Fallback evidence recovery for LLM errors (prefer grounded answer synthesis over generic "LLM unavailable")
Status: `done (local)`

- [x] `Q7` Long-document chunk grounding guard in query finalization path
Status: `done (local)`

- [x] `T4` Add regressions for fallback evidence recovery + long-doc chunk grounding upgrades
Status: `done (local)`

- [ ] `R4` Run focused query tests and fix regressions
Status: `done`
Run command: `.venv/bin/pytest -q tests/test_query_engine.py -m 'not integration'`
Result: `46 passed, 2 deselected`
Run command: `.venv/bin/pytest -q tests/test_query_control_plane.py tests/test_query_tools.py -m 'not integration'`
Result: `33 passed`

- [x] `G4` Commit + push phase-4 improvements
Status: `done`
Commit: `4d37fee`

## Phase 5 (In Progress)
- [x] `V5` Add explicit developer truth aliases in SDK verify payload (`claim_true`, `is_verified`, `is_contradicted`)
Status: `done (local)`

- [x] `V6` Clarify CLI claim adjudication label from "Claim truth" to "Truth" and keep compatibility fallback
Status: `done (local)`

- [x] `T5` Add SDK regressions for new truth alias fields
Status: `done (local)`

- [ ] `R5` Run focused verify/CLI tests and fix regressions
Status: `done`
Run command: `.venv/bin/pytest -q tests/test_sdk_verify.py tests/test_cli_verify.py tests/test_verify_engine.py -m 'not integration'`
Result: `62 passed`

- [x] `G5` Commit + push phase-5 improvements
Status: `done`
Commit: `3d2a029`

## Phase 6 (In Progress)
- [x] `Q8` Preserve structured query memory across sparse turns (do not wipe active source/column when current citations omit column metadata)
Status: `done (local)`

- [x] `Q9` Extend query trace with chunk grounding visibility (`chunk_citations`)
Status: `done (local)`

- [x] `T6` Add regression for state preservation under sparse citations
Status: `done (local)`

- [ ] `R6` Run focused query tests and fix regressions
Status: `done`
Run command: `.venv/bin/pytest -q tests/test_query_engine.py tests/test_query_control_plane.py tests/test_query_tools.py -m 'not integration'`
Result: `80 passed, 2 deselected`

- [x] `G6` Commit + push phase-6 improvements
Status: `done`
Commit: `2dbd8ee`
