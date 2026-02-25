# MOSAICX Roadmap Status Audit

Date: 2026-02-24
Branch: `feat/dspy-rlm-evolution`
Status: Active
Owner: Core platform
Authoritative: Yes (single source of truth for rollout status)

## 0) Canonical Status (Updated 2026-02-25 10:12)

This file is the canonical status board for DSPy roadmap execution.
Other plan files are design/history logs and must link here for status.

### Issue state

- Closed roadmap delivery items:
  - `QRY-001`, `QRY-002`, `VER-001`, `VER-002`, `EXT-001`, `EXT-002`, `OPS-001`, `EVAL-001`, `EVAL-002`, `DOC-001`
  - roadmap epics: `#33 [ROADMAP-OPS]`, `#34 [ROADMAP-QRY]`, `#35 [ROADMAP-EXT]`
  - legacy bugs: `#17`, `#18`, `#19`
  - current bug fixes: `#51`, `#52`, `#53`, `#54`, `#55`
  - closed duplicates: `#22`, `#27`
- Open stabilization items:
  - `#56 [SCHEMA-001] Robust schema generation with normalize/validate/repair pipeline`
- Closed DSPy capability items:
  - `#36 [DSPY-01] ReAct planner as primary query control-plane`
  - `#37 [DSPY-02] RLM executor robustness for long-document query + verify`
  - `#38 [DSPY-03] JSONAdapter default structured-output policy`
  - `#39 [DSPY-04] TwoStepAdapter fallback policy`
  - `#40 [DSPY-05] BestOfN grounded candidate selection`
  - `#42 [DSPY-08] CodeAct controlled fallback for complex tabular analytics`
  - `#43 [DSPY-06] Refine for extraction-critical steps`
  - `#41 [DSPY-07] MultiChainComparison contradiction adjudication`
  - `#45 [DSPY-09] ProgramOfThought fallback-only hardening`
  - `#46 [DSPY-10] CompleteAndGrounded as enforced quality gate`
  - `#47 [DSPY-13] SIMBA optimization run with persisted artifacts`
  - `#48 [DSPY-12] MIPROv2 optimization baseline with persisted artifacts`
  - `#49 [DSPY-14] GEPA optimization run with persisted artifacts`
  - `#50 [DSPY-15] Parallel/Async/Streaming execution with strict failure isolation`
  - `#44 [DSPY-11] answer_exact_match + answer_passage_match CI gates`
- Open roadmap epics:
  - none
- Open DSPy capability rollout items:
  - none

### Canonical control note

- Use this file as the only status authority for roadmap completion.
- Treat all other plan docs in `docs/plans/` as design/history/reference only.
- If status here conflicts with another file, this file wins.

### Evidence highlights

- Schema robustness hardening (`2026-02-25`, `SCHEMA-001`) implemented and validated:
  - `mosaicx/pipelines/schema_gen.py` now includes deterministic `normalize_schema_spec` + `validate_schema_spec`.
  - `SchemaGenerator` now uses `BestOfN` candidate selection when available, plus a repair loop (`RepairSchemaSpec`) with validation/compile feedback.
  - Added Outlines recovery when DSPy structured parsing fails during schema generation.
  - `mosaicx/cli.py` `template create` paths (`--from-json`, `--from-document`, `--describe`, RadReport) now normalize+validate schema and compile generated YAML before save.
  - Regression coverage added:
    - `tests/test_schema_gen.py`
    - `tests/test_cli_integration.py` (`template_create_from_json` block)
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_schema_gen.py` -> `22 passed`
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_cli_integration.py -k "template_create_from_json"` -> `7 passed`
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_template_compiler.py tests/test_radiology_refine.py tests/test_pathology_refine.py` -> `12 passed`
  - Live local `vllm-mlx` checks:
    - `template create --describe ...` -> valid `/tmp/robust_cspine_v4.yaml`
    - `template create --from-document ...` -> valid `/tmp/robust_cspine_doc_v4.yaml`
    - extract with document-derived template succeeded and produced structured JSON.
  - Extraction-safety hardening follow-up (`2026-02-25 07:34`):
    - `mosaicx/pipelines/extraction.py` now normalizes null-like strings for optional/required collections (for example, `"None"` -> `null` or `[]` depending on schema annotation).
    - Added Outlines constrained recovery when both typed DSPy extraction and JSON fallback fail in schema mode.
    - Regression coverage:
      - `tests/test_extraction_schema_coercion.py`
      - `tests/test_extraction_pipeline.py`
      - `tests/test_cli_extract.py`
    - Validation:
      - `PYTHONPATH=. .venv/bin/pytest -q tests/test_extraction_schema_coercion.py tests/test_extraction_pipeline.py tests/test_cli_extract.py` -> `22 passed`
    - Live local `vllm-mlx` repro fix:
      - previously failing `mosaicx extract --template /tmp/robust_cspine_v4.yaml ...` now succeeds and saves `/tmp/robust_cspine_v4_extract_after_fix.json`.
  - Extraction-aware schema runtime gate (`2026-02-25 07:49`):
    - `mosaicx/pipelines/schema_gen.py` now supports runtime dry-run validation (`_runtime_validate_schema`) integrated into the generator repair loop.
    - `mosaicx/cli.py` enables runtime dry-run for `template create --from-document`.
    - Runtime gate enforces extraction-readiness before schema acceptance (in document-grounded generation path).
    - Regression tests added/updated:
      - `tests/test_schema_gen.py`
    - Validation:
      - `PYTHONPATH=. .venv/bin/pytest -q tests/test_schema_gen.py` -> `24 passed`
      - `PYTHONPATH=. .venv/bin/pytest -q tests/test_extraction_schema_coercion.py tests/test_extraction_pipeline.py tests/test_cli_extract.py` -> `22 passed`
    - Live local `vllm-mlx` verification:
      - `template create --from-document ... --name RobustCSpineDocV5` generated a richer 17-field schema.
      - `extract --template /tmp/robust_cspine_doc_v5.yaml` succeeded with grounded structured output.
  - Describe-only runtime gate completion (`2026-02-25 10:12`):
    - `mosaicx/pipelines/schema_gen.py` now synthesizes deterministic runtime probe text when no source document is provided and `runtime_dryrun=True`.
    - Runtime validation is now enabled for describe-only schema generation (no document required), and still prefers real document text when available.
    - `mosaicx/cli.py` now enables runtime dry-run for all LLM `template create` paths (RadReport and generic `--describe/--from-document/--from-url` flow).
    - Regression coverage added:
      - `tests/test_schema_gen.py`:
        - synthetic probe payload construction
        - describe-only runtime dry-run path
        - runtime preference for real source text
    - Validation:
      - `scripts/clear_dspy_cache.sh && PYTHONPATH=. .venv/bin/pytest -q tests/test_schema_gen.py` -> `27 passed`
      - `PYTHONPATH=. .venv/bin/pytest -q tests/test_cli_integration.py -k "template_create"` -> `9 passed`
    - Live local `vllm-mlx` check:
      - `template create --describe ... --name RuntimeProbeTmp --output /tmp/runtime_probe_tmp.yaml` succeeded (`6,145 tokens`, `36.9s`) and produced a valid template.

- Verify audit recovery gating hardened (`2026-02-24`, `BUG-VER-001`):
  - `mosaicx/verify/audit.py` now attempts Outlines recovery only for structured serialization/JSON parse failures.
  - LM-unconfigured failures (`No LM is loaded ... dspy.configure(lm=...)`) now bubble to engine fallback instead of being force-recovered, restoring expected `thorough -> standard/deterministic` semantics.
  - Added regression tests:
    - `tests/test_verify_audit.py::test_claim_audit_skips_outlines_when_lm_unconfigured`
    - `tests/test_verify_audit.py::test_extraction_audit_skips_outlines_when_lm_unconfigured`
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_verify_audit.py tests/test_verify_engine.py tests/test_sdk_verify.py` -> `66 passed, 1 skipped`

- Hard-test pipeline reliability (`ROADMAP-OPS`):
  - `scripts/run_hard_test_matrix.sh` now enforces DSPy cache clears between major packs and before LLM checks via `scripts/clear_dspy_cache.sh`.
  - Added strict LLM preflight mode (`MOSAICX_HARDTEST_STRICT_LLM=1` default): hard-test run aborts if local model endpoint health-check fails.
  - Added local helpers required by matrix runner:
    - `scripts/ensure_vllm_mlx_server.sh`
    - `scripts/generate_hard_test_fixtures.py`
    - `scripts/clear_dspy_cache.sh`

- Primary-path guard for verify route semantics (`BUG-QA-001`):
  - Added SDK regression asserting primary `thorough -> audit` route remains explicit and non-fallback:
    - `tests/test_sdk_verify.py::test_thorough_claim_reports_primary_audit_route_without_fallback`
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_sdk_verify.py tests/test_verify_engine.py tests/test_verify_audit.py` -> `67 passed, 1 skipped`

- Verify contract simplification + structured recovery hardening (`2026-02-26`, commit `81fad8f`):
  - SDK `verify()` now returns compact top-level fields by default:
    - claim mode: `result`, `claim_is_true`, `confidence`, `claim`, `source_value`, `evidence`
    - extraction mode: `result`, `confidence`, `field_checks`
  - Verbose diagnostics moved to `debug` (opt-in in SDK via `include_debug=True`, enabled by CLI for rich rendering).
  - Added Outlines structured recovery in `mosaicx/verify/audit.py` for RLM/JSON parse failures before deterministic fallback.
  - Added regression tests for Outlines recovery and compact-output CLI/SDK behavior.
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_verify_audit.py tests/test_verify_engine.py` -> `44 passed`
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_sdk_verify.py` -> `20 passed, 1 skipped`
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_cli_verify.py` -> `16 passed`
  - Live local `vllm-mlx` checks (thorough):
    - true claim (`BP 128/82`) -> `result=verified`, `claim_is_true=true`, `debug.executed_mode=audit`, `debug.fallback_used=false`
    - false claim (`BP 120/82`) -> `result=contradicted`, `claim_is_true=false`, `debug.executed_mode=audit`, `debug.support_score=0.00`

- Query planner-first hardening (`QRY-002`) implemented and tested:
  - Non-integration: `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py tests/test_query_control_plane.py -m 'not integration'` -> `73 passed, 4 deselected`
  - Integration with local `vllm-mlx`: `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py -m integration` -> `5 passed, 68 deselected`
  - Multi-turn live CSV verification confirmed planner path for implicit category count asks.
- Query follow-up drift fix (`BUG-QRY-001`, `#51`) implemented:
  - `_resolve_followup_question` now preserves `last_tabular_column` for `count_values`/`count_distinct`/`list_values` turns, so follow-ups like `what are they?` map to distinct category values instead of schema/irrelevant columns.
  - Added regression:
    - `tests/test_query_engine.py::test_tabular_gender_count_followup_what_are_they_keeps_column_context`
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py -k "tabular_gender_count_followup_what_are_they_keeps_column_context or tabular_coreference_followup_preserves_entity_context or tabular_typo_prompt_returns_grounded_count_values"` -> `3 passed, 72 deselected`
- Planner-path determinism + fallback failure isolation hardening (`ROADMAP-QRY`, `DSPY-01`, `DSPY-15`):
  - Planned tabular execution now normalizes under-specified planner intents (`count`, `count_values`, `aggregate` without operation for count/value asks) into executable deterministic paths, reducing planner-used/not-executed drift.
  - Added parallel retrieval collection with strict failure isolation for LLM-fallback evidence gathering:
    - `QueryEngine._collect_retrieval_hits_parallel` executes document/table/chunk/computed/column retrieval concurrently and degrades safely per-tool.
    - Fallback payload now exposes `parallel_used` and `parallel_failures`.
  - Added regressions:
    - `tests/test_query_engine.py::test_planner_count_intent_normalizes_to_count_distinct_execution`
    - `tests/test_query_engine.py::test_planner_aggregate_without_operation_normalizes_to_count_distinct`
    - `tests/test_query_engine.py::test_collect_retrieval_hits_parallel_isolates_tool_failure`
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py tests/test_cli_display.py -m 'not integration'` -> `77 passed, 6 deselected`
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py -m integration` -> `6 passed, 72 deselected`
- Extract/template display stabilization (`ROADMAP-EXT`):
  - Preserved vertebral level hyphens in user-facing output (`C3-C4`, `C7-T1`) instead of degrading to `C3 4`.
  - Level-finding tables now retain sparse-but-important columns (for example, `Disc Bulge Type`) instead of hiding them via generic null-threshold filtering.
  - Added regressions:
    - `tests/test_cli_display.py::test_format_value_preserves_vertebral_level_hyphen`
    - `tests/test_cli_display.py::test_render_list_table_keeps_sparse_level_finding_columns`
  - Live check:
    - `mosaicx extract --template MRICervicalSpineV3` now renders `Level Findings` with `Disc Bulge Type` and correct level text (`C3-C4`).
- Optimizer artifacts (`EVAL-002`) produced with real local model execution:
  - `docs/runs/2026-02-24-query-optimizer-seq-tiny/optimizer_sequence_manifest.json`
  - `docs/runs/2026-02-24-query-optimizer-seq-tiny/miprov2_optimized.json`
  - `docs/runs/2026-02-24-query-optimizer-seq-tiny/simba_optimized.json`
  - `docs/runs/2026-02-24-query-optimizer-seq-tiny/gepa_optimized.json`
  - Baseline metrics persisted: `docs/runs/2026-02-24-query-optimizer-seq-tiny/baseline_metrics.json`
  - Run report with command provenance: `docs/runs/2026-02-24-eval-002-optimizer-sequence-report.md`
  - Verify pipeline artifact run:
    - `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/optimizer_sequence_manifest.json`
    - `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/baseline_metrics.json`
    - `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/miprov2_optimized.json`
    - `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/simba_optimized.json`
    - `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/gepa_optimized.json`
    - Run report: `docs/runs/2026-02-24-eval-002-verify-optimizer-sequence-report.md`
- DOC canonicalization (`DOC-001`) applied:
  - `docs/plans/2026-02-23-dspy-full-capability-rollout.md` marked as historical/superseded.
  - `docs/plans/2026-02-23-sota-execution-memory.md` marked as execution ledger with canonical pointer.
- Extraction robustness for legacy MIMIC failures (`#17/#18/#19`) implemented:
  - Added schema coercion + JSON-like recovery path in schema-mode extraction:
    - `mosaicx/pipelines/extraction.py`
  - Added regression tests for three failure classes:
    - scalar -> nested `LabelAssessment`
    - nested `{value,...}` -> literal trinary field
    - prose-wrapped JSON recovery
    - `tests/test_extraction_schema_coercion.py`
  - Test evidence:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_extraction_schema_coercion.py` -> `4 passed`
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_extraction_pipeline.py tests/test_cli_display.py` -> `8 passed`
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_cli_extract.py tests/test_report.py` -> `60 passed`
  - Live local model smoke on `s50184397.txt` with custom MIMIC-like schema succeeded via local `vllm-mlx`:
    - output parsed into typed schema object (`ok=True`) and no extraction crash.
- Fresh full hard-test matrix run (LLM+integration enabled) completed successfully:
  - `MOSAICX_HARDTEST_LLM=1 MOSAICX_HARDTEST_INTEGRATION=1 MOSAICX_HARDTEST_QUERY_ENGINE_INT=1 scripts/run_hard_test_matrix.sh`
  - run artifact: `docs/runs/2026-02-24-171602-hard-test-matrix.md`
  - summary: deterministic/unit packs passed, integration packs passed, local vLLM-MLX preflight passed, verify-thorough true/false checks passed, query temporal/tabular checks passed.
- Query citation signal cleanup for tabular analytics:
  - `mosaicx/query/engine.py` now suppresses `__eda_profile__.json` text evidence for strict tabular numeric/count+values prompts.
  - New regression: `tests/test_query_engine.py::test_build_citations_count_values_excludes_profile_text_chunk_noise`.
  - Validation: `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py -m 'not integration'` -> `69 passed, 5 deselected`.
  - Live check: count+values CLI output now surfaces computed rows + planner column match without profile noise.
- Claim adjudication hardening (`DSPY-05`, `DSPY-07`) implemented in SDK verify finalization:
  - Added DSPy ambiguity adjudicator with `MultiChainComparison` + `BestOfN` for grounded claim cases where values are neither clearly matched nor clearly conflicting.
  - Applied only in bounded ambiguous paths; deterministic conflict/match guards remain authoritative.
  - Files: `mosaicx/sdk.py`, `tests/test_sdk_verify.py`.
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_sdk_verify.py -k "dspy_adjudicator_prefers_multichain_output_when_available or ambiguous_grounded_claim_uses_dspy_adjudication or clear_claim_conflict_skips_dspy_adjudication or claim_value_conflict_overrides_verified_decision or thorough_claim_matching_source_rescues_partial_verdict"` -> `5 passed`.
- Adapter policy hardening (`DSPY-03`, `DSPY-04`) now unified across CLI/SDK/MCP initialization:
  - CLI and MCP initialization now route through `runtime_env.configure_dspy_lm` (JSONAdapter-first with TwoStep fallback).
  - Files: `mosaicx/cli.py`, `mosaicx/mcp_server.py`, `mosaicx/runtime_env.py`.
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_cli_dspy_config.py tests/test_mcp_server_dspy_config.py tests/test_runtime_env.py` -> `17 passed`.
- Query quality-gate expansion (`DSPY-11`) wired:
  - Added `answer_exact_match_metric` and `answer_passage_match_metric` wrappers with DSPy-evaluate path + deterministic fallback.
  - Query components now emit `exact_match` and `passage_match`; quality gates enforce `exact_match_mean` and `passage_match_mean` when present.
  - Files: `mosaicx/evaluation/grounding.py`, `mosaicx/evaluation/quality_gates.py`, `mosaicx/evaluation/__init__.py`, `mosaicx/cli.py`.
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_evaluation_grounding.py tests/test_evaluation_quality_gates.py` -> `16 passed`.

### Operational note

- LLM-backed integration/eval commands must run outside sandbox in this environment to access local `127.0.0.1:8000` reliably.
- Preflight requirement remains: validate both `/v1/models` and `/v1/chat/completions`.

### Historical sections below

The remaining sections in this document are retained as historical audit context from earlier checkpoints.

## 1) Snapshot

This audit reconciles roadmap claims against actual code paths, tests, and live CLI behavior.

Key finding:

- `docs/plans/2026-02-23-dspy-full-capability-rollout.md` claims `[Overall Progress: 100%]`.
- `docs/plans/2026-02-23-sota-execution-memory.md` still has nearly all `DSPY-*` and epic checkboxes unchecked.
- Real code/test/runtime evidence supports **partial** completion, not 100%.

## 2) Evidence Collected (this audit block)

Commands and outcomes:

1. `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py tests/test_query_control_plane.py tests/test_query_tools.py -m 'not integration'`
- Result: `92 passed, 2 deselected`

2. `PYTHONPATH=. .venv/bin/pytest -q tests/integration/test_full_pipeline.py tests/integration/test_e2e.py`
- Result: `18 passed`

3. `PYTHONPATH=. .venv/bin/pytest -q tests/test_design_doc_compliance.py tests/test_mcp_verify.py tests/test_sdk_verify.py`
- Result: `45 passed`

4. `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_loaders.py tests/test_verify_deterministic.py tests/test_verify_engine.py tests/test_cli_extract.py`
- Result: `80 passed, 1 failed`
- Failure: `tests/test_cli_extract.py::TestExtractAPIKeyCheck::test_extract_fails_without_api_key`

5. `PYTHONPATH=. .venv/bin/pytest -q tests/test_schema_gen.py tests/test_report.py`
- Result: `68 passed`

6. Local model probes:
- `curl -sS --max-time 5 http://127.0.0.1:8000/v1/models` -> model listed (`mlx-community/gpt-oss-120b-4bit`)
- `curl -sS --max-time 120 http://127.0.0.1:8000/v1/chat/completions ...` -> successful response (token usage returned)

7. Live query check:
- `mosaicx query --document tests/datasets/generated_hardcases/cohort_edge_cases.csv -q "what is the distribution of male and female?" --eda --trace --max-iterations 2`
- Correct output: `Sex distribution (3 groups): F=2, M=2, Other=1.`

8. Live query check (harder phrasing):
- `mosaicx query --document tests/datasets/generated_hardcases/cohort_edge_cases.csv -q "how many ethnicities are there and what are they?" --eda --trace --max-iterations 2`
- Output is technically grounded but poorly narrated:
  - returns generic "Grounded summary..." instead of direct answer sentence.

9. Live verify thorough (escalated local LLM path):
- `mosaicx verify --sources tests/datasets/extract/sample_patient_vitals.pdf --claim "patient BP is 128/82" --level thorough`
- Output: `Decision partially_supported`, `Truth inconclusive`, despite `Claimed: 128/82` and `Source: 128/82`.
- Evidence text contradictory: "source document does not contain any blood pressure reading."

## 3) DSPy Capability Ledger (Reality)

Legend:
- `done`: clearly implemented and validated
- `partial`: implemented in some paths but not primary/consistent/reliable
- `missing`: not implemented or not wired into gates/artifacts

1. `DSPY-01 ReAct planner primary`: **partial**
- Exists in `mosaicx/query/control_plane.py`.
- Not primary globally because deterministic lexical routing/fallbacks still dominate many query paths.

2. `DSPY-02 RLM executor (query + verify thorough)`: **partial**
- Exists in `mosaicx/query/engine.py` and `mosaicx/verify/audit.py`.
- Live thorough verify still produces inconsistent adjudication in a simple truth case.

3. `DSPY-03 JSONAdapter default policy`: **partial**
- Adapter policy exists in `mosaicx/runtime_env.py`.
- Not uniformly robust across all CLI paths in practice.

4. `DSPY-04 TwoStepAdapter fallback`: **partial**
- Implemented parse-fallback in query/verify.
- Parse and runtime failures still surface in live runs.

5. `DSPY-05 BestOfN grounded selection`: **done**
- Implemented in query evidence verifier and claim-level SDK adjudicator.
- Validated via targeted SDK regression tests plus existing query control-plane usage.

6. `DSPY-06 Refine for extraction-critical steps`: **done (module-level)**
- Present in radiology/pathology pipelines.
- End-to-end extraction reliability still affected by environment/model plumbing.

7. `DSPY-07 MultiChainComparison contradiction adjudication`: **done**
- Implemented in query verifier flow and SDK claim adjudication flow for contradiction-sensitive finalization.
- Bounded to ambiguous grounded claim cases with deterministic guards preserved.

8. `DSPY-08 CodeAct fallback`: **partial**
- Implemented in tabular programmatic analyst.
- Needs stronger runtime boundaries and success criteria in live workloads.

9. `DSPY-09 ProgramOfThought fallback-only with strict guards`: **partial**
- Gated fallback exists.
- Syntax/runtime instability still appears in user-facing runs.

10. `DSPY-10 CompleteAndGrounded in quality gates`: **partial**
- Metric wrapper present.
- Not a strict release blocker across all key flows.

11. `DSPY-11 answer_exact_match + answer_passage_match CI`: **missing**
- No code usage found.

12. `DSPY-12 MIPROv2 artifacts`: **missing**
- Optimizer scaffolding exists, no persisted run artifacts found.

13. `DSPY-13 SIMBA artifacts`: **missing**
- Same as above.

14. `DSPY-14 GEPA artifacts`: **missing**
- Same as above.

15. `DSPY-15 Parallel/Async/Streaming with failure isolation`: **missing/limited**
- No clear end-to-end staged parallel architecture with isolation in query/verify runtime.

## 4) Bug Report Reconciliation (`docs/bugs/2026-02-23-systematic-test-results.md`)

### Likely fixed

- Loader robustness set:
  - empty CSV handling
  - latin1/cp1252 fallback decode
  - malformed JSON clear error
  - multiline JSONL parsing
  - semicolon CSV detection
  - TSV dataframe loading
  - multi-sheet Excel merge behavior
  - large file guard

Evidence:
- `tests/test_query_loaders.py` passes.

- Verify deterministic crash/pass-through issues:
  - malformed findings now tolerated
  - no-check cases now return `insufficient_evidence` rather than false `verified`

Evidence:
- `tests/test_verify_deterministic.py`, `tests/test_verify_engine.py` pass.

- Nested list-object template bug:

Evidence:
- `tests/test_schema_gen.py`, `tests/test_report.py` pass.

### Still open / re-opened

1. Verify thorough adjudication inconsistency on simple true claim:
- Reproduced live.

2. Query narrative quality on tabular count+values:
- Reproduced live (`ethnicity` ask returns "Grounded summary..." instead of direct answer format).

3. Extract model-provider/runtime plumbing:
- Live `extract` path fails in current environment with model/provider mismatch or connection errors.

4. API-key check test instability:
- `tests/test_cli_extract.py::TestExtractAPIKeyCheck` currently failing.

## 5) Worktree Execution Plan (Parallel)

Use isolated worktrees to avoid cross-feature churn:

1. `wt-query-controlplane`
- Focus: Route/Plan/Execute/Verify/Narrate cleanup and tabular narration contract.
- Files: `mosaicx/query/engine.py`, `mosaicx/query/control_plane.py`, `mosaicx/query/tools.py`, query tests.

2. `wt-verify-adjudication`
- Focus: claim truth contract and thorough audit consistency.
- Files: `mosaicx/verify/engine.py`, `mosaicx/verify/audit.py`, `mosaicx/sdk.py`, `mosaicx/cli.py`, verify tests.

3. `wt-extract-runtime`
- Focus: extract LM/provider resolution, fallback behavior, enum rendering contract.
- Files: `mosaicx/cli.py`, `mosaicx/pipelines/extraction.py`, config/runtime paths, extract tests.

4. `wt-eval-optimize`
- Focus: wire `answer_exact_match` + `answer_passage_match`, enforce CI gates, produce optimizer artifacts.
- Files: `mosaicx/evaluation/*.py`, CI config, docs artifacts.

5. `wt-docs-ops`
- Focus: single source of truth docs + runbooks + bug status sync.
- Files: `README.md`, `docs/*`, plan trackers.

## 6) Issue Backlog (ready to create)

`gh` status currently blocked:
- token invalid (`gh auth status` fails).

Create these issues once auth is fixed:

1. `QRY-001` Query direct-answer contract for count+values questions.
2. `QRY-002` Remove/contain lexical routing debt; ReAct primary for ambiguous tabular asks.
3. `VER-001` Verify thorough true-claim inconsistency (`128/82` case).
4. `VER-002` Unified claim truth schema + contradiction-first rendering.
5. `EXT-001` Extract LM provider/base resolution hardening for local vLLM-MLX.
6. `EXT-002` Enum/internal-label rendering contract for template outputs.
7. `EVAL-001` Wire `answer_exact_match` and `answer_passage_match` into gates.
8. `EVAL-002` Run and store optimizer artifacts (`MIPROv2`, `SIMBA`, `GEPA`).
9. `OPS-001` LLM preflight gate before integration tests (hard fail on missing `/chat/completions`).
10. `DOC-001` Resolve roadmap doc contradictions and keep single authoritative progress board.

## 7) Immediate Next Step (highest leverage)

Execute in order:

1. Fix `VER-001` (truth inconsistency in thorough verify).
2. Fix `QRY-001` (tabular direct-answer contract).
3. Fix `EXT-001` (extract runtime provider/base robustness).
4. Add `EVAL-001` gates and baseline artifact generation.

Only then mark roadmap progress upward from current partial state.
