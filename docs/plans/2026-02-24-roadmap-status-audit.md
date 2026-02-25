# MOSAICX Roadmap Status Audit

Date: 2026-02-24
Branch: `feat/dspy-rlm-evolution`
Status: Active
Owner: Core platform
Authoritative: Yes (single source of truth for rollout status)

## 0) Canonical Status (Updated 2026-02-25 20:35)

This file is the canonical status board for DSPy roadmap execution.
Other plan files are design/history logs and must link here for status.

### Issue state

- Closed roadmap delivery items:
  - `QRY-001`, `QRY-002`, `VER-001`, `VER-002`, `EXT-001`, `EXT-002`, `OPS-001`, `EVAL-001`, `EVAL-002`, `DOC-001`
  - roadmap epics: `#33 [ROADMAP-OPS]`, `#34 [ROADMAP-QRY]`, `#35 [ROADMAP-EXT]`
  - legacy bugs: `#17`, `#18`, `#19`
  - current bug fixes: `#51`, `#52`, `#53`, `#54`, `#55`
  - closed duplicates: `#22`, `#27`
- Closed stabilization items:
  - `#56 [SCHEMA-001] Robust schema generation with normalize/validate/repair pipeline`
  - `#57 [SCHEMA-002] Add semantic granularity scoring for generated templates`
  - `#58 [EXT-003] Investigate V2 vs V3 cervical template extraction parity and enum rendering drift`
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
- Open extract atomic rollout items:
  - `#62 [EXTX-04]`, `#63 [EXTX-05]`, `#64 [EXTX-06]`, `#65 [EXTX-07]`, `#66 [EXTX-08]`, `#67 [EXTX-09]`, `#68 [EXTX-10]`
- Closed extract atomic rollout items:
  - `#59 [EXTX-01]` Canonical extract contract + fail-closed semantics across CLI/SDK/MCP
  - `#60 [EXTX-02]` ReAct planner-first section routing for extraction
  - `#61 [EXTX-03]` Outlines-first structured extraction with DSPy adapter fallback chain

### Canonical control note

- Use this file as the only status authority for roadmap completion.
- Use `docs/plans/2026-02-25-mosaicx-clear-roadmap.md` as the active execution plan.
- Treat all other plan docs in `docs/plans/` as design/history/reference only.
- If status here conflicts with another file, this file wins.

### Evidence highlights

- Extract contract propagation baseline completed (`2026-02-25`, `EXTX-01`, issue `#59`):
  - Added canonical extraction contract helper in `mosaicx/pipelines/extraction.py`:
    - `apply_extraction_contract(...)` with per-field `value`, `evidence`, `grounded`, `confidence`, `status`.
    - explicit fail-closed state for missing fields: `insufficient_evidence`.
  - Wired contract propagation across all extraction interfaces:
    - CLI: `mosaicx/cli.py`
    - SDK: `mosaicx/sdk.py`
    - MCP: `mosaicx/mcp_server.py` (`extract_document`)
  - Added regression coverage:
    - `tests/test_extract_contract.py`
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_extract_contract.py tests/test_extraction_pipeline.py tests/test_sdk_envelope.py tests/test_mcp_server_dspy_config.py` -> `24 passed`
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_mcp_query.py tests/test_mcp_verify.py tests/test_public_api.py -k "extract or verify_output or query_start or query_ask or query_close"` -> `27 passed, 37 deselected`
  - Run artifact:
    - `docs/runs/2026-02-25-extx-01-contract-propagation.md`

- ReAct planner-first extraction routing completed (`2026-02-25`, `EXTX-02`, issue `#60`):
  - Added planner flow in `mosaicx/pipelines/extraction.py`:
    - section splitting and complexity hints,
    - DSPy `ReAct` route planning with deterministic fallback,
    - routed extraction context composition.
  - `DocumentExtractor.forward()` now runs explicit `Plan extraction` then `Extract` in both schema and auto modes.
  - Planner diagnostics are propagated to user-facing outputs:
    - `mosaicx/cli.py`, `mosaicx/sdk.py`, `mosaicx/mcp_server.py` now emit `_planner`.
  - Regression coverage:
    - `tests/test_extraction_pipeline.py`
    - `tests/test_extract_contract.py`
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_extraction_pipeline.py tests/test_extract_contract.py tests/test_cli_extract.py tests/test_mcp_server_dspy_config.py tests/test_mcp_query.py tests/test_mcp_verify.py tests/test_public_api.py -k "extract or planner or verify_output or query_start or query_ask or query_close"` -> `51 passed, 39 deselected`
    - Live local `vllm-mlx` smoke:
      - `PYTHONPATH=. .venv/bin/mosaicx extract --document tests/datasets/standardize/Sample_Report_Cervical_Spine.pdf --template MRICervicalSpineV3 -o /tmp/mosaicx_extx02_smoke.json`
      - output `_planner` showed `planner=react`, `react_used=true`.
  - Run artifact:
    - `docs/runs/2026-02-25-extx-02-react-routing.md`

- Outlines-first structured extraction chain completed (`2026-02-25`, `EXTX-03`, issue `#61`):
  - Added deterministic chain in `mosaicx/pipelines/extraction.py`:
    1. `outlines_primary`
    2. `dspy_typed_direct`
    3. `dspy_json_adapter`
    4. `dspy_two_step_adapter`
    5. `existing_json_fallback`
    6. `existing_outlines_rescue`
  - Chain diagnostics are now persisted in `_planner`:
    - `structured_chain`, `selected_structured_path`, `structured_fallback_used`.
  - Added regression tests for malformed/non-JSON and ordered fallback behavior:
    - `tests/test_extraction_pipeline.py::TestStructuredExtractionChain`
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_extraction_pipeline.py tests/test_extract_contract.py tests/test_cli_extract.py tests/test_mcp_server_dspy_config.py tests/test_mcp_query.py tests/test_mcp_verify.py tests/test_public_api.py -k "extract or planner or verify_output or query_start or query_ask or query_close"` -> `54 passed, 39 deselected`
    - Live local `vllm-mlx` smoke:
      - `PYTHONPATH=. .venv/bin/mosaicx extract --document tests/datasets/standardize/Sample_Report_Cervical_Spine.pdf --template MRICervicalSpineV3 -o /tmp/mosaicx_extx03_smoke.json`
      - output `_planner.selected_structured_path=outlines_primary`, `structured_fallback_used=false`.
  - Run artifact:
    - `docs/runs/2026-02-25-extx-03-outlines-first-chain.md`

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
  - Semantic quality follow-up identified (`2026-02-25 10:18`):
    - `template create --from-document tests/datasets/standardize/Sample_Report_Cervical_Spine.pdf --name RuntimeProbeDoc` produced a valid and extractable template, but field granularity remained flatter than expert design (for example, broad `findings: list[str]` vs typed level-structured objects).
    - Opened follow-up issue `#57 [SCHEMA-002]` to enforce semantic granularity scoring and benchmark-driven improvement.
  - Hybrid granularity gate improvement (`2026-02-25 10:44`, `SCHEMA-002`):
    - Added deterministic semantic-granularity scoring to `mosaicx/pipelines/schema_gen.py` (typed coverage, nested structure, enum cues, repeated-structure cues).
    - Added DSPy semantic assessor (`AssessSchemaGranularity`) as an LLM critique path when deterministic score/issues indicate weak schema granularity.
    - Repair loop now receives semantic issues (`semantic_issue:*`) plus low-score signal (`semantic_score_low:*`) for document-grounded generation.
    - Regression coverage expanded in `tests/test_schema_gen.py`:
      - spine-level repeated-structure scenarios
      - non-spine numbered-lesion scenarios
      - flat vs structured schema scoring checks
    - Validation:
      - `scripts/clear_dspy_cache.sh && PYTHONPATH=. .venv/bin/pytest -q tests/test_schema_gen.py` -> `31 passed`
      - `PYTHONPATH=. .venv/bin/pytest -q tests/test_cli_integration.py -k "template_create"` -> `9 passed`
    - Live local `vllm-mlx` evidence:
      - `template create --from-document ... --name RuntimeProbeDocV2` now produced structured `findings: list[object]` with typed nested fields (`level`, `disc_condition(enum)`, `bulge(bool)`, etc.).
      - `extract --template /tmp/runtime_probe_doc_v2.yaml` succeeded and produced 6 structured finding rows with grounded values.
  - Schema granularity benchmark harness + first full adversarial run (`2026-02-25 08:47`, `SCHEMA-002`):
    - Added reproducible benchmark harness:
      - `scripts/run_schema_granularity_benchmark.py`
      - `tests/datasets/evaluation/schema_granularity_cases.json`
    - Harness behavior:
      - Runs baseline (`semantic gate off`) vs hybrid (`semantic gate on + DSPy assessor`) across multiple report styles.
      - Disables DSPy memory/disk cache for fair mode comparison.
      - Clears cache paths between case/mode executions.
      - Persists machine-readable and markdown artifacts.
    - Full-run artifact:
      - `docs/runs/2026-02-25-082136-schema-granularity-benchmark/schema_granularity_results.json`
      - `docs/runs/2026-02-25-082136-schema-granularity-benchmark/schema_granularity_report.md`
    - Result summary (full 5-case run):
      - `extraction_success_rate`: baseline `1.00`, hybrid `1.00` (no regression in extraction success).
      - `semantic_gate_trigger_rate`: baseline `0.00`, hybrid `0.20` (gate exercised on hard case).
      - Mixed aggregate quality signal (hybrid not uniformly better yet):
        - `semantic_score_mean`: baseline `0.5872`, hybrid `0.5759`.
        - `mean_enum_fields`: baseline `1.8`, hybrid `2.6`.
      - Interpretation: gating is active and improves typing in some scenarios, but robustness is not consistently superior across all cases yet; next step is optimizer-tuned and multi-trial evaluation in `#57`.
  - Schema granularity tuning pass + net-positive benchmark (`2026-02-25 10:14`, `SCHEMA-002`):
    - Hardened `SchemaGenerator` semantic policy in `mosaicx/pipelines/schema_gen.py`:
      - borderline-only LLM semantic assessor invocation (reduces unnecessary LLM critique calls),
      - required-coverage regression guard inside semantic gating,
      - candidate ranking/selection with required-coverage priority to prevent quality regressions.
    - Added runtime validation detail channel (`required_coverage`) and expanded schema tests:
      - `tests/test_schema_gen.py` now covers:
        - runtime coverage details,
        - assessor-skip behavior when deterministic signal is clear,
        - coverage-regression blocking in repair flow.
    - Validation:
      - `scripts/clear_dspy_cache.sh && PYTHONPATH=. .venv/bin/pytest -q tests/test_schema_gen.py` -> `35 passed`
      - `scripts/clear_dspy_cache.sh && PYTHONPATH=. .venv/bin/pytest -q tests/test_cli_integration.py -k "template_create"` -> `9 passed`
    - Full benchmark artifact:
      - `docs/runs/2026-02-25-095630-schema-granularity-benchmark/schema_granularity_results.json`
      - `docs/runs/2026-02-25-095630-schema-granularity-benchmark/schema_granularity_report.md`
    - Aggregate deltas (baseline -> hybrid):
      - `semantic_score_mean`: `0.5604` -> `0.5931` (`+0.0327`)
      - `required_coverage_mean`: `1.0000` -> `1.0000` (no regression)
      - `extraction_success_rate`: `1.0000` -> `1.0000` (no regression)
      - `mean_enum_fields`: `1.0` -> `1.8` (`+0.8`)
  - Describe-only adversarial benchmark + closure evidence (`2026-02-25 10:40`, `SCHEMA-001`):
    - Extended benchmark harness to support explicit generation context:
      - `scripts/run_schema_granularity_benchmark.py --generation-context describe_only`
    - Full local `vllm-mlx` run artifact:
      - `docs/runs/2026-02-25-102321-schema-granularity-benchmark/schema_granularity_results.json`
      - `docs/runs/2026-02-25-102321-schema-granularity-benchmark/schema_granularity_report.md`
    - Describe-only aggregate:
      - `required_coverage_mean`: `0.9000` -> `0.9333`
      - `extraction_success_rate`: `1.0000` -> `1.0000`
      - `repeated_structure_pass_rate`: `1.0000` -> `1.0000`
      - `enum_pass_rate`: `1.0000` -> `1.0000`
    - Fresh live checks with cache clears:
      - `template create --describe ... --name Schema56DescribeCheck` -> `/tmp/schema56_describe_check.yaml` (`9 fields`, local 120B)
      - `template create --from-document ... --name Schema56DocCheck` -> `/tmp/schema56_doc_check.yaml` (`13 fields`, local 120B)
      - `extract --template /tmp/schema56_doc_check.yaml ...` -> `/tmp/schema56_doc_extract.json` (successful structured extraction)

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
  - Pinned hard-test execution to repository source by exporting `PYTHONPATH=$ROOT_DIR` inside `scripts/run_hard_test_matrix.sh`.
    - this removes drift where `.venv/bin/mosaicx` could execute stale installed package code instead of current branch code.
    - validated with local full matrix run: `docs/runs/2026-02-25-121759-hard-test-matrix.md`
  - Added post-fix smoke validation (non-LLM mode) to verify script startup and command logging:
    - `docs/runs/2026-02-25-122302-hard-test-matrix.md`
  - Added post-EXT-003 full matrix validation (LLM + integration + query-engine integration):
    - `docs/runs/2026-02-25-124924-hard-test-matrix.md`
  - Added local helpers required by matrix runner:
    - `scripts/ensure_vllm_mlx_server.sh`
    - `scripts/generate_hard_test_fixtures.py`
    - `scripts/clear_dspy_cache.sh`

- Extract/template parity hardening (`2026-02-25`, `EXT-003`):
  - Ensured optional enum fields preserve explicit absence semantics across schema generation, template compilation, and extraction coercion:
    - `mosaicx/pipelines/schema_gen.py` now appends `none`/`None` to optional enums when no absence token is declared.
    - `mosaicx/schemas/template_compiler.py` now normalizes parsed templates to include absence enum values for optional enum fields.
    - `mosaicx/pipelines/extraction.py` now maps null/nullish values to declared absence enum members for Optional[Enum] annotations.
  - Hardened schema-mode extraction post-processing:
    - coercion now runs even when DSPy returns a fully-typed model instance, so normalization/cleanup is consistently applied.
  - Added spinal level canonicalization in extraction coercion (`C2-3` / unicode hyphen variants -> `C2-C3`).
  - Regression coverage:
    - `tests/test_schema_gen.py`
    - `tests/test_template_compiler.py`
    - `tests/test_extraction_schema_coercion.py`
    - `tests/test_extraction_pipeline.py`
    - `tests/test_cli_extract.py`
  - Validation:
    - `PYTHONPATH=. .venv/bin/pytest -q tests/test_extraction_pipeline.py tests/test_extraction_schema_coercion.py tests/test_template_compiler.py tests/test_schema_gen.py tests/test_cli_extract.py` -> `70 passed`
  - Live local `vllm-mlx` check on cervical sample:
    - `MRICervicalSpineV3` now renders level findings with canonical levels and human-readable enum values (for example, `C2-C3`, `Disc`, `None`) instead of internal labels/null-heavy output.

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
