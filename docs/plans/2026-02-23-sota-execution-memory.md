# MOSAICX SOTA Execution Memory

Date: 2026-02-23
Status: Active
Owner: Core platform
Primary blueprint: docs/plans/2026-02-23-mosaicx-sota-blueprint.md
Progress tag: [SOTA Rollout: 10%]
Canonical status board: docs/plans/2026-02-24-roadmap-status-audit.md

Note:
- This file is a session execution ledger (append-only progress memory).
- Do not treat checkbox totals here as canonical rollout completion.
- Canonical project status, issue mapping, and acceptance evidence live in
  `docs/plans/2026-02-24-roadmap-status-audit.md`.

## 1) Session Boot Protocol (anti-context-rot)

Every coding session must start with:

1. Read `docs/plans/2026-02-23-mosaicx-sota-blueprint.md`.
2. Read this file and continue from first unchecked task.
3. Run `git status --short` and avoid unrelated files.
4. Execute only one atomic task group at a time.
5. Update this file with:
- changed files
- test commands
- test outcomes
- commit hash

## 2) Execution Rule Set

1. No placeholder implementation.
2. No silent fallback that hides correctness risk.
3. No merge without tests for changed behavior.
4. No new lexical hack in place of model planning if DSPy module can solve it.
5. Deterministic truth remains mandatory for numeric analytics.

## 3) Atomic Task Backlog

### 3.1 Critical Bug Queue (must not regress)

- [x] `BUG-001` Fix nested `list[object]` template generation so object `fields` are preserved end-to-end.
  Source: `docs/bugs/nested-object-fields-lost-in-template-generation.md`
  Scope:
  - `mosaicx/pipelines/schema_gen.py`
  - `mosaicx/schemas/template_compiler.py`
  - `tests/test_schema_gen.py`
  - `tests/test_report.py`
  Acceptance:
  - `mosaicx template create` emits valid nested object `fields`.
  - extraction with generated template no longer crashes on `list[object]`.

### 3.2 DSPy Full-Capability Ledger (do not skip)

Each item requires code path + tests + integration evidence before check-off.

- [ ] `DSPY-01` `ReAct` planner as primary control-plane planner for ambiguous/mixed query flows.
- [ ] `DSPY-02` `RLM` executor for long-context document reasoning in query and verify-thorough.
- [ ] `DSPY-03` `JSONAdapter` default for structured stages with explicit fallback policy.
- [ ] `DSPY-04` `TwoStepAdapter` fallback for reasoning-heavy parse-unstable stages.
- [ ] `DSPY-05` `BestOfN` candidate selection in evidence verification / grounded answer revision.
- [ ] `DSPY-06` `Refine` self-improvement for extraction-critical steps (findings/impression/diagnosis).
- [ ] `DSPY-07` `MultiChainComparison` contradiction adjudication where multiple candidates exist.
- [ ] `DSPY-08` `CodeAct` controlled fallback for complex tabular/program synthesis.
- [ ] `DSPY-09` `ProgramOfThought` fallback-only with strict safety/timeout/error boundaries.
- [ ] `DSPY-10` `CompleteAndGrounded` in evaluation gates for grounding quality.
- [ ] `DSPY-11` `answer_exact_match` and `answer_passage_match` in CI gate set.
- [ ] `DSPY-12` `MIPROv2` baseline optimization run stored with artifacts.
- [ ] `DSPY-13` `SIMBA` sample-efficient optimization run stored with artifacts.
- [ ] `DSPY-14` `GEPA` heavy optimization run stored with artifacts.
- [ ] `DSPY-15` `Parallel/Async/Streaming` where beneficial, with strict failure isolation.

### A. Extract V2

- [ ] `A1` Implement `AssuredDocumentExtractor` stage pipeline in `mosaicx/pipelines/extraction.py`.
- [ ] `A2` Add field-level extraction verdict schema and wire into output envelope.
- [ ] `A3` Add verifier+repair loop for unsupported/mismatch fields.
- [ ] `A4` Enable assured mode in CLI/SDK (`--assured`, SDK flag).
- [ ] `A5` Add extraction regressions for adversarial clinical examples.

### B. Query V3

- [ ] `B1` Enforce explicit Route/Plan/Execute/Verify/Narrate staging in `mosaicx/query/engine.py`.
- [ ] `B2` Tighten deterministic tabular protocol and reject unsupported numeric narration.
- [ ] `B3` Harden program synthesis fallback boundaries (CodeAct/PoT).
- [ ] `B4` Expand structured session state graph in `mosaicx/query/session.py`.
- [ ] `B5` Add adversarial multi-turn integration tests (pronouns, corrections, temporal deltas, ambiguous asks).

### C. Verify V2

- [ ] `C1` Unify truth contract fields in all levels (`claim_true`, `decision`, `grounded`, `support_score`).
- [ ] `C2` Add contradiction-first rendering contract in CLI and SDK outputs.
- [ ] `C3` Strengthen thorough audit decomposition and evidence span consistency checks.
- [ ] `C4` Add long-document adversarial integration tests.

### D. Runtime/Adapter Reliability

- [ ] `D1` Add startup checks for Deno, interpreter, and LM endpoint readiness.
- [ ] `D2` Add adapter selection policy (`JSONAdapter` first, `TwoStepAdapter` fallback).
- [ ] `D3` Add parse-retry and error taxonomy for structured failures.
- [ ] `D4` Ensure zero crash policy for query/verify path failures.

### E. Evaluation + Optimization

- [ ] `E1` Expand quality-gate dataset coverage for extract/query/verify.
- [ ] `E2` Enforce gate blocking in CI for release paths.
- [ ] `E3` Run optimizer sequence (`MIPROv2` -> `SIMBA` -> `GEPA`) and save artifacts.
- [ ] `E4` Add benchmark report artifact generator.

### F. Docs + Operator UX

- [ ] `F1` Rewrite README around trust workflow (`extract -> verify -> query`).
- [ ] `F2` Add quickstart with local vLLM-MLX 120B and troubleshooting.
- [ ] `F3` Add production playbook and observability runbook.

## 4) Integration Test Matrix (must run)

### Query integration

- [ ] Numeric distribution with non-trivial column naming
- [ ] Follow-up references (`how many are there`, `what are they`, `which one changed`)
- [ ] Temporal comparison from two+ reports
- [ ] Ambiguous user phrasing and typo robustness
- [ ] Mixed query with text evidence and table computation

### Verify integration

- [ ] Correct claim with exact support span
- [ ] Contradicted claim with explicit claimed vs source evidence
- [ ] Insufficient evidence handling
- [ ] Long-document claim where evidence appears late in source

### Extract integration

- [ ] Radiology measurements and impression consistency
- [ ] Pathology staging and biomarker extraction
- [ ] Auto schema extraction on heterogeneous free-text docs
- [ ] Adversarial negation and temporal phrasing

## 5) Decision Log

- `DL-001` Persistent memory is file-based. Model memory is not durable across sessions.
- `DL-002` Deterministic data plane is authoritative for numeric truth.
- `DL-003` ProgramOfThought remains fallback-only due syntax/runtime volatility.
- `DL-004` Any confidence metric without evidence is non-authoritative.
- `DL-005` Local LLM validation for `/v1/chat/completions` may be blocked by sandbox control-paths; use escalated command path when validating real token/GPU usage.

## 6) Current Execution Snapshot

- Active phase: `Phase 0`
- Active tasks: `A1, B1, C1, D1, E1`
- Last update: 2026-02-23
- Next action: Implement and validate Query V3 Route/Plan/Execute/Verify/Narrate staging (`B1`).

## 6.1 Operational Notes (Local LLM Validation)

1. `curl /v1/models` can succeed while direct `/v1/chat/completions` is blocked in default sandbox path.
2. For proof of real local inference (and GPU/token consumption), run `/v1/chat/completions` via approved escalated command prefix.
3. Always cap `max_tokens` during validation checks to avoid accidental long generations/cost.
4. In this environment, local `vllm-mlx` is the default integration backend; do not treat non-LLM deterministic fallbacks as full integration validation.
5. Minimum preflight before integration runs:
- `curl -sS --max-time 5 http://127.0.0.1:8000/v1/models`
- `curl -sS --max-time 120 http://127.0.0.1:8000/v1/chat/completions ... \"max_tokens\":8`

## 7) Update Template (append per work block)

```
### Update YYYY-MM-DD HH:MM
- Tasks completed:
- Files changed:
- Tests run:
- Results:
- Commit:
- Remaining blockers:
```

## 8) Capability Evidence Format

When completing `DSPY-*` items, append:

1. Code path(s): exact file references.
2. Unit/functional tests: command + result.
3. Integration evidence: command + observed behavior.
4. Safety boundary: fallback/guard behavior under failure.

### Update 2026-02-23 18:05
- Tasks completed:
  - `BUG-001` nested `list[object]` template generation fixed end-to-end.
- Files changed:
  - `mosaicx/pipelines/schema_gen.py`
  - `mosaicx/schemas/template_compiler.py`
  - `tests/test_schema_gen.py`
  - `tests/test_report.py`
- Tests run:
  - `.venv/bin/pytest -q tests/test_schema_gen.py tests/test_report.py`
- Results:
  - `68 passed` (0 failures).
  - Added regression coverage for:
    - compile of `list[object]` with nested fields,
    - fallback behavior for `list[object]` with missing nested fields,
    - YAML conversion roundtrip with nested object items.
- Commit:
  - pending
- Remaining blockers:
  - Production validation of `mosaicx template create --from-radreport` on real report IDs still pending.

### Update 2026-02-23 19:08
- Tasks completed:
  - Established persistent hard-test framework for edge-case coverage.
  - Added synthetic multi-format fixture generator for adversarial cases.
  - Added reproducible hard-test runner with deterministic/integration/LLM modes.
  - Logged deterministic and partial LLM preflight runs in `docs/runs/`.
- Files changed:
  - `docs/testing/edge-case-matrix.md`
  - `scripts/generate_hard_test_fixtures.py`
  - `scripts/run_hard_test_matrix.sh`
  - `docs/runs/2026-02-23-190126-hard-test-matrix.md`
  - `docs/runs/2026-02-23-190535-hard-test-matrix.md`
  - `docs/runs/2026-02-23-190703-hard-test-matrix.md`
- Tests run:
  - `MOSAICX_HARDTEST_INTEGRATION=1 MOSAICX_HARDTEST_LLM=0 scripts/run_hard_test_matrix.sh`
  - `MOSAICX_HARDTEST_INTEGRATION=0 MOSAICX_HARDTEST_LLM=0 scripts/run_hard_test_matrix.sh`
  - `MOSAICX_HARDTEST_INTEGRATION=0 MOSAICX_HARDTEST_LLM=1 scripts/run_hard_test_matrix.sh`
- Results:
  - Deterministic suite passes cleanly (`68 + 56 + 47 + 24` tests across packs).
  - Integration pack (`tests/integration/test_full_pipeline.py`) passes.
  - LLM preflight currently fails when local server is not listening on `127.0.0.1:8000`; runner now records this and skips LLM checks instead of aborting.
  - Fixture generator produced 11 artifacts across text/tabular/image/pdf; DOCX/PPTX generation skipped if optional deps missing.
- Commit:
  - pending
- Remaining blockers:
  - Need full LLM-mode pass while `vllm-mlx` server is live for `verify/query` runtime assertions.
  - Need expansion of adversarial integration tests for Query V3 stateful follow-ups (`B5`).

### Update 2026-02-23 19:45
- Tasks completed:
  - Hardened query loaders for real-world tabular/text ingestion (`csv/tsv`, multiline JSONL, decode fallback, large-file guard, multi-sheet Excel).
  - Fixed verify deterministic truth semantics for empty/uncheckable extraction (`insufficient_evidence` instead of false-positive `verified`).
  - Added query control-plane guardrails for conversational phrasing (`i mean ...`) and safer aggregate-op detection.
  - Added robust runtime fallback for writable `DENO_DIR` in sandboxed environments.
  - Validated live `vllm-mlx` behavior against the OCELOT CSV with local 120B model using `PYTHONPATH=. .venv/bin/mosaicx ...`.
- Files changed:
  - `mosaicx/query/loaders.py`
  - `mosaicx/query/control_plane.py`
  - `mosaicx/query/engine.py`
  - `mosaicx/query/tools.py`
  - `mosaicx/verify/deterministic.py`
  - `mosaicx/verify/engine.py`
  - `mosaicx/runtime_env.py`
  - `tests/test_query_loaders.py`
  - `tests/test_query_control_plane.py`
  - `tests/test_query_engine.py`
  - `tests/test_query_tools.py`
  - `tests/test_verify_deterministic.py`
  - `tests/test_verify_engine.py`
  - `tests/test_runtime_env.py`
- Tests run:
  - `.venv/bin/pytest -q tests/test_query_loaders.py`
  - `.venv/bin/pytest -q tests/test_verify_deterministic.py tests/test_verify_engine.py`
  - `.venv/bin/pytest -q tests/test_query_control_plane.py tests/test_query_engine.py -m 'not integration'`
  - `.venv/bin/pytest -q tests/test_query_loaders.py tests/test_verify_deterministic.py tests/test_verify_engine.py tests/test_query_control_plane.py tests/test_query_tools.py tests/test_query_engine.py -m 'not integration'`
  - `.venv/bin/pytest -q tests/test_runtime_env.py`
- Results:
  - All listed packs pass (`159 passed, 2 deselected` for combined regression set; runtime env tests pass).
  - Live LLM checks on real CSV now return correct values for:
    - sex distribution (`M=700, F=379`)
    - ethnicity values (`Japanese, Caucasian`)
    - schema count follow-up (`127 columns`) with typo/filler phrasing.
  - Outstanding runtime warning remains from DSPy disk cache path (`memory-only cache fallback`) and will be normalized next.
- Commit:
  - pending
- Remaining blockers:
  - Normalize DSPy disk-cache directory to a guaranteed writable path in this environment.
  - Run full hard test matrix with server-first gate active and record final overnight run artifacts.
  - Continue DSPy roadmap implementation items (`DSPY-01`, `DSPY-03`, `DSPY-04`, `B1`, `B5`) with integration evidence logs.

### Update 2026-02-24 09:10
- Tasks completed:
  - Created/normalized roadmap GitHub issues and deduplicated duplicates.
  - Implemented `VER-001` coherence guard for claim verification:
    - matching grounded claim/source values now rescue inconclusive partial verdicts to `verified`,
    - contradiction path remains strict and unchanged.
  - Fixed post-rescue coherence so claim evidence/citations no longer show "not found" prose when grounded source value is present.
  - Added regression coverage for the exact failure mode (unsupported verdict + matching source value).
- Files changed:
  - `mosaicx/sdk.py`
  - `tests/test_sdk_verify.py`
- Tests run:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_sdk_verify.py -k "claim_value_conflict_overrides_verified_decision or thorough_claim_matching_source_rescues_partial_verdict"`
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_verify_engine.py tests/test_verify_audit.py tests/test_sdk_verify.py`
  - `curl -sS --max-time 5 http://127.0.0.1:8000/v1/models`
  - `PYTHONPATH=. .venv/bin/mosaicx verify --sources tests/datasets/extract/sample_patient_vitals.pdf --claim "patient BP is 128/82" --level thorough`
- Results:
  - Regression tests pass.
  - Verify-focused suite passes (`85 passed`).
  - Live local 120B verify run now returns coherent output:
    - `Decision: verified`, `Truth: true`, `Support score: 1.00`,
    - `Claim Comparison` and `Evidence` both grounded on the same BP source signal.
- Commit:
  - pending
- Remaining blockers:
  - Finish and land `QRY-001` direct-answer contract improvements for count+values prompts.
  - Continue DSPy roadmap staging (`DSPY-01`, `DSPY-03`, `DSPY-04`, `B1`) and log integration evidence per item.

### Update 2026-02-24 10:05
- Tasks completed:
  - Advanced `QRY-002` planner-first behavior in query engine.
  - Added planner column-recovery path so valid ReAct intent/source plans are not discarded when `column` is omitted.
  - Column recovery now uses LM-guided selection with session-state fallback for follow-up/coreference prompts.
  - Reduced lexical routing pressure by keeping execution on planned path when recovered column is available.
- Files changed:
  - `mosaicx/query/engine.py`
  - `tests/test_query_engine.py`
- Tests run:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py -k "planner_executes_count_values_plan_before_lexical_fallback or planner_recovers_missing_column_via_llm_before_fallback"`
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py -k "planner or count_values_after_schema_turns_uses_topic_not_prior_column_dump or schema_followup_how_many_are_there_refers_to_columns or ask_structured_count_values_accepts_table_value_computed_evidence"`
  - `PYTHONPATH=. .venv/bin/mosaicx query --document "...ALL_data_complete_SUVbw+SUVlbm_N1079.csv" -q "how many genders are there and what are they?" --eda --trace --max-iterations 2`
- Results:
  - New planner regression test passes.
  - Planner/count-values regression subset passes.
  - Live local 120B run returns grounded distribution answer (`M=700, F=379`) with computed evidence.
- Commit:
  - pending
- Remaining blockers:
  - Continue `QRY-002` by surfacing explicit planner/execution trace fields in structured payload.
  - Continue DSPy roadmap items (`DSPY-01`, `DSPY-03`, `DSPY-04`) with end-to-end integration evidence.

### Update 2026-02-24 10:35
- Tasks completed:
  - Improved grounding confidence calibration for deterministic tabular answers with computed evidence.
  - Added explicit computed-evidence boost when counts/values in answer are directly supported by `table_stat`/`table_value` citations.
  - Added guard to suppress over-boost when answer introduces unsupported category labels.
- Files changed:
  - `mosaicx/query/engine.py`
  - `tests/test_query_engine.py`
- Tests run:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py -k "planner_recovers_missing_column_via_llm_before_fallback or confidence_high_for_tabular_distribution_with_computed_values or confidence_not_overboosted_for_unbacked_tabular_label or confidence_high_for_short_modality_answers or confidence_drops_for_answer_not_in_evidence"`
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py -k "planner or count_values or schema_followup_how_many_are_there_refers_to_columns or ask_structured_count_values_accepts_table_value_computed_evidence or confidence"`
  - `PYTHONPATH=. .venv/bin/mosaicx query --document "...ALL_data_complete_SUVbw+SUVlbm_N1079.csv" -q "what is the distribution of male and female in the cohort?" --eda --trace --max-iterations 2`
- Results:
  - Confidence regressions pass.
  - Planner/count-values/confidence subset passes.
  - Live local 120B run now reports `Grounding: high (0.82)` for sex-distribution answer with computed citations.
- Commit:
  - pending
- Remaining blockers:
  - Add explicit planner/execution trace fields to query structured payload.
  - Continue DSPy roadmap execution (`DSPY-01`, `DSPY-03`, `DSPY-04`) with integration evidence.

### Update 2026-02-24 11:05
- Tasks completed:
  - Exposed planner/execution trace fields in `query` structured payload for easier developer debugging.
  - Added trace outputs: `execution_path`, `target_resolution`, `planner_used`, `planner_executed`, `planner_intent`, `planner_source`, `planner_column`, `planner_column_recovered`.
  - Verified full query engine suite including integration tests against local 120B endpoint.
- Files changed:
  - `mosaicx/query/engine.py`
  - `tests/test_query_engine.py`
- Tests run:
  - `PYTHONPATH=. .venv/bin/pytest -x -vv tests/test_query_engine.py`
- Results:
  - Full query engine suite passes (`62 passed`, including integration tests).
- Commit:
  - `6094678` Expose planner/target trace fields in structured query payload
- Remaining blockers:
  - Continue `VER-002` contract hardening and claim comparison coherence.
  - Continue DSPy roadmap execution and integration logging.

### Update 2026-02-24 11:30
- Tasks completed:
  - Fixed claim comparison grounding bug where BP conflicts could collapse to partial numeric tokens (`120`) instead of full BP pairs (`128/82`).
  - Added BP-aware source selection with context scoring to prefer measured values over nearby reference ranges.
  - Added regression coverage for quick-verify contradiction output to ensure full grounded BP value is preserved in `claim_comparison` and issue detail.
  - Confirmed local `vllm-mlx` endpoint health before/after verification runs.
- Files changed:
  - `mosaicx/sdk.py`
  - `tests/test_sdk_verify.py`
- Tests run:
  - `curl -sS --max-time 5 http://127.0.0.1:8000/v1/models`
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_sdk_verify.py -k "claim_value_conflict_overrides_verified_decision or claim_value_conflict_prefers_full_bp_pair_from_source or thorough_claim_matching_source_rescues_partial_verdict"`
  - `PYTHONPATH=. .venv/bin/mosaicx verify --sources tests/datasets/extract/sample_patient_vitals.pdf --claim "patient BP is 120/82" --level quick -o /tmp/verify_false_quick_after.json`
- Results:
  - Targeted SDK verify tests pass.
  - Quick verify output now reports `source=128/82` and conflict detail references the full grounded value.
- Commit:
  - pending
- Remaining blockers:
  - Land/close `VER-002` with any remaining CLI/JSON contract cleanup.
  - Continue roadmap issues: `QRY-001`, `EXT-001`, `EXT-002`, `OPS-001`, `EVAL-001`.

### Update 2026-02-24 11:45
- Tasks completed:
  - Unified claim-mode machine contract by aligning `verdict` with final `decision`.
  - Eliminated conflicting states where `verdict` and `decision` could disagree for claim checks.
  - Preserved canonical `claim_true` while keeping `decision/verdict` synchronized for SDK and CLI consumers.
- Files changed:
  - `mosaicx/sdk.py`
  - `tests/test_sdk_verify.py`
- Tests run:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_sdk_verify.py`
  - `PYTHONPATH=. .venv/bin/mosaicx verify --sources tests/datasets/extract/sample_patient_vitals.pdf --claim "patient BP is 120/82" --level quick -o /tmp/verify_false_quick_after2.json`
- Results:
  - SDK verify test suite passes (`16 passed`).
  - Quick verify now yields coherent machine output:
    - `verdict=contradicted`, `decision=contradicted`, `claim_true=false`,
    - `claim_comparison.source=128/82`.
- Commit:
  - pending
- Remaining blockers:
  - Continue `QRY-001` direct-answer contract hardening and close `QRY-002`/`VER-002` issues after final sweep.
  - Advance roadmap items `EXT-001`, `EXT-002`, `OPS-001`, `EVAL-001`.

### Update 2026-02-24 12:30
- Tasks completed:
  - Implemented `EXT-001`: local OpenAI-compatible model/provider normalization.
  - `make_harmony_lm` now normalizes local `api_base` hosts and auto-prefixes unqualified local model IDs as `openai/<model>` for stable LiteLLM routing.
  - Implemented `EXT-002`: enum/internal-label rendering cleanup in extraction CLI output.
  - Added enum humanization for internal template enum strings and Enum instances in nested/list displays.
  - Reproduced and verified `MRICervicalReportV2` extraction output now shows clean values (`Normal`, `None`, `Disc`) instead of internal enum paths.
- Files changed:
  - `mosaicx/metrics.py`
  - `mosaicx/cli_display.py`
  - `tests/test_metrics.py`
  - `tests/test_cli_display.py`
  - `tests/test_cli_extract.py`
- Tests run:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_metrics.py tests/test_cli_display.py`
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_cli_extract.py tests/test_cli_display.py tests/test_metrics.py`
  - `PYTHONPATH=. .venv/bin/mosaicx extract --document tests/datasets/standardize/Sample_Report_Cervical_Spine.pdf --template MRICervicalReportV2` (escalated, local vllm-mlx)
- Results:
  - Focused test suites pass (`26 passed`).
  - End-to-end extract output confirms enum rendering fix on cervical template.
- Commit:
  - pending
- Remaining blockers:
  - Close `EXT-001` and `EXT-002` issues after commit + push.
  - Implement `OPS-001` LLM endpoint preflight gating for LLM-dependent integration tests.
  - Implement `EVAL-001` quality-gate metrics expansion.

### Update 2026-02-24 13:05
- Tasks completed:
  - Implemented `OPS-001` endpoint preflight utility for OpenAI-compatible LLM servers.
  - Added `check_openai_endpoint_ready` in `runtime_env` to validate both `/models` and `/chat/completions`, with explicit failure reasons and selected model id.
  - Wired query integration fixture to gate on endpoint preflight and skip with actionable reason when endpoint is unreachable.
  - Strengthened hard-test matrix preflight to verify both endpoints (not just `/models`) before LLM-specific checks.
- Files changed:
  - `mosaicx/runtime_env.py`
  - `tests/test_runtime_env.py`
  - `tests/test_query_engine.py`
  - `scripts/run_hard_test_matrix.sh`
- Tests run:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_runtime_env.py` (`13 passed`)
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py -m integration -k ask_returns_string` (sandbox: skipped with explicit preflight reason)
  - escalated: `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py -m integration -k ask_returns_string` (`1 passed`)
- Results:
  - Preflight now fails fast with explicit reason in constrained/sandboxed runs.
  - With local vllm-mlx reachable, integration test passes end-to-end.
- Commit:
  - pending
- Remaining blockers:
  - Close `OPS-001` issue after commit + push.
  - Implement `EVAL-001` and `EVAL-002` quality/optimizer steps.
  - Continue `QRY-001` and remaining lexical-routing cleanup on `QRY-002`.

### Update 2026-02-24 15:05
- Tasks completed:
  - Hardened claim-mode contradiction normalization so contradictory outcomes do not leak secondary `unsupported` signals in machine fields.
  - In conflict path, SDK now canonicalizes claim verdict payload to:
    - `decision=contradicted`
    - `claim_true=false`
    - `field_verdicts[0].status=mismatch`
    - claim-level non-critical absence-language issues/citations filtered out.
  - Added absence-language cue coverage for patterns like `found: []` / `values found: []`.
  - Added regression test for contradiction-mode canonicalization.
  - Revalidated both false and true claim cases with live local `vllm-mlx` (`mlx-community/gpt-oss-120b-4bit`) in thorough mode.
  - Documented execution rule for on-prem runs: do not interrupt long `thorough` jobs; wait until terminal completion.
- Files changed:
  - `mosaicx/sdk.py`
  - `tests/test_sdk_verify.py`
- Tests run:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_sdk_verify.py` (`18 passed`)
  - `PYTHONPATH=. .venv/bin/mosaicx verify --sources tests/datasets/extract/sample_patient_vitals.pdf --claim "patient BP is 120/82" --level thorough -o /tmp/mosaicx_verify_false_after3.json` (live 120B)
  - `PYTHONPATH=. .venv/bin/mosaicx verify --sources tests/datasets/extract/sample_patient_vitals.pdf --claim "patient BP is 128/82" --level thorough -o /tmp/mosaicx_verify_true_after4.json` (live 120B)
- Results:
  - False claim JSON is coherent and contradiction-first:
    - `decision=contradicted`, `claim_true=false`, `support_score=0.0`, `grounded=true`
    - `claim_comparison.source=128/82`
    - `field_verdicts[0].status=mismatch`
    - no claim-level `unsupported` residue.
  - True claim JSON is coherent and verification-first:
    - `decision=verified`, `claim_true=true`, `support_score=1.0`, `grounded=true`
    - `claim_comparison.source=128/82`
    - `field_verdicts[0].status=verified`
    - `issues=[]`.
- Ops notes:
  - Starting `vllm-mlx` from sandboxed runs can fail or be reaped; reliable startup in this environment used detached `screen`:
    - `screen -dmS mosaicx_vllm zsh -lc '/Users/nutellabear/.local/bin/vllm-mlx serve mlx-community/gpt-oss-120b-4bit --port 8000 --continuous-batching >/tmp/vllm_mlx_server.log 2>&1'`
  - Health checks should include both endpoints:
    - `/v1/models`
    - `/v1/chat/completions`

### Update 2026-02-24 15:40
- Tasks completed:
  - Added query execution telemetry to disambiguate deterministic-vs-LLM paths per turn.
  - Added payload fields:
    - `execution_mode` (`deterministic_truth`, `hybrid_planner_deterministic`, `llm_generation`, `deterministic_fallback`)
    - `llm_primary_used` (bool)
    - `planner_error` (string when LLM planner fails and lexical fallback is used)
  - Extended CLI `--trace` diagnostics to render:
    - `execution_mode`, `llm_primary_used`, `execution_path`, `target_resolution`
    - `planner_used`, `planner_executed`, `planner_error`
  - Added planner error capture in `ReActTabularPlanner`:
    - explicit categories for import/config/react/predict/invalid-plan failures.
  - Verified trace now shows root cause instead of silent fallback.
- Files changed:
  - `mosaicx/query/control_plane.py`
  - `mosaicx/query/engine.py`
  - `mosaicx/cli.py`
- Tests run:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_control_plane.py tests/test_query_engine.py -m 'not integration'` (`71 passed`)
  - `PYTHONPATH=. .venv/bin/mosaicx query --document tests/datasets/generated_hardcases/cohort_edge_cases.csv --question "how many ethnicities are there and what are they?" --eda --trace`
- Results:
  - Trace now clearly reports when LM planner is unavailable (example: `planner_error=planner_predict_error:InternalServerError`) while deterministic answer remains grounded.
  - This removes ambiguity when GPU appears idle by showing whether LM was intentionally skipped or failed.

### Update 2026-02-24 16:45
- Tasks completed:
  - Advanced `QRY-002` planner-first routing for count asks without explicit columns to avoid lexical mis-resolution.
  - Added integration coverage for implicit category-count prompts using live LLM planner path.
  - Implemented DSPy optimizer compatibility hardening across versions (`MIPROv2`, `SIMBA`, `GEPA`) with signature-aware init/compile kwargs.
  - Executed bounded real optimizer sequence (`MIPROv2` -> `SIMBA` -> `GEPA`) on local `vllm-mlx` and persisted artifacts.
  - Persisted baseline metrics and before/after report for `EVAL-002`.
  - Canonicalized roadmap docs for `DOC-001` and marked historical files accordingly.
  - Added periodic automation hook script for optimizer artifact refresh.
- Files changed:
  - `mosaicx/query/engine.py`
  - `tests/test_query_engine.py`
  - `mosaicx/evaluation/optimize.py`
  - `scripts/run_optimizer_sequence.py`
  - `scripts/run_eval_optimizer_artifacts.sh`
  - `tests/datasets/evaluation/query_optimizer_train.jsonl`
  - `tests/datasets/evaluation/query_optimizer_val.jsonl`
  - `tests/datasets/evaluation/query_optimizer_tiny_train.jsonl`
  - `tests/datasets/evaluation/query_optimizer_tiny_val.jsonl`
  - `tests/datasets/evaluation/verify_optimizer_train.jsonl`
  - `tests/datasets/evaluation/verify_optimizer_val.jsonl`
  - `tests/test_optimize.py`
  - `Makefile`
  - `docs/runs/2026-02-24-query-optimizer-seq-tiny/optimizer_sequence_manifest.json`
  - `docs/runs/2026-02-24-query-optimizer-seq-tiny/baseline_metrics.json`
  - `docs/runs/2026-02-24-eval-002-optimizer-sequence-report.md`
  - `docs/plans/2026-02-23-dspy-full-capability-rollout.md`
  - `docs/plans/2026-02-23-sota-execution-memory.md`
  - `docs/plans/2026-02-24-roadmap-status-audit.md`
- Tests run:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py tests/test_query_control_plane.py -m 'not integration'`
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_query_engine.py -m integration` (escalated)
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_optimize.py`
  - Live multi-turn query validation script on `tests/datasets/generated_hardcases/cohort_edge_cases.csv`
  - `scripts/run_optimizer_sequence.py` on tiny query dataset with `--profile quick` (escalated)
- Results:
  - Query non-integration tests: `73 passed, 4 deselected`.
  - Query integration tests with local LLM: `5 passed, 68 deselected`.
  - Optimize tests: `7 passed`.
  - Optimizer artifacts generated for all three strategies with manifest + baseline metrics.
- Commit:
  - pending
- Remaining blockers:
  - Close roadmap issues in GitHub after final commit/push (`QRY-002`, `EVAL-002`, `DOC-001`).
  - Optional: run bounded optimizer sequence for `verify` pipeline dataset to extend artifact coverage.

### Update 2026-02-24 16:55
- Tasks completed:
  - Extended `EVAL-002` coverage by running bounded optimizer artifacts for `verify` pipeline on local `vllm-mlx`.
  - Added strategy-level compile fallback in optimizer runner to handle DSPy multi-LM incompatibility (`SIMBA` on verify pipeline).
  - Persisted verify optimizer artifacts + baseline metrics + run report.
- Files changed:
  - `mosaicx/evaluation/optimize.py`
  - `scripts/run_eval_optimizer_artifacts.sh`
  - `docs/runs/2026-02-24-eval-002-verify-optimizer-sequence-report.md`
  - `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/optimizer_sequence_manifest.json`
  - `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/baseline_metrics.json`
  - `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/miprov2_optimized.json`
  - `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/simba_optimized.json`
  - `docs/runs/2026-02-24T155229Z-verify-optimizer-seq/gepa_optimized.json`
  - `docs/plans/2026-02-24-roadmap-status-audit.md`
- Tests run:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_optimize.py`
  - `MOSAICX_OPT_PROFILE=quick scripts/run_eval_optimizer_artifacts.sh verify` (escalated)
- Results:
  - Optimize tests pass (`7 passed`).
  - Verify optimizer sequence completed with artifacts.
  - SIMBA strategy uses fallback path on verify module due multi-LM DSPy limitation; fallback is recorded in manifest (`effective_strategy`, `compile_error`).
- Commit:
  - pending
- Remaining blockers:
  - None for this extension slice; optional follow-up is broadening eval datasets for stronger optimizer signal.
