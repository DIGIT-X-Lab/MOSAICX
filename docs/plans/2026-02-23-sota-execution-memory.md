# MOSAICX SOTA Execution Memory

Date: 2026-02-23
Status: Active
Owner: Core platform
Primary blueprint: docs/plans/2026-02-23-mosaicx-sota-blueprint.md
Progress tag: [SOTA Rollout: 10%]

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
