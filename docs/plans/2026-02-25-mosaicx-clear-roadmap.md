# MOSAICX Clear Roadmap (Execution Canonical)

Date: 2026-02-25  
Branch: `feat/dspy-rlm-evolution`  
Status: Active  
Scope: `extract`, `template/schema`, `verify`, `query`, evaluation, ops

## 1) Goal

Make MOSAICX production-grade for clinical/medical-document extraction and truth-grounded querying by combining:

- DSPy planning + reasoning (`ReAct`, `RLM`, structured adapters, adjudication modules)
- Deterministic truth plane for tables (DuckDB/Polars/Pandas)
- Strict evidence/grounding contracts in CLI + SDK

Success means: reliable outputs, stable contracts, adversarial test coverage, and reproducible evaluation artifacts.

## 2) Non-Negotiable Runtime Rules

Before any LLM-dependent integration test or benchmark:

1. Ensure local inference endpoint is healthy:
   - `scripts/ensure_vllm_mlx_server.sh`
2. Clear DSPy caches:
   - `scripts/clear_dspy_cache.sh`
3. Run from repo source, not stale installed package:
   - `PYTHONPATH=. ...`
4. On local Mac (`vllm-mlx`), run one heavy LLM job at a time.
5. Persist every major run artifact under `docs/runs/`.

## 3) Current State Snapshot (2026-02-25)

- `extract`: improved, not final.
- `template/schema`: improved, not final.
- `verify`: improved, not final.
- `query` (tabular): improved, not final.
- `query` (long-document): partial robustness.
- evaluation gates: partially complete.

## 4) Remaining Gaps To Close

1. `verify` still occasionally receives non-JSON audit output and relies on recovery.
2. `verify` confidence is heuristic; not yet calibrated/validated as probability.
3. `query` can drift on ambiguous multi-turn/coreference prompts.
4. `query` tabular correctness needs broader adversarial coverage across arbitrary schemas.
5. long-document QA needs larger hostile benchmark and citation-quality checks.
6. `extract`/template quality is strongly improved for cervical workflows, but broader modality/template matrix is incomplete.
7. schema generation can still produce valid-but-semantically-suboptimal outputs.
8. CI release gates do not yet represent full production-grade quality thresholds across all core modes.

## 5) Roadmap Phases

## Phase A: Verify Hardening (Truth Contract Finalization)

Deliverables:

- Enforce strict structured output contract across all verify levels.
- Minimize recovery path usage; improve first-pass parse success.
- Calibrate confidence against eval set (binning + reliability report).

Acceptance:

- `thorough` claim checks consistently return coherent:
  - `result`, `claim_is_true`, `source_value`, `evidence`, `support_score`, `debug.executed_mode`.
- Parse-recovery usage rate below target threshold on benchmark suite.
- No contradictory top-level vs diagnostic fields.

## Phase B: Query Robustness (Planner + Deterministic Truth Plane)

Deliverables:

- Strengthen planner/executor interface for ambiguous prompts.
- Expand deterministic tabular answer contract:
  - count, count+values, distribution, grouped stats, follow-up coreference.
- Improve structured session memory for tabular multi-turn context.

Acceptance:

- Ambiguous prompts resolved correctly or explicitly clarified.
- Distribution/count+values queries return direct answer + computed evidence.
- Multi-turn follow-ups keep correct entity/column context.

## Phase C: Long-Document QA Robustness

Deliverables:

- Improve retrieval/citation quality and answer-evidence consistency checks.
- Add adversarial long-doc benchmark (contradictions, dispersed evidence, near-duplicates).

Acceptance:

- Higher exactness/groundedness on long-doc suite.
- Better citation relevance and lower unsupported-answer rate.

## Phase D: Extract + Template/Schema Generalization

Deliverables:

- Expand parity/robustness matrix across multiple template families and modalities.
- Harden schema granularity scoring + repair loop on non-spine datasets.
- Keep runtime validation in schema generation mandatory for LLM-created templates.

Acceptance:

- No internal enum-label leakage in CLI/SDK output.
- Stable extraction across legacy + newly generated templates.
- Semantic schema quality improves on benchmark without extraction regressions.

## Phase E: DSPy Optimization + Quality Gates

Deliverables:

- Re-run and persist optimizer artifacts (`MIPROv2`, `SIMBA`, `GEPA`) on current pipelines.
- Enforce gates in CI:
  - `CompleteAndGrounded`
  - `answer_exact_match`
  - `answer_passage_match`
  - numeric exactness for tabular QA

Acceptance:

- Artifacts stored under `docs/runs/` with reproducible commands.
- CI blocks regressions on all core contracts.

## Phase F: Release Readiness

Deliverables:

- Final docs polish for developers and agents:
  - quickstart, troubleshooting, truth-contract reference, eval runbooks.
- Performance and reliability report on local (`vllm-mlx`) and NVIDIA (`vllm`).

Acceptance:

- New developer can run end-to-end quickly and interpret outputs unambiguously.
- Release checklist complete.

## 6) Production-Grade Definition of Done

MOSAICX is "production-grade" only when all are true:

1. `verify` has stable structured contract and high first-pass structured success.
2. `query` tabular answers are exact and evidence-backed on adversarial sets.
3. long-document QA is grounded with reliable citations.
4. `extract` + generated templates are robust across a broad modality/template matrix.
5. Quality gates are enforced in CI with regression-blocking thresholds.
6. Runbooks/docs are clear enough for first-time users to execute and debug.

## 7) Execution Order (Immediate)

1. Phase A (`verify`)  
2. Phase B (`query` tabular)  
3. Phase C (`query` long-doc)  
4. Phase D (`extract` + schema generalization)  
5. Phase E (optimizer + CI gates)  
6. Phase F (release docs/report)

## 8) Extract Atomic Issue Board

Execution is tracked by atomic GitHub issues:

1. `#59 [EXTX-01]` Canonical extract contract + fail-closed semantics across CLI/SDK/MCP (`done`)
2. `#60 [EXTX-02]` ReAct planner-first section routing for extraction (`done`)
3. `#61 [EXTX-03]` Outlines-first structured extraction with DSPy adapter fallback chain (`done`)
4. `#62 [EXTX-04]` Selective BestOfN for uncertain extraction sections (`done`)
5. `#63 [EXTX-05]` MultiChainComparison adjudication + Refine field-level repair (`in_progress`)
6. `#64 [EXTX-06]` Deterministic validators for units/dates/ranges/null semantics (`todo`)
7. `#65 [EXTX-07]` Extract optimizer datasets + MIPROv2/SIMBA/GEPA artifact runs (`todo`)
8. `#66 [EXTX-08]` Adversarial extract integration matrix on local vllm-mlx (`todo`)
9. `#67 [EXTX-09]` CI quality gates for extraction reliability metrics (`todo`)
10. `#68 [EXTX-10]` Manual gold-vs-model extraction comparison protocol and artifacts (`todo`)

Status update policy:

1. Move exactly one issue to `in_progress` at a time.
2. Issue is only `done` after tests pass and run artifact is logged under `docs/runs/`.
3. Roadmap status here must match GitHub issue state.

## 9) Product Feature Backlog (Post-Hardening)

1. `verify --batch` for RAG truth testing with confusion-matrix output.
2. Evidence coordinates in SDK/CLI output (`source`, `page`, `chunk`, `char_start`, `char_end`).
3. Query planner explain mode (plan trace, tool choice reason, column resolution reason).
4. Cross-document contradiction timeline (same entity, conflicting facts over time).
5. Template/schema playground with pre-run validation and auto-repair suggestions.
6. Domain benchmark packs (radiology/pathology/labs/tabular) with one-command eval.
7. Confidence calibration report generator (reliability curve, thresholds by mode).
8. Drift monitor across model/template versions (regression diff report).
9. Human correction capture loop (review edits -> eval corpus).
10. Synthetic edge-case generator per modality and query family.

Feature acceptance baseline:

1. Every feature ships with deterministic tests + at least one local `vllm-mlx` integration run artifact.
2. New outputs must preserve stable machine-readable contracts.
3. No feature can bypass grounding/evidence requirements.

## 10) Artifact Policy

Every major run must record:

- exact command
- model endpoint used
- cache policy used
- pass/fail summary
- output path under `docs/runs/`

If a run is not logged, it does not count as roadmap progress.
