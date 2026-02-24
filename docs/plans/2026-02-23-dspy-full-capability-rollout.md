# MOSAICX DSPy Full-Capability Rollout

**Date:** 2026-02-23  
**Owner:** Core platform (`query`, `verify`, `evaluation`)  
**Status:** Historical plan (superseded)  
**Canonical Status Board:** `docs/plans/2026-02-24-roadmap-status-audit.md`  
**Coverage Target:** Use DSPy as full control plane, deterministic execution as truth plane.

`[Overall Progress: moved to canonical status board]`

> This document captures intended rollout design and target architecture.
> For real implementation status, issue state, and validation evidence, use:
> `docs/plans/2026-02-24-roadmap-status-audit.md`.

## Objective

Make MOSAICX:

1. **Agentically smart** (LLM plans and adapts, not brittle lexical hacks),
2. **Deterministically correct** (numeric/tabular truth always computed),
3. **Grounding-first** (evidence contracts fail closed),
4. **Production-optimizable** (eval + optimizer loop drives continuous gains).

## DSPy Capability Coverage Matrix

| DSPy capability | Target in MOSAICX | Status |
|---|---|---|
| `dspy.RLM` | Long-context tool-use for `query` + `verify thorough` | Implemented |
| `dspy.ReAct` | Explicit planning loop over table/document tools | Implemented |
| `dspy.CodeAct` / `dspy.ProgramOfThought` | Programmatic tabular reasoning fallback for complex analytics | Implemented (ProgramOfThought) |
| `dspy.BestOfN` / `dspy.Refine` | Grounded answer candidate selection | Implemented (`BestOfN`) |
| `dspy.MultiChainComparison` | Contradiction adjudication in answer verification | Implemented |
| `dspy.JSONAdapter` policy | Structured parse robustness + controlled fallback | Implemented |
| `dspy.evaluate.CompleteAndGrounded` | Grounding quality gate in CI/eval | Implemented (metric wrapper) |
| `dspy.SIMBA` -> `dspy.GEPA` | Prompt/program optimization loop on MOSAICX benchmarks | Implemented |

## Architecture Contract

Control plane (DSPy modules):

1. `IntentRouter`
2. `Planner` (`ReAct`)
3. `EvidenceVerifier` (`BestOfN` + `MultiChainComparison`)
4. `Narrator`

Data plane (truth execution):

1. `DuckDB/Polars/Pandas` deterministic tabular compute
2. Chunk-indexed text evidence retrieval with source spans

Guard plane:

1. Strict evidence contract by intent class
2. Numeric questions require computed evidence
3. Long-document answers require span-backed evidence

## Core Functionality Coverage

### `query`

1. Intent route (`text_qa`, `schema`, `count`, `count_values`, `aggregate`, `mixed`)
2. Planner chooses tools and plan fields explicitly
3. Deterministic executor computes truth for tabular analytics
4. EvidenceVerifier can reject/correct unsupported drafts
5. Narrator composes final answer from verified evidence
6. StateStore tracks entities/metrics/timeframe/last source/last column

### `verify`

1. `quick`: deterministic checks
2. `standard`: deterministic + LLM spot-check
3. `thorough`: RLM audit with chunk-aware source tools and no hard 5k truncation dependence
4. Final verdict uses explicit contradiction adjudication

### Other core features impacted

1. `mcp` query/verify endpoints share upgraded engine behavior
2. SDK and CLI share adapter/runtime policy
3. Evaluation/optimize pipeline can tune query/verify prompts/programs

## Implementation Phases

## Phase 1: Query Control Plane (Current)

1. Add dedicated `query/control_plane.py` modules:
   - `IntentRouter`
   - `ReActTabularPlanner`
   - `EvidenceVerifier` (`BestOfN` + `MultiChainComparison`)
2. Integrate into `QueryEngine` and reduce monolithic flow
3. Expand stateful memory beyond prompt history
4. Add tests for:
   - schema turns
   - count+values
   - aggregate
   - follow-up stability

**Exit criteria**

1. Deterministic tabular failures from lexical drift are eliminated on current fixtures.
2. Planner path is used for ambiguous tabular intent.

## Phase 2: Verify Long-Doc Robustness

1. Add chunk index + span tools in `verify/audit.py`
2. Remove direct dependence on `source_text[:5000]` prompt truncation
3. Force evidence-backed claim comparison in both verified and contradicted outcomes
4. Add adversarial long-doc tests

**Exit criteria**

1. Thorough verify remains grounded on long inputs.
2. Evidence paths are explicit in output metadata.

## Phase 3: Adapter + Parse Hardening

1. Set `JSONAdapter` policy in CLI/SDK/package configuration path
2. Add controlled parse retry strategy
3. Normalize fallback reasons to stable machine-readable codes

**Exit criteria**

1. Adapter parse failures are rare and recoverable.
2. No silent degradations.

## Phase 4: Eval + Optimization Loop

1. Add eval tasks:
   - tabular truth
   - long-doc grounded QA
   - multi-turn robustness
2. Wire `CompleteAndGrounded` and numeric exactness metric
3. Add optimizer entrypoints:
   - `SIMBA` (fast iteration)
   - `GEPA` (heavy final tuning)

**Exit criteria**

1. Quality gates block regressions.
2. Optimization artifacts improve benchmark metrics measurably.

## Non-Negotiables

1. Do not replace deterministic truth with model guesses.
2. Do not hide guard triggers from user output.
3. Do not let retrieval-only snippets masquerade as computed analytics.
4. Do not grow lexical hacks as primary logic.

## Progress Tracking

`[Phase 1: see canonical status board]`  
`[Phase 2: see canonical status board]`  
`[Phase 3: see canonical status board]`  
`[Phase 4: see canonical status board]`
