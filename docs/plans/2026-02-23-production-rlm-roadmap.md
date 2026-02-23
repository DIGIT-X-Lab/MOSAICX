# MOSAICX Production RLM Roadmap

**Date:** 2026-02-23  
**Status:** Proposed  
**Owner:** Core platform (query + verify)

## Goal

Make `mosaicx query` and `mosaicx verify` production-ready for:

1. long-document grounded QA
2. deterministic tabular truth
3. multi-turn reliability with low context drift

## Architecture Direction

Use an **agentic control plane** with a **deterministic data plane**.

1. Control plane: DSPy `RLM` for decomposition, tool orchestration, and synthesis.
2. Data plane: deterministic retrieval/execution/verification for truth.
3. Guard plane: fail-closed policies when required evidence is missing.

## What We Keep

1. DSPy + RLM as the orchestration layer.
2. Existing tabular compute tools (`compute_table_stat`, `run_table_sql`, `list_distinct_values`).
3. Verify layering (`quick`, `standard`, `thorough`) and source-grounded outputs.

## New Libraries: Decision

We **will** use new libraries selectively.

### Add Now (high value, low risk)

1. `rank-bm25` or `bm25s`
   - Purpose: stronger lexical retrieval over long document chunks.
2. `rapidfuzz`
   - Purpose: schema/value alias matching for robust column resolution.
3. `numpy`
   - Purpose: efficient scoring/reranking primitives in retrieval pipeline.

### Add Next (optional but likely)

1. `faiss-cpu` (or local vector DB alternative)
   - Purpose: semantic retrieval at scale for very long corpora.
2. `sentence-transformers` (or compatible local embeddings backend)
   - Purpose: embedding index for semantic recall.

### Keep Optional / Do Not Force Yet

1. LangGraph / external agent frameworks
   - Not required right now; DSPy is enough for the control plane if retrieval + guards are strengthened.

## Production Milestones

## M1: Retrieval and Grounding Foundation

1. Implement chunked document indexing with stable chunk IDs and source spans.
2. Add hybrid retrieval (BM25 + semantic optional) and reranking.
3. Require quote-span citations for long-doc answers.
4. Add text truth guard (fail closed when no supporting quote spans).

Exit criteria:
- long-doc QA no longer depends on a single lexical snippet path.
- answers include source-backed span evidence by default.

## M2: Program Decomposition + State

1. Split query into modules: intent -> planner -> retriever/executor -> verifier -> narrator.
2. Add conversation state graph (`entity`, `metric`, `timeframe`, `last_query_plan`).
3. Resolve co-reference/follow-ups from state, not prompt memory.
4. Add typed output validation + parse-retry/fix loop.

Exit criteria:
- follow-up queries remain stable without ad hoc prompting.
- adapter parse failures are rare and auto-recovered.

## M3: Reliability and Optimization

1. Build benchmark suites:
   - long-doc evidence QA
   - tabular analytics truth
   - multi-turn follow-up reliability
2. Add DSPy optimization/compile loop against these suites.
3. Add CI quality gates and telemetry for fallback/guard triggers.
4. Expose strict machine-readable output contract for developers.

Exit criteria:
- regression tests gate merges.
- query output is trustworthy for downstream RAG evaluation pipelines.

## Reliability Targets

1. Structured parse success >= 99%.
2. Deterministic tabular metric correctness = 100% on benchmark fixtures.
3. Long-doc answer grounding accuracy >= 95% on curated eval set.
4. Guard/fallback reasons always explicit in CLI + JSON output.

## Risks and Mitigations

1. Risk: slower responses after stronger retrieval/verification.
   - Mitigation: `fast` vs `thorough` execution levels.
2. Risk: semantic retrieval drift on domain jargon.
   - Mitigation: hybrid retrieval + medical synonym dictionaries.
3. Risk: excessive framework complexity.
   - Mitigation: keep DSPy as control-plane core; add only focused libs.

## Immediate Next Actions

1. Implement M1 chunking + BM25 retrieval + quote-span citations.
2. Add text truth guard equivalent to current tabular computed guard.
3. Add long-doc adversarial tests and wire them into CI.
