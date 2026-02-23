"""Query optimization module for grounded QA over source text.

This module exposes query behavior as a DSPy-optimizable pipeline so
`mosaicx optimize --pipeline query` can tune answer narration while
keeping truth retrieval grounded in the query engine.
"""

from __future__ import annotations


def _build_dspy_classes():
    import dspy

    class NarrateGroundedAnswer(dspy.Signature):
        """Produce a concise grounded answer from question + evidence."""

        question: str = dspy.InputField(desc="User question")
        draft_response: str = dspy.InputField(desc="Draft answer from query engine")
        evidence: str = dspy.InputField(desc="Grounded evidence snippets")
        response: str = dspy.OutputField(
            desc="Final grounded answer. Be concise and do not invent facts."
        )

    class QueryGroundedResponder(dspy.Module):
        """DSPy module wrapper around QueryEngine for optimization/evaluation."""

        def __init__(self, max_iterations: int = 20, verbose: bool = False) -> None:
            super().__init__()
            self.max_iterations = max_iterations
            self.verbose = verbose
            self.narrate = dspy.ChainOfThought(NarrateGroundedAnswer)

        def forward(
            self,
            question: str,
            source_text: str,
            source_name: str = "source.txt",
        ) -> dspy.Prediction:
            from mosaicx.query.engine import QueryEngine
            from mosaicx.query.session import QuerySession

            session = QuerySession()
            session.add_text_source(source_name, source_text)
            engine = QueryEngine(
                session=session,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
            )

            payload = engine.ask_structured(question, top_k_citations=4)
            draft = str(payload.get("answer") or "").strip()
            citations = payload.get("citations") or []
            evidence_lines = []
            for c in citations[:8]:
                source = str(c.get("source") or "unknown")
                snippet = " ".join(str(c.get("snippet") or "").split())
                if snippet:
                    evidence_lines.append(f"- {source}: {snippet[:280]}")
            evidence = "\n".join(evidence_lines) if evidence_lines else "(no citations)"

            response = draft
            try:
                nar = self.narrate(
                    question=question,
                    draft_response=draft,
                    evidence=evidence,
                )
                candidate = str(getattr(nar, "response", "") or "").strip()
                if candidate:
                    response = candidate
            except Exception:
                # Keep deterministic payload result when narration refinement fails.
                response = draft

            context = "\n".join(
                " ".join(str(c.get("snippet") or "").split()) for c in citations[:10]
            )
            return dspy.Prediction(
                response=response,
                context=context,
                citations=citations,
                confidence=float(payload.get("confidence") or 0.0),
                route_intent=str(payload.get("route_intent") or ""),
            )

    return {
        "NarrateGroundedAnswer": NarrateGroundedAnswer,
        "QueryGroundedResponder": QueryGroundedResponder,
    }


_dspy_classes: dict[str, type] | None = None
_DSPY_CLASS_NAMES = frozenset({"NarrateGroundedAnswer", "QueryGroundedResponder"})


def __getattr__(name: str):
    global _dspy_classes

    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
