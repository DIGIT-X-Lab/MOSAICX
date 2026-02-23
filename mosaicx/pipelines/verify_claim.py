"""Verification optimization module for grounded claim adjudication.

This module exposes verify behavior as a DSPy-optimizable pipeline so
`mosaicx optimize --pipeline verify` can tune prompt/program behavior
while preserving the existing verification engine contract.
"""

from __future__ import annotations


def _build_dspy_classes():
    import dspy

    class VerifyClaimResponder(dspy.Module):
        """DSPy module wrapper around ``mosaicx.verify.engine.verify``."""

        def __init__(self, default_level: str = "thorough") -> None:
            super().__init__()
            self.default_level = default_level

        def forward(
            self,
            claim: str,
            source_text: str,
            level: str = "",
        ) -> dspy.Prediction:
            from mosaicx.verify.engine import verify as verify_claim

            effective_level = (level or self.default_level or "thorough").strip().lower()
            report = verify_claim(
                claim=claim,
                source_text=source_text,
                level=effective_level,
            )

            evidence_lines: list[str] = []
            citations: list[dict[str, str]] = []

            for ev in report.evidence[:8]:
                excerpt = " ".join(str(ev.excerpt or "").split())
                source = str(ev.source or "source")
                if excerpt:
                    evidence_lines.append(f"- {source}: {excerpt[:280]}")
                    citations.append({"source": source, "snippet": excerpt})

            if not evidence_lines:
                for fv in report.field_verdicts[:8]:
                    excerpt = " ".join(str(fv.evidence_excerpt or "").split())
                    source_value = " ".join(str(fv.source_value or "").split())
                    label = str(fv.field_path or "field")
                    if source_value:
                        snippet = f"{label}: source={source_value}"
                    elif excerpt:
                        snippet = f"{label}: {excerpt[:220]}"
                    else:
                        continue
                    evidence_lines.append(f"- {snippet}")
                    citations.append({"source": "verification", "snippet": snippet})

            context = "\n".join(evidence_lines)
            return dspy.Prediction(
                response=report.verdict,
                verdict=report.verdict,
                confidence=float(report.confidence),
                level=report.level,
                context=context,
                citations=citations,
                issues=[issue.model_dump() for issue in report.issues],
                field_verdicts=[v.model_dump() for v in report.field_verdicts],
            )

    return {"VerifyClaimResponder": VerifyClaimResponder}


_dspy_classes: dict[str, type] | None = None
_DSPY_CLASS_NAMES = frozenset({"VerifyClaimResponder"})


def __getattr__(name: str):
    global _dspy_classes

    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
