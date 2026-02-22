"""Level 2: LLM spot-check verification (single LLM call, 3-10 seconds)."""

from __future__ import annotations

from typing import Any, List


# High-risk field patterns that should be spot-checked
_HIGH_RISK_KEYS = {"measurement", "value", "severity", "grade", "stage", "margin", "diagnosis"}


def select_high_risk_fields(extraction: dict[str, Any]) -> list[str]:
    """Select fields from an extraction that are high-risk for errors.

    High-risk fields include measurements, severity ratings, staging,
    margins, and diagnoses -- fields where errors have clinical impact.

    Returns a list of dotted field paths (e.g., "findings[0].measurement").
    """
    paths: list[str] = []

    def _walk(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else k
                if k in _HIGH_RISK_KEYS and v is not None:
                    paths.append(path)
                _walk(v, path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{prefix}[{i}]")

    _walk(extraction, "")
    return paths


# ---------------------------------------------------------------------------
# DSPy Signatures & Module (lazy)
# ---------------------------------------------------------------------------

def _build_dspy_classes():
    import dspy

    class VerifyClaim(dspy.Signature):
        """Verify extracted claims against source text."""

        source_text: str = dspy.InputField(desc="Original document text")
        claims: str = dspy.InputField(desc="JSON list of claims to verify (field_path + claimed_value pairs)")
        verdicts: str = dspy.OutputField(desc="JSON list of verdicts: for each claim, {field_path, status, source_value, detail}")

    class SpotChecker(dspy.Module):
        """DSPy Module for LLM-based spot-check verification."""

        def __init__(self) -> None:
            super().__init__()
            self.verify = dspy.ChainOfThought(VerifyClaim)

        def forward(self, source_text: str, claims: list[dict[str, Any]]) -> dspy.Prediction:
            import json
            claims_json = json.dumps(claims, default=str)
            result = self.verify(source_text=source_text, claims=claims_json)
            return result

    return {
        "VerifyClaim": VerifyClaim,
        "SpotChecker": SpotChecker,
    }


_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({"VerifyClaim", "SpotChecker"})


def __getattr__(name: str):
    global _dspy_classes
    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
