"""Level 2: LLM spot-check verification (single LLM call, 3-10 seconds)."""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from .models import FieldVerdict, Issue, VerificationReport
from .parse_utils import parse_json_like


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


def _normalize_status(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value in {"verified", "supported", "correct", "true", "match", "matches"}:
        return "verified"
    if value in {"mismatch", "contradicted", "incorrect", "wrong", "false"}:
        return "mismatch"
    if value in {"unsupported", "missing", "not_found", "not found", "unknown"}:
        return "unsupported"
    return "not_checked"


def _extract_claim_value(extraction: dict[str, Any], path: str) -> Any:
    current: Any = extraction
    if not path:
        return None
    tokens = [t for t in re.split(r"\.(?![^\[]*\])", path) if t]
    for token in tokens:
        while "[" in token and token.endswith("]"):
            name, rest = token.split("[", 1)
            if name:
                if not isinstance(current, dict):
                    return None
                current = current.get(name)
            idx_raw = rest[:-1]
            try:
                idx = int(idx_raw)
            except ValueError:
                return None
            if not isinstance(current, list) or idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            token = ""
        if token:
            if not isinstance(current, dict):
                return None
            current = current.get(token)
    return current


def _parse_verdict_payload(
    raw_verdicts: str,
    *,
    claim_values_by_path: dict[str, str] | None = None,
) -> tuple[list[Issue], list[FieldVerdict]]:
    claim_values_by_path = claim_values_by_path or {}
    parsed = parse_json_like(raw_verdicts or "")
    if parsed is None:
        return [], []
    if isinstance(parsed, dict):
        parsed = parsed.get("verdicts", [])
    if not isinstance(parsed, list):
        return [], []

    issues: list[Issue] = []
    verdicts: list[FieldVerdict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        field_path = str(item.get("field_path") or item.get("field") or "unknown")
        status = _normalize_status(str(item.get("status") or "not_checked"))
        claimed_value = item.get("claimed_value")
        if claimed_value is None:
            claimed_value = claim_values_by_path.get(field_path)
        source_value = item.get("source_value") or item.get("source")
        detail = str(item.get("detail") or item.get("reason") or "").strip()

        verdicts.append(
            FieldVerdict(
                status=status,
                field_path=field_path,
                claimed_value=None if claimed_value is None else str(claimed_value),
                source_value=None if source_value is None else str(source_value),
                evidence_excerpt=detail or None,
                severity="critical" if status == "mismatch" else "info",
            )
        )

        if status == "mismatch":
            issues.append(
                Issue(
                    type="spot_check_mismatch",
                    field=field_path,
                    detail=detail or f"Spot-check mismatch for {field_path}",
                    severity="critical",
                )
            )
        elif status == "unsupported":
            issues.append(
                Issue(
                    type="spot_check_unsupported",
                    field=field_path,
                    detail=detail or f"Spot-check unsupported for {field_path}",
                    severity="warning",
                )
            )
    return issues, verdicts


def run_spot_check(
    source_text: str,
    extraction: dict[str, Any],
    field_paths: list[str],
) -> tuple[list[Issue], list[FieldVerdict]]:
    """Run LLM spot-check on selected extraction fields."""
    if not field_paths:
        return [], []

    claims: list[dict[str, Any]] = []
    claim_values_by_path: dict[str, str] = {}
    for path in field_paths:
        claimed_value = _extract_claim_value(extraction, path)
        claims.append({"field_path": path, "claimed_value": claimed_value})
        claim_values_by_path[path] = "" if claimed_value is None else str(claimed_value)

    checker_cls = _get_dspy_class("SpotChecker")
    checker = checker_cls()
    result = checker.forward(source_text=source_text, claims=claims)
    raw_verdicts = str(getattr(result, "verdicts", "") or "")
    return _parse_verdict_payload(
        raw_verdicts,
        claim_values_by_path=claim_values_by_path,
    )


def verify_claim_with_llm(
    claim: str,
    source_text: str,
) -> VerificationReport:
    """Run LLM spot-check for a single free-text claim."""
    checker_cls = _get_dspy_class("SpotChecker")
    checker = checker_cls()
    claims = [{"field_path": "claim", "claimed_value": claim}]
    result = checker.forward(source_text=source_text, claims=claims)
    raw_verdicts = str(getattr(result, "verdicts", "") or "")
    issues, field_verdicts = _parse_verdict_payload(
        raw_verdicts,
        claim_values_by_path={"claim": claim},
    )

    statuses = {fv.status for fv in field_verdicts}
    if "mismatch" in statuses:
        verdict = "contradicted"
        confidence = 0.8
    elif "verified" in statuses:
        verdict = "verified"
        confidence = 0.85
    elif "unsupported" in statuses:
        verdict = "insufficient_evidence"
        confidence = 0.35
    else:
        verdict = "insufficient_evidence"
        confidence = 0.3

    return VerificationReport(
        verdict=verdict,
        confidence=confidence,
        level="spot_check",
        issues=issues,
        field_verdicts=field_verdicts,
    )


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
            claims_json = json.dumps(claims, default=str)
            result = self.verify(source_text=source_text, claims=claims_json)
            return result

    return {
        "VerifyClaim": VerifyClaim,
        "SpotChecker": SpotChecker,
    }


_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({"VerifyClaim", "SpotChecker"})


def _get_dspy_class(name: str):
    patched = sys.modules[__name__].__dict__.get(name)
    if patched is not None:
        return patched
    global _dspy_classes
    if _dspy_classes is None:
        _dspy_classes = _build_dspy_classes()
    return _dspy_classes[name]


def __getattr__(name: str):
    if name in _DSPY_CLASS_NAMES:
        return _get_dspy_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
