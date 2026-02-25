"""Level 1: Deterministic verification (no LLM, < 1 second).

Walks any extraction dict shape (flat strings, nested findings, mixed)
and checks grounding of scalar values against source text.
"""

from __future__ import annotations

import re
from typing import Any

from .models import FieldVerdict, Issue, VerificationReport


def verify_deterministic(
    extraction: dict[str, Any],
    source_text: str,
) -> VerificationReport:
    """Verify an extraction against source text using deterministic checks.

    Works with any extraction shape:
    - Flat dicts: {"findings": "string", "impression": "string"}
    - Nested dicts: {"findings": [{"measurement": {"value": 12}}]}
    - Mixed structures

    Checks:
    - Scalar grounding: are claimed string/numeric values found in source?
    - Numeric precision: are claimed numbers present in source text?
    - Reference validity: do list-index refs point to valid entries?
    """
    issues: list[Issue] = []
    field_verdicts: list[FieldVerdict] = []
    # Normalize whitespace so PDF \r\n line breaks don't break substring matching
    source_norm = " ".join(source_text.split())
    source_lower = source_norm.lower()

    # Walk the entire extraction tree
    scalars = _collect_scalars(extraction, prefix="")
    if not scalars:
        return VerificationReport(
            verdict="insufficient_evidence",
            confidence=0.0,
            level="deterministic",
            issues=[
                Issue(
                    type="empty_extraction",
                    field="*",
                    detail="Extraction contains no verifiable values",
                    severity="warning",
                )
            ],
        )

    checked = 0
    grounded = 0

    for path, value in scalars:
        if isinstance(value, (int, float)):
            value_str = (
                str(int(value)) if isinstance(value, float) and value == int(value)
                else str(value)
            )
            found = value_str in source_text
            checked += 1
            if found:
                grounded += 1
                field_verdicts.append(FieldVerdict(
                    status="verified",
                    field_path=path,
                    claimed_value=value_str,
                    severity="info",
                ))
            else:
                issues.append(Issue(
                    type="value_not_found",
                    field=path,
                    detail=f"Claimed numeric value '{value_str}' not found in source text",
                    severity="warning",
                ))
                field_verdicts.append(FieldVerdict(
                    status="unsupported",
                    field_path=path,
                    claimed_value=value_str,
                    severity="warning",
                ))

        elif isinstance(value, str):
            needle = " ".join(value.strip().split())  # normalize whitespace
            if not needle or len(needle) < 4:
                continue
            checked += 1
            # Try exact substring match first (whitespace-normalized)
            if needle.lower() in source_lower:
                grounded += 1
                idx = source_lower.find(needle.lower())
                start = max(0, idx - 40)
                end = min(len(source_norm), idx + len(needle) + 40)
                excerpt = source_norm[start:end]
                field_verdicts.append(FieldVerdict(
                    status="verified",
                    field_path=path,
                    claimed_value=_truncate(needle, 120),
                    evidence_excerpt=excerpt,
                    severity="info",
                ))
            else:
                # Fall back to word overlap for longer values
                overlap = _word_overlap(needle, source_norm)
                if overlap >= 0.6:
                    grounded += 1
                    field_verdicts.append(FieldVerdict(
                        status="verified",
                        field_path=path,
                        claimed_value=_truncate(needle, 120),
                        severity="info",
                    ))
                else:
                    severity = "warning" if overlap < 0.3 else "info"
                    if overlap < 0.3:
                        issues.append(Issue(
                            type="low_grounding",
                            field=path,
                            detail=(
                                f"Only {overlap:.0%} word overlap with source text"
                            ),
                            severity=severity,
                        ))
                    field_verdicts.append(FieldVerdict(
                        status="unsupported" if overlap < 0.3 else "not_checked",
                        field_path=path,
                        claimed_value=_truncate(needle, 120),
                        severity=severity,
                    ))

    # Check reference validity (findings[N].finding_refs → findings length)
    _check_references(extraction, issues)

    # Compute verdict
    if checked == 0:
        verdict = "insufficient_evidence"
        confidence = 0.0
    elif len(issues) == 0:
        verdict = "verified"
        confidence = min(0.95, grounded / max(checked, 1))
    elif any(i.severity == "critical" for i in issues):
        verdict = "contradicted"
        confidence = 0.2
    else:
        ratio = grounded / max(checked, 1)
        verdict = "verified" if ratio >= 0.7 else "partially_supported"
        confidence = max(0.1, ratio)

    return VerificationReport(
        verdict=verdict,
        confidence=round(confidence, 3),
        level="deterministic",
        issues=issues,
        field_verdicts=field_verdicts,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_scalars(
    obj: Any, *, prefix: str, out: list[tuple[str, Any]] | None = None
) -> list[tuple[str, Any]]:
    """Walk any nested structure, collecting (dotted_path, scalar_value) pairs."""
    if out is None:
        out = []
    if len(out) >= 500:
        return out

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                _collect_scalars(value, prefix=path, out=out)
            elif value is not None and not isinstance(value, bool):
                out.append((path, value))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            path = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                _collect_scalars(item, prefix=path, out=out)
            elif item is not None and not isinstance(item, bool):
                out.append((path, item))

    return out


def _word_overlap(claimed: str, source: str) -> float:
    """Compute word-level overlap ratio between claimed text and source."""
    claim_words = set(re.findall(r"[a-z0-9]+", claimed.lower()))
    source_words = set(re.findall(r"[a-z0-9]+", source.lower()))
    if not claim_words:
        return 0.0
    return len(claim_words & source_words) / len(claim_words)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _check_references(extraction: dict[str, Any], issues: list[Issue]) -> None:
    """Check cross-reference validity (e.g., finding_refs indices)."""
    findings = extraction.get("findings")
    if not isinstance(findings, list):
        return
    n_findings = len(findings)

    impressions = extraction.get("impressions")
    if not isinstance(impressions, list):
        return

    for j, imp in enumerate(impressions):
        if not isinstance(imp, dict):
            continue
        refs = imp.get("finding_refs", [])
        if not isinstance(refs, list):
            continue
        for ref_idx in refs:
            if isinstance(ref_idx, int) and ref_idx >= n_findings:
                issues.append(
                    Issue(
                        type="invalid_reference",
                        field=f"impressions[{j}].finding_refs",
                        detail=(
                            f"References finding[{ref_idx}] but only "
                            f"{n_findings} findings exist"
                        ),
                        severity="warning",
                    )
                )
