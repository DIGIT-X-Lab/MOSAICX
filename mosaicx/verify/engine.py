"""Verification engine orchestrator -- unified verify() entry point."""

from __future__ import annotations

import re
from typing import Any

from .models import FieldVerdict, Issue, VerificationReport


def verify(
    *,
    extraction: dict[str, Any] | None = None,
    claim: str | None = None,
    source_text: str,
    level: str = "quick",
) -> VerificationReport:
    """Verify an extraction or claim against source text."""
    valid_levels = {"quick", "standard", "thorough"}
    if level not in valid_levels:
        raise ValueError(
            f"Unknown verification level {level!r}. "
            f"Choose from: {sorted(valid_levels)}"
        )

    if extraction is None and claim is None:
        raise ValueError("Must provide either extraction or claim")

    if claim is not None:
        if level == "quick":
            return _verify_claim(claim, source_text)
        if level == "standard":
            return _verify_claim_with_spot_check(claim, source_text)
        return _verify_claim_with_audit(claim, source_text)

    from .deterministic import verify_deterministic

    report = verify_deterministic(extraction, source_text)
    if level == "quick":
        return report
    if level == "standard":
        return _enhance_with_spot_check(report, extraction, source_text)
    return _enhance_with_audit(report, extraction, source_text)


def _merge_extraction_report(
    base_report: VerificationReport,
    *,
    issues: list[Issue],
    field_verdicts: list[FieldVerdict],
    level: str,
) -> VerificationReport:
    merged_issues = [*base_report.issues, *issues]
    merged_fv = [*base_report.field_verdicts, *field_verdicts]
    statuses = {fv.status for fv in merged_fv}

    if "mismatch" in statuses:
        verdict = "contradicted"
        confidence = min(base_report.confidence, 0.2)
    elif "unsupported" in statuses:
        verdict = "partially_supported"
        confidence = min(base_report.confidence, 0.6)
    elif "verified" in statuses:
        verdict = "verified"
        confidence = max(base_report.confidence, 0.85)
    else:
        verdict = base_report.verdict
        confidence = base_report.confidence

    return VerificationReport(
        verdict=verdict,
        confidence=max(0.0, min(1.0, confidence)),
        level=level,
        issues=merged_issues,
        field_verdicts=merged_fv,
        evidence=list(base_report.evidence),
        missed_content=list(base_report.missed_content),
    )


def _build_claim_report(
    *,
    issues: list[Issue],
    field_verdicts: list[FieldVerdict],
    level: str,
) -> VerificationReport:
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
        critical = any(i.severity == "critical" for i in issues)
        verdict = "contradicted" if critical else "insufficient_evidence"
        confidence = 0.25 if critical else 0.3

    return VerificationReport(
        verdict=verdict,
        confidence=confidence,
        level=level,
        issues=issues,
        field_verdicts=field_verdicts,
    )


def _enhance_with_spot_check(
    base_report: VerificationReport,
    extraction: dict[str, Any],
    source_text: str,
) -> VerificationReport:
    from .spot_check import run_spot_check, select_high_risk_fields

    try:
        high_risk_paths = select_high_risk_fields(extraction)
        issues, field_verdicts = run_spot_check(source_text, extraction, high_risk_paths)
        return _merge_extraction_report(
            base_report,
            issues=issues,
            field_verdicts=field_verdicts,
            level="spot_check",
        )
    except Exception as exc:
        return VerificationReport(
            verdict=base_report.verdict,
            confidence=base_report.confidence,
            level="deterministic",
            issues=[
                *base_report.issues,
                Issue(
                    type="llm_unavailable",
                    field="verify.spot_check",
                    detail=f"LLM spot-check unavailable: {exc}",
                    severity="warning",
                ),
            ],
            field_verdicts=list(base_report.field_verdicts),
            evidence=list(base_report.evidence),
            missed_content=list(base_report.missed_content),
        )


def _enhance_with_audit(
    base_report: VerificationReport,
    extraction: dict[str, Any],
    source_text: str,
) -> VerificationReport:
    from .audit import run_audit

    try:
        issues, field_verdicts = run_audit(source_text, extraction)
        return _merge_extraction_report(
            base_report,
            issues=issues,
            field_verdicts=field_verdicts,
            level="audit",
        )
    except Exception as exc:
        fallback = _enhance_with_spot_check(base_report, extraction, source_text)
        fallback.issues.append(
            Issue(
                type="rlm_unavailable",
                field="verify.audit",
                detail=f"RLM audit unavailable: {exc}",
                severity="warning",
            )
        )
        return fallback


def _verify_claim_with_spot_check(claim: str, source_text: str) -> VerificationReport:
    from .spot_check import verify_claim_with_llm

    try:
        return verify_claim_with_llm(claim, source_text)
    except Exception as exc:
        report = _verify_claim(claim, source_text)
        report.issues.append(
            Issue(
                type="llm_unavailable",
                field="claim",
                detail=f"LLM spot-check unavailable: {exc}",
                severity="warning",
            )
        )
        return report


def _verify_claim_with_audit(claim: str, source_text: str) -> VerificationReport:
    from .audit import run_claim_audit

    try:
        issues, field_verdicts = run_claim_audit(claim, source_text)
        return _build_claim_report(
            issues=issues,
            field_verdicts=field_verdicts,
            level="audit",
        )
    except Exception as exc:
        report = _verify_claim_with_spot_check(claim, source_text)
        report.issues.append(
            Issue(
                type="rlm_unavailable",
                field="claim",
                detail=f"RLM audit unavailable: {exc}",
                severity="warning",
            )
        )
        return report


def _verify_claim(claim: str, source_text: str) -> VerificationReport:
    """Deterministic claim verification by textual overlap and numeric checks."""
    claim_lower = claim.lower()
    source_lower = source_text.lower()

    numbers = re.findall(r"\d+(?:\.\d+)?", claim)
    terms_found = all(n in source_text for n in numbers)

    claim_words = set(re.findall(r"[a-z0-9]+", claim_lower))
    source_words = set(re.findall(r"[a-z0-9]+", source_lower))
    overlap = claim_words & source_words
    word_overlap_ratio = len(overlap) / max(len(claim_words), 1)

    if terms_found and word_overlap_ratio > 0.5:
        return VerificationReport(
            verdict="verified",
            confidence=min(0.95, word_overlap_ratio),
            level="deterministic",
            issues=[],
        )

    issues: list[Issue] = []
    if not terms_found:
        missing = [n for n in numbers if n not in source_text]
        issues.append(
            Issue(
                type="value_not_found",
                field="claim",
                detail=f"Numeric values {missing} not found in source",
                severity="warning",
            )
        )

    verdict = (
        "partially_supported" if word_overlap_ratio > 0.3 else "insufficient_evidence"
    )

    return VerificationReport(
        verdict=verdict,
        confidence=max(0.0, word_overlap_ratio),
        level="deterministic",
        issues=issues,
    )
