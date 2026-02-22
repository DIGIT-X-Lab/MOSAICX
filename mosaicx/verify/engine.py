"""Verification engine orchestrator -- unified verify() entry point."""

from __future__ import annotations

import re
from typing import Any

from .models import Issue, VerificationReport


def verify(
    *,
    extraction: dict[str, Any] | None = None,
    claim: str | None = None,
    source_text: str,
    level: str = "quick",
) -> VerificationReport:
    """Verify an extraction or claim against source text.

    Parameters
    ----------
    extraction:
        Structured extraction dict to verify. Used for structured verification.
    claim:
        A single claim string to verify. Used for claim-based verification.
        Exactly one of extraction or claim must be provided.
    source_text:
        The original document text to verify against.
    level:
        Verification level:
        - "quick" -- Level 1 deterministic only (no LLM, < 1s)
        - "standard" -- Level 1 + Level 2 LLM spot-check (3-10s)
        - "thorough" -- Level 1 + Level 2 + Level 3 RLM audit (30-90s)

    Returns
    -------
    VerificationReport
        Complete verification result.

    Raises
    ------
    ValueError
        If level is unknown or neither extraction nor claim is provided.
    """
    valid_levels = {"quick", "standard", "thorough"}
    if level not in valid_levels:
        raise ValueError(
            f"Unknown verification level {level!r}. "
            f"Choose from: {sorted(valid_levels)}"
        )

    if extraction is None and claim is None:
        raise ValueError("Must provide either extraction or claim")

    # Claim-based verification: simple text search
    if claim is not None:
        return _verify_claim(claim, source_text)

    # Extraction-based verification
    from .deterministic import verify_deterministic

    report = verify_deterministic(extraction, source_text)

    # Level 2 and 3 require LLM -- not implemented yet, return Level 1 result.
    # Future: if level in ("standard", "thorough"):
    #     report = _enhance_with_spot_check(report, extraction, source_text)
    # if level == "thorough":
    #     report = _enhance_with_audit(report, extraction, source_text)

    return report


def _verify_claim(claim: str, source_text: str) -> VerificationReport:
    """Simple claim verification by checking if the claim text appears in source."""
    claim_lower = claim.lower()
    source_lower = source_text.lower()

    # Find numbers in claim
    numbers = re.findall(r"\d+(?:\.\d+)?", claim)
    terms_found = all(n in source_text for n in numbers)

    # Tokenize by extracting alphanumeric sequences for robust comparison
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
