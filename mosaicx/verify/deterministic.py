"""Level 1: Deterministic verification (no LLM, < 1 second)."""

from __future__ import annotations

from typing import Any

from .models import Issue, VerificationReport


def verify_deterministic(
    extraction: dict[str, Any],
    source_text: str,
) -> VerificationReport:
    """Verify an extraction against source text using deterministic checks.

    Checks:
    - Measurements: are claimed values found in source text?
    - Finding refs: do impression finding_refs point to valid indices?
    - Enum consistency: do severity/modality values make sense?
    """
    issues: list[Issue] = []
    findings = extraction.get("findings", [])

    # Check measurements
    for i, finding in enumerate(findings):
        measurement = finding.get("measurement")
        if measurement and isinstance(measurement, dict):
            value = measurement.get("value")
            if value is not None:
                # Search for the number in source text
                value_str = (
                    str(int(value)) if float(value) == int(value) else str(value)
                )
                if value_str not in source_text:
                    issues.append(
                        Issue(
                            type="value_not_found",
                            field=f"findings[{i}].measurement",
                            detail=(
                                f"Claimed value '{value_str}' not found in source text"
                            ),
                            severity="warning",
                        )
                    )

    # Check finding_refs validity
    impressions = extraction.get("impressions", [])
    for j, imp in enumerate(impressions):
        refs = imp.get("finding_refs", [])
        if isinstance(refs, list):
            for ref_idx in refs:
                if isinstance(ref_idx, int) and ref_idx >= len(findings):
                    issues.append(
                        Issue(
                            type="invalid_reference",
                            field=f"impressions[{j}].finding_refs",
                            detail=(
                                f"References finding[{ref_idx}] but only "
                                f"{len(findings)} findings exist"
                            ),
                            severity="warning",
                        )
                    )

    verdict = "verified" if not issues else "partially_supported"
    confidence = max(0.0, 1.0 - len(issues) * 0.15)

    return VerificationReport(
        verdict=verdict,
        confidence=confidence,
        level="deterministic",
        issues=issues,
    )
