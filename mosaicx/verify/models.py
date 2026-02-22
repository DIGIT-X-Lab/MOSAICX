"""Data models for the verification engine."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Issue(BaseModel):
    """A specific verification issue found."""

    type: str = Field(
        description="Issue type: value_mismatch, hallucination, omission, "
        "invalid_reference, value_not_found"
    )
    field: str = Field(description="Field path where issue was found")
    detail: str = Field(description="Human-readable description")
    severity: Literal["info", "warning", "critical"] = "warning"


class FieldVerdict(BaseModel):
    """Verification verdict for a single field."""

    status: Literal["verified", "mismatch", "unsupported", "not_checked"] = (
        "not_checked"
    )
    claimed_value: str | None = None
    source_value: str | None = None
    evidence_excerpt: str | None = None
    severity: Literal["info", "warning", "critical"] = "info"


class VerificationReport(BaseModel):
    """Complete verification result."""

    verdict: Literal[
        "verified", "partially_supported", "contradicted", "insufficient_evidence"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    level: str = Field(
        description="Verification level: deterministic, spot_check, audit"
    )
    issues: list[Issue] = Field(default_factory=list)
    field_verdicts: list[FieldVerdict] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialization."""
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "level": self.level,
            "issues": [i.model_dump() for i in self.issues],
            "field_verdicts": [fv.model_dump() for fv in self.field_verdicts],
        }
