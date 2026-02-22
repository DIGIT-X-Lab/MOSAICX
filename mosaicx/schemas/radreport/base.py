# mosaicx/schemas/radreport/base.py
"""Base Pydantic models for structured radiology reports.

These models capture the core data structures used to represent parsed
radiology reports: measurements, findings, impression items, and report
sections.  They are intentionally free of any DSPy dependency so they can
be imported and used independently for validation, serialization, or FHIR
export.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


__all__ = [
    "CaseInsensitiveModel",
    "Measurement",
    "ChangeType",
    "RadReportFinding",
    "ImpressionItem",
    "ReportSections",
]


class CaseInsensitiveModel(BaseModel):
    """Base model that accepts keys in any case (e.g., 'Findings' -> 'findings').

    LLMs sometimes return title-cased or upper-cased JSON keys. This validator
    normalises incoming dict keys to lowercase before Pydantic field matching.
    """

    @model_validator(mode="before")
    @classmethod
    def _lowercase_keys(cls, data: dict) -> dict:  # type: ignore[override]
        if isinstance(data, dict):
            return {k.lower(): v for k, v in data.items()}
        return data


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

class Measurement(CaseInsensitiveModel):
    """A single quantitative measurement (e.g., lesion diameter)."""

    value: float = Field(..., description="Numeric measurement value")
    unit: str = Field("mm", description="Unit of measurement (e.g., mm, cm)")
    dimension: str = Field(
        "", description="What is being measured (e.g., diameter, volume)"
    )
    prior_value: Optional[float] = Field(
        None, description="Value from the prior comparison study"
    )


# ---------------------------------------------------------------------------
# ChangeType
# ---------------------------------------------------------------------------

class ChangeType(CaseInsensitiveModel):
    """Describes change relative to a prior study."""

    status: Literal["new", "stable", "increased", "decreased", "resolved"] = Field(
        ..., description="Type of change from prior"
    )
    prior_date: Optional[str] = Field(
        None, description="Date of the prior comparison study (ISO 8601)"
    )
    prior_measurement: Optional[Measurement] = Field(
        None, description="Measurement from the prior study"
    )


# ---------------------------------------------------------------------------
# RadReportFinding
# ---------------------------------------------------------------------------

class RadReportFinding(CaseInsensitiveModel):
    """A single radiology finding extracted from a report."""

    anatomy: str = Field(..., description="Anatomical location of the finding")
    radlex_id: Optional[str] = Field(
        None, description="RadLex ontology identifier (e.g., RID1301)"
    )
    observation: str = Field(
        "", description="Short observation label (e.g., nodule, mass)"
    )
    description: str = Field(
        "", description="Full-text description of the finding"
    )
    measurement: Optional[Measurement] = Field(
        None, description="Quantitative measurement, if available"
    )
    change_from_prior: Optional[ChangeType] = Field(
        None, description="Change compared to a prior study"
    )
    severity: Optional[str] = Field(
        None, description="Severity or grade (e.g., mild, moderate, severe)"
    )
    template_field_id: Optional[str] = Field(
        None,
        description="RadReport template field ID this finding maps to",
    )


# ---------------------------------------------------------------------------
# ImpressionItem
# ---------------------------------------------------------------------------

class ImpressionItem(CaseInsensitiveModel):
    """A single item from the report's impression / conclusion section."""

    statement: str = Field(
        ..., description="Impression statement text"
    )
    category: Optional[str] = Field(
        None,
        description="Classification category (e.g., Lung-RADS 3, BI-RADS 4)",
    )
    icd10_code: Optional[str] = Field(
        None, description="ICD-10 code associated with this impression"
    )
    actionable: bool = Field(
        False,
        description="Whether this impression requires follow-up action",
    )
    finding_refs: List[int] = Field(
        default_factory=list,
        description="Indices into the findings list that support this impression",
    )


# ---------------------------------------------------------------------------
# ReportSections
# ---------------------------------------------------------------------------

class ReportSections(CaseInsensitiveModel):
    """Top-level sections parsed from a radiology report."""

    indication: str = Field(
        "", description="Clinical indication / reason for exam"
    )
    comparison: str = Field(
        "", description="Prior comparison studies referenced"
    )
    technique: str = Field(
        "", description="Imaging technique / protocol description"
    )
    findings: str = Field(
        "", description="Raw findings section text"
    )
    impression: str = Field(
        "", description="Raw impression / conclusion section text"
    )
