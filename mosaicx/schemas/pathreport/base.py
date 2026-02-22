"""Base Pydantic models for structured pathology reports.

These models capture the core data structures used to represent parsed
pathology reports: sections, findings, biomarkers, and diagnoses.
They are intentionally free of any DSPy dependency so they can be
imported and used independently for validation, serialization, or export.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from mosaicx.schemas.radreport.base import CaseInsensitiveModel

__all__ = [
    "PathSections",
    "PathFinding",
    "Biomarker",
    "PathDiagnosis",
]


class PathSections(CaseInsensitiveModel):
    """Top-level sections parsed from a pathology report."""

    clinical_history: str = Field(
        "", description="Clinical history / reason for biopsy"
    )
    gross_description: str = Field(
        "", description="Gross / macroscopic description of the specimen"
    )
    microscopic: str = Field(
        "", description="Microscopic description"
    )
    diagnosis: str = Field(
        "", description="Final diagnosis section text"
    )
    ancillary_studies: str = Field(
        "", description="Ancillary studies (IHC, molecular, flow cytometry)"
    )
    comment: str = Field(
        "", description="Additional comments or notes"
    )


class PathFinding(CaseInsensitiveModel):
    """A single microscopic finding from a pathology report."""

    description: str = Field(
        ..., description="Description of the finding"
    )
    histologic_type: Optional[str] = Field(
        None, description="Histologic type (e.g., adenocarcinoma, squamous cell)"
    )
    grade: Optional[str] = Field(
        None, description="Histologic grade (e.g., G2, Gleason 3+4=7, Nottingham 2)"
    )
    margins: Optional[str] = Field(
        None, description="Margin status (positive/negative/close + distance)"
    )
    lymphovascular_invasion: Optional[str] = Field(
        None, description="Lymphovascular invasion (present/absent/indeterminate)"
    )
    perineural_invasion: Optional[str] = Field(
        None, description="Perineural invasion (present/absent/indeterminate)"
    )


class Biomarker(CaseInsensitiveModel):
    """A single biomarker result (IHC, molecular, etc.)."""

    name: str = Field(
        ..., description="Biomarker name (e.g., ER, PR, HER2, Ki-67, EGFR)"
    )
    result: str = Field(
        ..., description="Result (e.g., positive (95%), negative, 3+)"
    )
    method: Optional[str] = Field(
        None, description="Method used (e.g., IHC, FISH, PCR)"
    )


class PathDiagnosis(CaseInsensitiveModel):
    """A final pathologic diagnosis with staging and biomarker data."""

    diagnosis: str = Field(
        ..., description="Primary diagnosis text"
    )
    who_classification: Optional[str] = Field(
        None, description="WHO tumor classification"
    )
    tnm_stage: Optional[str] = Field(
        None, description="Pathologic TNM stage (e.g., pT2 pN1a)"
    )
    icd_o_morphology: Optional[str] = Field(
        None, description="ICD-O morphology code (e.g., 8500/3)"
    )
    biomarkers: List[Biomarker] = Field(
        default_factory=list,
        description="Biomarker results (IHC, molecular)"
    )
    ancillary_results: Optional[str] = Field(
        None, description="Summary of ancillary study results"
    )
