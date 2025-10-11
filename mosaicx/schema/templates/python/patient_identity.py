from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PatientIdentity(BaseModel):
    """Basic patient identity details used in MOSAICX demos and smoke tests."""

    patient_id: Optional[str] = Field(
        default=None,
        description="Medical record number or pseudonym for the patient.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Full patient name as documented in the report.",
        min_length=1,
    )
    date_of_birth: Optional[str] = Field(
        default=None,
        description="Date of birth in ISO format (YYYY-MM-DD).",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    sex: Optional[str] = Field(
        default=None,
        description="Recorded sex or gender (e.g., Male, Female, Other).",
        max_length=16,
    )
