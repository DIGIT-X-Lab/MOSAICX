from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class PatientScanInfo(BaseModel):
    """Schema for extracted patient and scan metadata"""
    patient_name: str = Field(..., description="Full name of the patient")
    age: int = Field(..., description="Age of the patient in years [units=years]")
    sex: str = Field(..., description="Sex of the patient (e.g., male, female, other)")
    date_of_scan: str = Field(..., description="Date when the scan was performed")
    modality: str = Field(..., description="Imaging modality used for the scan (e.g., MRI, CT). Not strictly validated.")
