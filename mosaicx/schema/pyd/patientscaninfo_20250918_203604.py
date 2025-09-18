from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class PatientScanInfo(BaseModel):
    """Schema for basic patient demographics and imaging scan metadata."""
    patient_name: str = Field(..., description="Full name of the patient.")
    age: int = Field(..., description="Age of the patient in years. [units=years]")
    sex: Literal['Male', 'Female', 'Other'] = Field(..., description="Biological sex of the patient.")
    date_of_birth: str = Field(..., description="Patient's date of birth.")
    scan_date: str = Field(..., description="Date when the imaging scan was performed.")
    modality: Literal['CT', 'MRI', 'X-ray', 'Ultrasound', 'PET', 'SPECT'] = Field(..., description="Imaging modality used for the scan.")
