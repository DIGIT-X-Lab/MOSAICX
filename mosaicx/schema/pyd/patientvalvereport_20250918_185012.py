from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class PatientValveReport(BaseModel):
    """Schema for extracting patient demographics, scan information, and valve condition details."""
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    age: int = Field(..., description="Patient age in years.")
    sex: Literal['Male', 'Female', 'Other'] = Field(..., description="Patient sex.")
    modality_name: str = Field(..., description="Name of the imaging modality (e.g., Echocardiography, MRI, CT).")
    scan_date: str = Field(..., description="Date and time of the scan.")
    mitral_valve_issue: bool = Field(..., description="Whether mitral valve issues are present (true = yes, false = no).")
    mitral_valve_grade: Optional[Literal['Normal', 'Mild', 'Moderate', 'Severe']] = Field(None, description="Severity grade of the mitral valve issue.")
    mitral_valve_stenosis: Optional[bool] = Field(None, description="Presence of mitral valve stenosis (true = yes, false = no).")
    tricuspid_valve_issue: bool = Field(..., description="Whether tricuspid valve issues are present (true = yes, false = no).")
    tricuspid_valve_grade: Optional[Literal['Normal', 'Mild', 'Moderate', 'Severe']] = Field(None, description="Severity grade of the tricuspid valve issue.")
    tricuspid_valve_stenosis: Optional[bool] = Field(None, description="Presence of tricuspid valve stenosis (true = yes, false = no).")
