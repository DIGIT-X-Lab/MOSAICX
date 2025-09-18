from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class PatientCardiacValveAssessment(BaseModel):
    """Schema for recording patient demographics, cardiac valve assessment results, and associated scan information."""
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    name: str = Field(None, description="Full name of the patient.")
    date_of_birth: str = Field(None, description="Patient's date of birth.")
    age: int = Field(None, description="Patient's age in years at the time of assessment. [units=years]")
    sex: Literal['Male', 'Female', 'Other'] = Field(None, description="Biological sex of the patient.")
    report_id: str = Field(..., description="Unique identifier for this valve assessment report.")
    assessment_date: str = Field(..., description="Date when the valve assessment was performed.")
    valve_type: Literal['Aortic', 'Mitral', 'Tricuspid', 'Pulmonary'] = Field(..., description="Cardiac valve being assessed.")
    imaging_modality: Literal['Echocardiogram', 'MRI', 'CT', 'Ultrasound'] = Field(..., description="Imaging technique used for the assessment.")
    measurements: Dict[str, Any] = Field(..., description="Quantitative and qualitative measurements from the valve assessment.")
    scan_info: Dict[str, Any] = Field(None, description="Details about the imaging acquisition.")
    physician: str = Field(None, description="Name or identifier of the interpreting physician.")
