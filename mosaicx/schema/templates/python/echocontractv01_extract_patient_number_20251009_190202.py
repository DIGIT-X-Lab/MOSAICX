from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Literal


class EchoContractV01(BaseModel):
    """
    EchoContractV01 captures extracted echocardiogram report data.
    """

    patient_number: str = Field(..., description="Unique identifier for the patient")
    age: int = Field(..., ge=0, description="Patient age in years")
    sex: Literal["Male", "Female", "Other"] = Field(..., description="Patient biological sex")
    dob: str = Field(..., description="Date of birth (ISO format)")
    date_of_scan: str = Field(..., description="Date when the scan was performed (ISO format)")
    modality: str = Field(..., description="Imaging modality used")

    tricuspid_valve_insufficiency: Literal["True", "False"] = Field(...,
        description="Presence of tricuspid valve insufficiency")
    tricuspid_valve_grade: Literal["None", "Mild", "Medium", "Severe"] = Field(...,
        description="Severity grade of tricuspid valve insufficiency. Note: Physiological insufficiency is classified as 'none'")
    tricuspid_valve_confidence: float = Field(..., ge=0.0, le=1.0,
        description="Confidence score for the tricuspid valve assessment")
    tricuspid_valve_text: str = Field(...,
        description="Extracted text supporting the tricuspid valve conclusion")

    mitral_valve_insufficiency: Literal["True", "False"] = Field(...,
        description="Presence of mitral valve insufficiency")
    mitral_valve_grade: Literal["None", "Mild", "Medium", "Severe"] = Field(...,
        description="Severity grade of mitral valve insufficiency. Note: Physiological insufficiency is classified as 'none'")
    mitral_valve_confidence: float = Field(..., ge=0.0, le=1.0,
        description="Confidence score for the mitral valve assessment")
    mitral_valve_text: str = Field(...,
        description="Extracted text supporting the mitral valve conclusion")