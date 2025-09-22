from datetime import date
from typing import Optional, Literal

from pydantic import BaseModel, Field, validator


class EchocardiographyContract(BaseModel):
    """Data contract for an echocardiography study."""

    patient_name: str = Field(..., description="Full name of the patient")
    patient_number: str = Field(..., description="Unique identifier for the patient")
    date_of_study: date = Field(..., description="Date when the study was performed")
    birth_date: date = Field(..., description="Patient's date of birth")
    modality: str = Field(..., description="Imaging modality used (e.g., 'ECHO')")

    mitral_valve_insufficiency: Literal["yes", "no"] = Field(
        ..., description="Whether mitral valve insufficiency is present. Important: Physiological insufficiency is also considered None."
    )
    mitral_valve_grade: Optional[Literal["Unavailable", "None","Mild", "Moderate", "Severe"]] = Field(
        None,
        description="Severity grade of mitral valve insufficiency if present. Important Physiological insufficiency is also considered None.",
    )
    mitral_valve_stenosis: Literal["yes", "no"] = Field(
        ..., description="Whether mitral valve stenosis is present"
    )

    tricuspid_valve_insufficiency: Literal["yes", "no"] = Field(
        ..., description="Whether tricuspid valve insufficiency is present. Important: Physiological insufficiency is also considered None."
    )
    tricuspid_valve_grade: Optional[Literal["Unavailable", "None","Mild", "Moderate", "Severe"]] = Field(
        None,
        description="Severity grade of tricuspid valve insufficiency if present. Important: Physiological insufficiency is also considered None.",
    )
    tricuspid_valve_stenosis: Literal["yes", "no"] = Field(
        ..., description="Whether tricuspid valve stenosis is present"
    )

    @validator("mitral_valve_grade")
    def check_mitral_grade(cls, v, values):
        if values.get("mitral_valve_insufficiency") == "yes" and v is None:
            raise ValueError("Mitral valve grade must be provided when insufficiency is 'yes'")
        if values.get("mitral_valve_insufficiency") == "no" and v is not None:
            raise ValueError("Mitral valve grade must be omitted when insufficiency is 'no'")
        return v

    @validator("tricuspid_valve_grade")
    def check_tricuspid_grade(cls, v, values):
        if values.get("tricuspid_valve_insufficiency") == "yes" and v is None:
            raise ValueError("Tricuspid valve grade must be provided when insufficiency is 'yes'")
        if values.get("tricuspid_valve_insufficiency") == "no" and v is not None:
            raise ValueError("Tricuspid valve grade must be omitted when insufficiency is 'no'")
        return v
