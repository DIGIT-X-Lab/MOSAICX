from pydantic import BaseModel, Field
from typing import Optional, Literal


class GeneratedModel(BaseModel):
    """
    Model representing extracted basic patient information and vital signs.
    """

    patient_name: Optional[str] = Field(
        None,
        description="Full name of the patient."
    )
    patient_age: Optional[int] = Field(
        None,
        ge=0,
        description="Age of the patient in years."
    )
    patient_gender: Optional[Literal["Male", "Female", "Other", "Unknown"]] = Field(
        None,
        description="Gender of the patient."
    )
    systolic_bp: int = Field(
        ...,
        gt=0,
        description="Systolic blood pressure in mmHg."
    )
    diastolic_bp: int = Field(
        ...,
        gt=0,
        description="Diastolic blood pressure in mmHg."
    )
    heart_rate: int = Field(
        ...,
        gt=0,
        description="Heart rate in beats per minute."
    )
    temperature_c: float = Field(
        ...,
        ge=30.0,
        le=45.0,
        description="Body temperature in degrees Celsius."
    )