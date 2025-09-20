from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal


class GeneratedModel(BaseModel):
    """
    Model representing extracted patient vital signs and basic demographic information.
    """

    patient_id: Optional[str] = Field(
        None,
        description="Unique identifier for the patient."
    )
    name: Optional[str] = Field(
        None,
        description="Full name of the patient."
    )
    age: Optional[int] = Field(
        None,
        ge=0,
        description="Age of the patient in years."
    )
    gender: Optional[Literal["male", "female", "other"]] = Field(
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