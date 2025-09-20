from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal


class GeneratedModel(BaseModel):
    """
    Model representing a patient's vital statistics and attending physician.
    """

    patient_name: str = Field(..., min_length=1, description="Full name of the patient")
    age: int = Field(..., ge=0, description="Age of the patient in years")
    weight_kg: float = Field(..., gt=0, description="Weight of the patient in kilograms")
    bmi: float = Field(..., gt=0, description="Body Mass Index of the patient")
    blood_pressure_systolic: int = Field(..., gt=0, description="Systolic blood pressure (mmHg)")
    blood_pressure_diastolic: int = Field(..., gt=0, description="Diastolic blood pressure (mmHg)")
    heart_rate: int = Field(..., gt=0, description="Heart rate in beats per minute")
    temperature_celsius: float = Field(..., gt=30, lt=45, description="Body temperature in Celsius")
    attending_physician: str = Field(..., min_length=1, description="Name of the attending physician")