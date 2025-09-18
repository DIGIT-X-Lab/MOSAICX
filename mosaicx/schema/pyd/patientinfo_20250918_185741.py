from __future__ import annotations
from pydantic import BaseModel, Field

class PatientInfo(BaseModel):
    """Schema for basic patient information including name and age."""
    name: str = Field(..., description="Full name of the patient.")
    age: int = Field(..., description="Age of the patient in years.")
