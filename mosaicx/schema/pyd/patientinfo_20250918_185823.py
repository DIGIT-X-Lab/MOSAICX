from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class PatientInfo(BaseModel):
    """Schema for basic patient information including name and age."""
    name: str = Field(..., description="Full name of the patient.")
    age: int = Field(..., description="Age of the patient in years. [units=years]")
