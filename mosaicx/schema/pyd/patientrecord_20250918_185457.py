from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class PatientRecord(BaseModel):
    """Simple patient record with name, age, and blood type"""
    name: str = Field(..., description="Full name of the patient")
    age: int = Field(..., description="Age of the patient in years [units=years]")
    blood_type: Literal['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'] = Field(..., description="ABO and Rh blood type of the patient")
