from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class PatientRecord(BaseModel):
    """Patient record with name, age, and diagnosis"""
    name: str = Field(..., description="Full name of the patient")
    age: int = Field(..., description="Age of the patient in years [units=years]")
    diagnosis: str = Field(..., description="Medical diagnosis for the patient")
