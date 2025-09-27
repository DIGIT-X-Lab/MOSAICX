from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Literal

class GeneratedModel(BaseModel):
    """Generated model for patient data"""
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0)