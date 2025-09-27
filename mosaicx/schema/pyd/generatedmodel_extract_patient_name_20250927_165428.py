from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class GeneratedModel(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., gt=0)