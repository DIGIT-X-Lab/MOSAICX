from pydantic import BaseModel, Field
from typing import Optional

class GeneratedModel(BaseModel):
    """
    A model to extract patient name and age from clinical data.
    """
    name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="The full name of the patient"
    )
    
    age: Optional[int] = Field(
        default=None,
        gt=0,  # Age must be positive
        le=120,  # Age cannot exceed typical human lifespan (120 years)
        ge=0,   # Allow zero for newborns or missing data via optional field
        description="The age of the patient in years"
    )