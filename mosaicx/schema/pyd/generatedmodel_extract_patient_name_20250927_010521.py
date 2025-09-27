from pydantic import BaseModel, Field
from typing import Optional

class GeneratedModel(BaseModel):
    name: str = Field(...,
        title="Patient Name",
        description="The full name of the patient.",
        min_length=1)
    
    age: Optional[int] = Field(None,
        title="Patient Age",
        description=(
            "The age of the patient in years. "
            "This field is optional for infants or unknown ages."
        ),
        ge=0)