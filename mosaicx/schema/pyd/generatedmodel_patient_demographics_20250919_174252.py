from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Literal

class GeneratedModel(BaseModel):
    """Represents patient demographics"""

    first_name: str = Field(..., min_length=2, max_length=50)
    last_name: str = Field(..., min_length=2, max_length=50)
    age: int = Field(..., gt=0, le=120)
    gender: Literal["Male", "Female", "Other"]
    email: Optional[EmailStr] = None
    phone_number: str = Field(..., regex="^\\d{10}$")
    address: Dict[str, str] = Field({...}, min_items=3)
    is_active: bool = True