from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Literal

class GeneratedModel(BaseModel):
    patient_name: str = Field(..., min_length=1)
    patient_number: int = Field(...)
    date_of_study: str = Field(..., format="yyyy-mm-dd")
    birthdate: str = Field(..., format="yyyy-mm-dd")
    sex: Literal["male", "female", "other"] = Field(...)
    mitral_valve_insufficiency: Literal["yes", "no"] = Field(...)
    mitral_valve_grade: Optional[Literal["mild", "moderate", "severe", "V1", "V2", "V3", "V4", "V5"]]
    mitral_valve_grade_number: Optional[int]
    special_note_mitral: Literal["none"] = Field(...)
    tricuspid_valve_insufficiency: Literal["yes", "no"] = Field(...)
    tricuspid_valve_grade: Optional[Literal["mild", "moderate", "severe", "V1", "V2", "V3", "V4", "V5"]]
    tricuspid_valve_grade_number: Optional[int]
    special_note_tricuspid: Literal["none"] = Field(...)
    confidence_score: float = Field(..., gt=0, lt=1)
    inferred_text: str = Field(...)