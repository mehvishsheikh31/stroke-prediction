from pydantic import BaseModel, Field
from typing import Literal, Optional

class StrokeInput(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: float = Field(..., gt=0, lt=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    ever_married: Literal["Yes", "No"]
    work_type: Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    Residence_type: Literal["Urban", "Rural"]
    avg_glucose_level: float = Field(..., gt=0, lt=500)
    bmi: Optional[float] = Field(None, gt=0, lt=100)
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]