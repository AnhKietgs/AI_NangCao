# ============================================================
# schemas/predict.py — Pydantic data models
# ============================================================

from pydantic import BaseModel, Field
from typing import List


class Prediction(BaseModel):
    word: str
    probability: float
    count: int


class PredictRequest(BaseModel):
    text: str = Field(..., example="the cat sat on")
    top_k: int = Field(5, ge=1, le=20, description="Number of predictions to return")
    n: int = Field(2, ge=2, le=3, description="N-gram order: 2=bigram, 3=trigram")


class PredictResponse(BaseModel):
    input_text: str
    predictions: List[Prediction]
    model_order: int
    context_used: str = ""
