from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    commentaire: str = Field(..., min_length=1)
    concurrent: Optional[str] = None


class PredictResponse(BaseModel):
    commentaire: str
    concurrent: Optional[str]
    problemes_predits: List[str]
    predictions: Dict[str, int]
    probabilities: Dict[str, float]
    thresholds: Dict[str, float]