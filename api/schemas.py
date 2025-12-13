from pydantic import BaseModel
from typing import Dict, Any, Optional


class FlowInput(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    binary_prediction: str
    final_prediction: str
    confidence: float
    anomaly: bool

from typing import List
from pydantic import BaseModel


class CSVRowPrediction(BaseModel):
    final_prediction: str
    binary_route: str
    confidence: float
    anomaly: bool


class CSVPredictionResponse(BaseModel):
    total_rows: int
    attack_rows: int
    benign_rows: int
    results: List[CSVRowPrediction]
