from pydantic import BaseModel, Field
from typing import Dict, List


class FlowInput(BaseModel):
    data: Dict[str, float]


class PredictionResponse(BaseModel):
    label: str
    probabilities: Dict[str, float]
    anomaly_score: float | None
    is_anomaly: bool


class SHAPResponse(BaseModel):
    shap_values: List[float]
    expected_value: float
