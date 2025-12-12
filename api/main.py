# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch

from api.model_loader import model_bundle
from api.utils_preprocess import preprocess_input


class PredictRequest(BaseModel):
    data: dict


app = FastAPI(
    title="Hybrid NIDS ML + Autoencoder + SHAP API",
    version="1.0"
)


@app.get("/")
def root():
    return {"status": "Hybrid NIDS API running"}


@app.post("/predict")
def predict(req: PredictRequest):

    # input → dataframe → scaled → numpy array
    X = preprocess_input(req.data, model_bundle)

    # --------------------- LGBM PRED -----------------------
    raw_pred = model_bundle.model.predict(X)

    if raw_pred.ndim == 2:            # multiclass
        proba = raw_pred[0]
        pred_class = int(np.argmax(proba))
    else:                             # binary
        proba = np.array([1 - raw_pred[0], raw_pred[0]])
        pred_class = int(raw_pred[0] > 0.5)

    label = model_bundle.label_encoder.inverse_transform([pred_class])[0]

    # --------------------- AUTOENCODER ----------------------
    ae_score = None
    is_anomaly = None

    if model_bundle.autoencoder is not None:

        x_tensor = model_bundle.to_tensor(X)

        with torch.no_grad():
            recon = model_bundle.autoencoder(x_tensor)
            mse = ((x_tensor - recon) ** 2).mean().item()

        ae_score = mse
        is_anomaly = mse > model_bundle.autoencoder_threshold

    # --------------------- SHAP -----------------------------
    shap_values = None
    if model_bundle.shap_explainer is not None:
        try:
            sv = model_bundle.shap_explainer(X)
            shap_values = sv.values.tolist()
        except:
            shap_values = None

    return {
        "label": label,
        "probabilities": {str(i): float(p) for i, p in enumerate(proba)},
        "anomaly_score": ae_score,
        "is_anomaly": is_anomaly,
        "shap_values": shap_values
    }
