from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch

from api.model_loader import model_bundle
from api.utils_preprocess import preprocess_input


class PredictRequest(BaseModel):
    data: dict


app = FastAPI(
    title="Hybrid NIDS (Binary + Attack + Autoencoder)",
    version="2.0"
)


@app.post("/predict")
def predict(req: PredictRequest):

    # -------- Binary Stage --------
    X_bin = preprocess_input(
        [req.data],
        model_bundle.binary_features,
        model_bundle.scaler
    )

    attack_prob = model_bundle.binary_model.predict(X_bin)[0]
    is_attack = attack_prob > 0.5

    result = {
        "binary_decision": "ATTACK" if is_attack else "BENIGN",
        "binary_confidence": float(attack_prob),
        "attack_type": None,
        "attack_confidence": None,
        "anomaly_score": None,
        "is_anomaly": None
    }

    # -------- Attack Type Stage --------
    if is_attack:
        X_att = preprocess_input(
            [req.data],
            model_bundle.attack_features,
            model_bundle.scaler
        )

        probs = model_bundle.attack_model.predict(X_att)[0]
        cls = int(np.argmax(probs))

        result["attack_type"] = (
            model_bundle.attack_label_encoder.inverse_transform([cls])[0]
        )
        result["attack_confidence"] = float(probs[cls])

    # -------- Autoencoder --------
    if model_bundle.autoencoder is not None:
        with torch.no_grad():
            x_tensor = model_bundle.to_tensor(X_bin)
            recon = model_bundle.autoencoder(x_tensor)
            mse = ((x_tensor - recon) ** 2).mean().item()

        result["anomaly_score"] = mse
        result["is_anomaly"] = mse > model_bundle.autoencoder_threshold

    return result
