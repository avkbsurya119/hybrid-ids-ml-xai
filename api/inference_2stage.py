import pandas as pd
import numpy as np
import torch

from pipeline.csv_preprocessor import preprocess_csv

ATTACK_THRESHOLD = 0.01


def run_2stage_inference(record: dict, bundle):
    """
    2-stage inference for API (single flow).
    Missing features are AUTO-FILLED with 0 (same as training).
    """

    # Convert JSON record → DataFrame
    df = pd.DataFrame([record])

    # ---------------- Binary stage ----------------
    X_bin = preprocess_csv(
        df,
        bundle.binary_features,
        bundle.transforms,
        bundle.scaler
    )

    p_attack = float(bundle.binary_model.predict(X_bin)[0])

    # -------- Stage 1: BENIGN --------
    if p_attack <= ATTACK_THRESHOLD:
        return {
            "binary_prediction": "BENIGN",
            "final_prediction": "BENIGN",
            "confidence": round(1 - p_attack, 4),
            "anomaly": False
        }

    # ---------------- Stage 2: ATTACK ----------------
    X_atk = preprocess_csv(
        df,
        bundle.attack_features,
        bundle.transforms,
        bundle.scaler
    )

    atk_probs = bundle.attack_model.predict(X_atk)[0]
    cls = int(np.argmax(atk_probs))

    attack_label = bundle.attack_label_encoder.inverse_transform([cls])[0]
    confidence = float(atk_probs[cls])

    # ---------------- Autoencoder (optional) ----------------
    anomaly = False
    if bundle.autoencoder is not None:
        with torch.no_grad():
            x_tensor = torch.tensor(X_atk, dtype=torch.float32)
            recon = bundle.autoencoder(x_tensor)
            mse = torch.mean((x_tensor - recon) ** 2).item()
            anomaly = mse > bundle.autoencoder_threshold

    return {
        "binary_prediction": "ATTACK",
        "final_prediction": attack_label,
        "confidence": round(confidence, 4),
        "anomaly": anomaly
    }
