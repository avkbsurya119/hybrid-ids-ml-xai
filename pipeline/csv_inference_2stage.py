import numpy as np
import pandas as pd
import torch

from pipeline.csv_preprocessor import preprocess_csv

# ✅ Calibrated from your observed distribution
ATTACK_THRESHOLD = 0.01


def run_2stage_csv_inference(
    df_raw,
    bin_model,
    atk_model,
    atk_encoder,
    scaler,
    transforms,
    bin_features,
    atk_features,
    autoencoder=None,
    ae_threshold=None
):
    """
    Full 2-stage IDS inference on RAW CSV.

    Stage 1: BENIGN vs ATTACK
    Stage 2: Attack type (only if ATTACK)
    """

    # ---------------- Preprocess ----------------
    X_bin = preprocess_csv(df_raw, bin_features, transforms, scaler)
    X_atk = preprocess_csv(df_raw, atk_features, transforms, scaler)

    bin_probs = bin_model.predict(X_bin)

    results = []

    for i, p_attack in enumerate(bin_probs):

        # ---------------- Stage 1: Binary ----------------
        if p_attack < ATTACK_THRESHOLD:
            results.append({
                "Final_Prediction": "BENIGN",
                "Binary_Route": "BENIGN",
                "Confidence": round(float(1 - p_attack), 4),
                "Anomaly": False
            })
            continue

        # ---------------- Stage 2: Attack ----------------
        atk_probs = atk_model.predict(X_atk[i:i+1])[0]
        cls = int(np.argmax(atk_probs))

        attack_label = atk_encoder.inverse_transform([cls])[0]
        confidence = float(atk_probs[cls])

        # ---------------- Autoencoder ----------------
        anomaly = False
        if autoencoder is not None and ae_threshold is not None:
            with torch.no_grad():
                x_tensor = torch.tensor(X_atk[i:i+1], dtype=torch.float32)
                recon = autoencoder(x_tensor)
                mse = torch.mean((x_tensor - recon) ** 2).item()
                anomaly = mse > ae_threshold

        results.append({
            "Final_Prediction": attack_label,
            "Binary_Route": "ATTACK",
            "Confidence": round(confidence, 4),
            "Anomaly": anomaly
        })

    return pd.DataFrame(results)
