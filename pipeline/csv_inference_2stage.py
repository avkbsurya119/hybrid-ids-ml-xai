# pipeline/csv_inference_2stage.py

import numpy as np
import pandas as pd

from pipeline.csv_preprocessor import preprocess_csv

ATTACK_THRESHOLD = 0.01  # calibrated


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
    """

    # Preprocess ONCE per model
    X_bin = preprocess_csv(df_raw, bin_features, transforms, scaler)
    X_atk = preprocess_csv(df_raw, atk_features, transforms, scaler)

    bin_probs = bin_model.predict(X_bin)

    results = []

    for i, p_attack in enumerate(bin_probs):

        # ---------------- Stage 1: Binary ----------------
        if p_attack <= ATTACK_THRESHOLD:
            results.append({
                "Final_Prediction": "BENIGN",
                "Binary_Route": "BENIGN",
                "Confidence": round(1 - p_attack, 4),
                "Anomaly": False
            })
            continue

        # ---------------- Stage 2: Attack ----------------
        atk_probs = atk_model.predict(X_atk[i:i+1])[0]
        cls = atk_probs.argmax()

        attack_label = atk_encoder.inverse_transform([cls])[0]
        confidence = atk_probs[cls]

        # ---------------- Autoencoder (optional) ----------
        anomaly = False
        if autoencoder is not None:
            recon = autoencoder(
                autoencoder.to_tensor(X_atk[i:i+1])
            ).detach().numpy()

            mse = np.mean((X_atk[i:i+1] - recon) ** 2)
            anomaly = mse > ae_threshold

        results.append({
            "Final_Prediction": attack_label,
            "Binary_Route": "ATTACK",
            "Confidence": round(float(confidence), 4),
            "Anomaly": anomaly
        })

    return pd.DataFrame(results)
