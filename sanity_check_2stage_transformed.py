"""
Sanity check for 2-stage IDS on TRANSFORMED data.

Stage 1: Binary classifier (BENIGN vs ATTACK)
Stage 2: Attack-type classifier (only if ATTACK)

IMPORTANT:
- Transformed parquet is ALREADY SCALED
- DO NOT apply scaler again
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
from pathlib import Path

# --------------------------------------------------
# Config
# --------------------------------------------------
ATTACK_THRESHOLD = 0.01
SAMPLE_PER_CLASS = 10
RANDOM_STATE = 42

ROOT = Path.cwd()
ART = ROOT / "models" / "artifacts"

# --------------------------------------------------
# Load models
# --------------------------------------------------
print("✓ Loading binary model")
bin_model = lgb.Booster(model_file=str(ART / "binary_lightgbm_model.txt"))

print("✓ Loading attack classifier")
atk_model = lgb.Booster(model_file=str(ART / "attack_lightgbm_model.txt"))
atk_le = joblib.load(ART / "attack_label_encoder.joblib")

with open(ART / "binary_feature_order.json") as f:
    bin_feats = json.load(f)

with open(ART / "attack_feature_order.json") as f:
    atk_feats = json.load(f)

print("✓ Model bundle ready")

# --------------------------------------------------
# Load TRANSFORMED data (already scaled)
# --------------------------------------------------
df = pd.read_parquet("data/processed/transformed_part0.parquet")

print("\nLabel distribution:")
print(df["Label"].value_counts())

# --------------------------------------------------
# Controlled sample: BENIGN + one ATTACK
# --------------------------------------------------
df_benign = df[df["Label"] == "BENIGN"]

attack_labels = df[df["Label"] != "BENIGN"]["Label"].unique()
assert len(attack_labels) > 0, "No attack samples found"

chosen_attack = attack_labels[0]
print(f"\nUsing attack class → {chosen_attack}")

df_attack = df[df["Label"] == chosen_attack]

sample = pd.concat([
    df_benign.sample(min(SAMPLE_PER_CLASS, len(df_benign)), random_state=RANDOM_STATE),
    df_attack.sample(min(SAMPLE_PER_CLASS, len(df_attack)), random_state=RANDOM_STATE)
]).sample(frac=1, random_state=99)

print("\nSample label counts:")
print(sample["Label"].value_counts())

# --------------------------------------------------
# Inference
# --------------------------------------------------
results = []

for _, row in sample.iterrows():

    # -------- Stage 1: Binary (NO SCALING) --------
    X_bin = row[bin_feats].to_frame().T.values
    p_attack = bin_model.predict(X_bin)[0]

    if p_attack <= ATTACK_THRESHOLD:
        binary_pred = "BENIGN"
        final_pred = "BENIGN"
        confidence = 1 - p_attack

    else:
        binary_pred = "ATTACK"

        # -------- Stage 2: Attack classifier --------
        X_atk = row[atk_feats].to_frame().T.values
        probs = atk_model.predict(X_atk)[0]
        cls = probs.argmax()

        final_pred = atk_le.inverse_transform([cls])[0]
        confidence = probs[cls]

    results.append({
        "Actual": row["Label"],
        "Binary": binary_pred,
        "Final_Prediction": final_pred,
        "Confidence": round(float(confidence), 4)
    })

# --------------------------------------------------
# Results
# --------------------------------------------------
out = pd.DataFrame(results)

print("\n===== 2-STAGE SANITY CHECK =====")
print(out)

print("\nFinal prediction counts:")
print(out["Final_Prediction"].value_counts())

print("\nBinary routing counts:")
print(out["Binary"].value_counts())

matches = (out["Actual"] == out["Final_Prediction"]).sum()
print(f"\nExact matches: {matches} / {len(out)}")
