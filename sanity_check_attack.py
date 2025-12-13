"""
Sanity check for Attack Classifier (multi-class attack types)
Tests the attack model on transformed parquet data (ATTACK samples only)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
from api.model_loader import model_bundle
from sklearn.metrics import classification_report

# Load sample of transformed data
processed_dir = ROOT / "data" / "processed"
parts = sorted(processed_dir.glob("transformed_part*.parquet"))

if not parts:
    print("❌ No transformed parquet files found. Run transform_and_scale.py first.")
    sys.exit(1)

# Load first part
df = pd.read_parquet(parts[0])
print(f"Loaded {len(df):,} rows from {parts[0].name}")

# Get labels
if "Label" not in df.columns:
    print("❌ No Label column found")
    sys.exit(1)

# Filter ATTACK samples only
attacks = df[df["Label"] != "BENIGN"].copy()

if len(attacks) == 0:
    print("❌ No attack samples found in data")
    sys.exit(1)

# Sample up to 200 attack samples
test_df = attacks.sample(n=min(200, len(attacks)), random_state=42)
true_labels = test_df["Label"].values
X = test_df.drop(columns=["Label"])

# Get predictions
probs = model_bundle.attack_model.predict(X[model_bundle.attack_features].values)
pred_classes = np.argmax(probs, axis=1)
pred_labels = model_bundle.attack_label_encoder.inverse_transform(pred_classes)

# Calculate accuracy
correct = (pred_labels == true_labels).sum()
accuracy = correct / len(pred_labels)

print("\n" + "="*60)
print("ATTACK CLASSIFIER SANITY CHECK")
print("="*60)
print(f"Test samples: {len(test_df)}")
print(f"Attack types in test: {len(set(true_labels))}")
print(f"\nAccuracy: {accuracy:.2%}")

# Classification report
print(f"\nClassification Report:")
print(classification_report(true_labels, pred_labels, zero_division=0))

if accuracy >= 0.90:
    print(f"\n✅ PASS - Attack classifier working correctly (accuracy: {accuracy:.2%})")
else:
    print(f"\n⚠️  WARNING - Lower than expected accuracy: {accuracy:.2%}")

print("="*60)
