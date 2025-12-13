"""
Sanity check for Binary Classifier (BENIGN vs ATTACK)
Tests the binary model on transformed parquet data
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
from api.model_loader import model_bundle

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

# Sample 100 BENIGN and 100 ATTACK
benign = df[df["Label"] == "BENIGN"].sample(n=min(100, len(df[df["Label"] == "BENIGN"])), random_state=42)
attacks = df[df["Label"] != "BENIGN"].sample(n=min(100, len(df[df["Label"] != "BENIGN"])), random_state=42)

test_df = pd.concat([benign, attacks]).reset_index(drop=True)
true_labels = test_df["Label"].values
X = test_df.drop(columns=["Label"])

# Get predictions
predictions = model_bundle.binary_model.predict(X[model_bundle.binary_features].values)

# Binary classification (threshold 0.5)
binary_preds = (predictions > 0.5).astype(int)
binary_true = (true_labels != "BENIGN").astype(int)

# Calculate metrics
correct = (binary_preds == binary_true).sum()
accuracy = correct / len(binary_preds)

print("\n" + "="*60)
print("BINARY MODEL SANITY CHECK")
print("="*60)
print(f"Test samples: {len(test_df)}")
print(f"  BENIGN: {len(benign)}")
print(f"  ATTACK: {len(attacks)}")
print(f"\nAccuracy: {accuracy:.2%}")

# Confusion matrix
tp = ((binary_preds == 1) & (binary_true == 1)).sum()
tn = ((binary_preds == 0) & (binary_true == 0)).sum()
fp = ((binary_preds == 1) & (binary_true == 0)).sum()
fn = ((binary_preds == 0) & (binary_true == 1)).sum()

print(f"\nConfusion Matrix:")
print(f"  True Positives:  {tp}")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")

if accuracy >= 0.95:
    print(f"\n✅ PASS - Binary model working correctly (accuracy: {accuracy:.2%})")
else:
    print(f"\n⚠️  WARNING - Lower than expected accuracy: {accuracy:.2%}")

print("="*60)
