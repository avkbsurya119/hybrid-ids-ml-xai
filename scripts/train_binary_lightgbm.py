"""
Stage-1 Binary IDS Model
BENIGN (0) vs ATTACK (1)

Outputs:
- models/artifacts/binary_lightgbm_model.txt
- models/artifacts/binary_label_encoder.joblib
- models/artifacts/binary_feature_order.json
- docs/binary_training_report.md
"""

# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from preprocessor.helpers import load_config

# ------------------------------------------------------------
# Load transformed parquet data
# ------------------------------------------------------------
def load_transformed_data():
    cfg = load_config(ROOT / "preprocessor" / "config.yaml")
    processed_dir = Path(cfg["processed_dir"]).resolve()

    parts = sorted(processed_dir.glob("transformed_part*.parquet"))
    if not parts:
        raise SystemExit("❌ No transformed parquet files found")

    dfs = []
    for p in parts:
        print("Loading", p.name)
        dfs.append(pd.read_parquet(p))

    df = pd.concat(dfs, ignore_index=True)
    print("Loaded dataset:", df.shape)
    return df

# ------------------------------------------------------------
# Binary training pipeline
# ------------------------------------------------------------
def train_binary_model(df: pd.DataFrame):

    if "Label" not in df.columns:
        raise SystemExit("❌ Label column not found")

    # ----------------------
    # Binary conversion
    # ----------------------
    df["BinaryLabel"] = (df["Label"] != "BENIGN").astype(int)

    X = df.drop(columns=["Label", "BinaryLabel"])
    y = df["BinaryLabel"]

    # ----------------------
    # Save feature order
    # ----------------------
    feature_order = list(X.columns)
    feature_path = ROOT / "models/artifacts/binary_feature_order.json"
    with open(feature_path, "w") as f:
        json.dump(feature_order, f, indent=2)

    print("✅ Saved binary feature order:", len(feature_order))

    # ----------------------
    # Encode labels (0/1)
    # ----------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ----------------------
    # Train / Val split
    # ----------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc
    )

    print("Train:", X_train.shape, "Val:", X_val.shape)

    # ----------------------
    # LightGBM datasets
    # ----------------------
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # ----------------------
    # Parameters (binary)
    # ----------------------
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 3,
        "verbosity": -1
    }

    # ----------------------
    # Train
    # ----------------------
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )

    print("✅ Binary model training complete")

    # ----------------------
    # Evaluation
    # ----------------------
    val_probs = model.predict(X_val)
    val_preds = (val_probs > 0.5).astype(int)

    acc = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    cm = confusion_matrix(y_val, val_preds)
    report = classification_report(
        y_val,
        val_preds,
        target_names=["BENIGN", "ATTACK"]
    )

    print("\nAccuracy:", acc)
    print("F1:", f1)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    # ----------------------
    # Save artifacts
    # ----------------------
    model.save_model(ROOT / "models/artifacts/binary_lightgbm_model.txt")
    joblib.dump(le, ROOT / "models/artifacts/binary_label_encoder.joblib")

    # ----------------------
    # Save report
    # ----------------------
    report_path = ROOT / "docs/binary_training_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Binary IDS Training Report\n\n")
        f.write(f"Accuracy: {acc}\n\n")
        f.write(f"F1 Score: {f1}\n\n")
        f.write("## Confusion Matrix\n")
        f.write(str(cm))
        f.write("\n\n## Classification Report\n")
        f.write("```\n" + report + "\n```\n")

    print("📄 Report saved →", report_path)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    df = load_transformed_data()
    train_binary_model(df)
