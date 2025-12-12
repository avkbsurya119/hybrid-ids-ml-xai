# scripts/train_lightgbm.py
"""
Train a LightGBM classifier on transformed parquet data.

Outputs:
 - models/artifacts/lightgbm_model.txt
 - models/artifacts/label_encoder.joblib
 - docs/training_report.md
"""

# ------------------------------------------------------------
# Import & Path Fix
# ------------------------------------------------------------
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import lightgbm as lgb
import joblib

from preprocessor.helpers import load_config

# ------------------------------------------------------------
# Load transformed data
# ------------------------------------------------------------
def load_transformed_data():
    cfg = load_config(ROOT / "preprocessor" / "config.yaml")
    processed_dir = Path(cfg["processed_dir"]).resolve()

    parts = sorted(processed_dir.glob("transformed_part*.parquet"))
    if not parts:
        raise SystemExit("No transformed parquet files found. Run transform_and_scale.py first.")

    dfs = []
    for p in parts:
        print("Loading", p)
        df = pd.read_parquet(p)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print("Loaded full dataset:", df.shape)
    return df


# ------------------------------------------------------------
# Training Pipeline
# ------------------------------------------------------------
def train_lightgbm(df: pd.DataFrame):

    # ----------------------
    # 1. Separate features/label
    # ----------------------
    target_col = "Label"
    if target_col not in df.columns:
        raise SystemExit(f"Label column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Label encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ----------------------
    # 2. Train/Validation Split
    # ----------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)

    # ----------------------
    # 3. LightGBM Dataset
    # ----------------------
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # ----------------------
    # 4. LightGBM Parameters
    # ----------------------
    params = {
        "objective": "multiclass",
        "num_class": len(le.classes_),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 3,
    }

    # ----------------------
    # 5. Train model
    # ----------------------
    callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=100)
    ]

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=1000,
        callbacks=callbacks
    )


    print("Training complete.")

    # ----------------------
    # 6. Evaluate
    # ----------------------
    preds = model.predict(X_val)
    preds_classes = np.argmax(preds, axis=1)

    acc = accuracy_score(y_val, preds_classes)
    f1 = f1_score(y_val, preds_classes, average="weighted")
    cm = confusion_matrix(y_val, preds_classes)
    report = classification_report(y_val, preds_classes, target_names=le.classes_)

    print("\nAccuracy:", acc)
    print("F1 Score:", f1)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    # ----------------------
    # 7. Save model & encoder
    # ----------------------
    model_path = ROOT / "models/artifacts/lightgbm_model.txt"
    encoder_path = ROOT / "models/artifacts/label_encoder.joblib"

    model.save_model(str(model_path))
    joblib.dump(le, encoder_path)

    print("Saved LightGBM model →", model_path)
    print("Saved Label Encoder →", encoder_path)

    # ----------------------
    # 8. Save training report
    # ----------------------
    report_path = ROOT / "docs/training_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Training Report (LightGBM)\n\n")
        f.write(f"**Accuracy:** {acc}\n\n")
        f.write(f"**F1 Score:** {f1}\n\n")
        f.write("## Confusion Matrix\n")
        f.write(str(cm))
        f.write("\n\n## Classification Report\n")
        f.write("```\n" + report + "\n```\n")

    print("Wrote training report →", report_path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    df = load_transformed_data()
    train_lightgbm(df)
