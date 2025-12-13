"""
Train a LightGBM classifier on transformed parquet data.

Outputs:
 - models/artifacts/lightgbm_model.txt
 - models/artifacts/label_encoder.joblib
 - models/artifacts/full_feature_order.json
 - models/artifacts/selected_features.json
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
import json
import joblib
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

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
        dfs.append(pd.read_parquet(p))

    df = pd.concat(dfs, ignore_index=True)
    print("Loaded full dataset:", df.shape)
    return df


# ------------------------------------------------------------
# Training Pipeline
# ------------------------------------------------------------
def train_lightgbm(df: pd.DataFrame):

    # ----------------------
    # 1. Separate features / label
    # ----------------------
    target_col = "Label"
    if target_col not in df.columns:
        raise SystemExit("Label column not found")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ----------------------
    # Save FULL feature order (critical)
    # ----------------------
    full_feature_order = list(X.columns)
    full_feature_path = ROOT / "models/artifacts/full_feature_order.json"
    with open(full_feature_path, "w") as f:
        json.dump(full_feature_order, f, indent=2)

    print("✅ Saved FULL feature order →", full_feature_path)
    print("Feature count:", len(full_feature_order))

    # ----------------------
    # Label encoding
    # ----------------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ----------------------
    # Compute class weights (LightGBM-correct way)
    # ----------------------
    classes = np.unique(y_encoded)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_encoded
    )
    class_weight_map = dict(zip(classes, class_weights))

    print("✅ Computed class weights:")
    for cls, w in class_weight_map.items():
        print(f"  {le.inverse_transform([cls])[0]} → {w:.4f}")

    sample_weight = np.array([class_weight_map[c] for c in y_encoded])

    # ----------------------
    # Train / Validation split
    # ----------------------
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X,
        y_encoded,
        sample_weight,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)

    # ----------------------
    # LightGBM Dataset
    # ----------------------
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        weight=w_val,
        reference=train_data
    )

    # ----------------------
    # LightGBM Parameters
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
        "verbosity": -1
    }

    # ----------------------
    # Train model
    # ----------------------
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    print("Training complete.")

    # ----------------------
    # Save SELECTED features (used by model)
    # ----------------------
    selected_features = model.feature_name()
    selected_feature_path = ROOT / "models/artifacts/selected_features.json"
    with open(selected_feature_path, "w") as f:
        json.dump(selected_features, f, indent=2)

    print("✅ Saved SELECTED features →", selected_feature_path)
    print("Selected feature count:", len(selected_features))

    # ----------------------
    # Evaluate
    # ----------------------
    preds = model.predict(X_val)
    preds_classes = np.argmax(preds, axis=1)

    acc = accuracy_score(y_val, preds_classes)
    f1 = f1_score(y_val, preds_classes, average="weighted")
    cm = confusion_matrix(y_val, preds_classes)
    report = classification_report(
        y_val,
        preds_classes,
        target_names=le.classes_
    )

    print("\nAccuracy:", acc)
    print("F1 Score:", f1)

    # ----------------------
    # Save model & encoder
    # ----------------------
    model.save_model(str(ROOT / "models/artifacts/lightgbm_model.txt"))
    joblib.dump(le, ROOT / "models/artifacts/label_encoder.joblib")

    # ----------------------
    # Save training report
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
