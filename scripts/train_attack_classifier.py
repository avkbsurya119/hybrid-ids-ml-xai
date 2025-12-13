"""
Stage-2 Attack Classifier Training Script
-----------------------------------------

Trains a LightGBM multiclass model ONLY on attack traffic
(after BENIGN has been filtered by the binary model).

Outputs:
 - models/artifacts/attack_lightgbm_model.txt
 - models/artifacts/attack_label_encoder.joblib
 - models/artifacts/attack_feature_order.json
 - docs/attack_training_report.md
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
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json

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
# Load transformed data (same as binary)
# ------------------------------------------------------------
def load_transformed_data():
    cfg = load_config(ROOT / "preprocessor" / "config.yaml")
    processed_dir = Path(cfg["processed_dir"]).resolve()

    parts = sorted(processed_dir.glob("transformed_part*.parquet"))
    if not parts:
        raise SystemExit("❌ No transformed parquet files found")

    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)

    print("Loaded dataset:", df.shape)
    return df


# ------------------------------------------------------------
# Train attack-type classifier
# ------------------------------------------------------------
def train_attack_classifier(df: pd.DataFrame):

    # ------------------------------
    # 1. Remove BENIGN completely
    # ------------------------------
    df = df[df["Label"] != "BENIGN"].copy()
    print("Attack-only dataset:", df.shape)

    X = df.drop(columns=["Label"])
    y = df["Label"]

    # ------------------------------
    # 2. Save feature order
    # ------------------------------
    feature_order = list(X.columns)
    feature_path = ROOT / "models/artifacts/attack_feature_order.json"

    with open(feature_path, "w") as f:
        json.dump(feature_order, f, indent=2)

    print("Saved attack feature order →", feature_path)
    print("Feature count:", len(feature_order))

    # ------------------------------
    # 3. Encode labels
    # ------------------------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Attack classes:", list(le.classes_))

    # ------------------------------
    # 4. Train / Validation split
    # ------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # ------------------------------
    # 5. LightGBM datasets
    # ------------------------------
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    # ------------------------------
    # 6. Model parameters
    # ------------------------------
    params = {
        "objective": "multiclass",
        "num_class": len(le.classes_),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 3,
        "verbosity": -1
    }

    # ------------------------------
    # 7. Train model
    # ------------------------------
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

    print("Training complete.")

    # ------------------------------
    # 8. Evaluation
    # ------------------------------
    preds = model.predict(X_val)
    preds_cls = preds.argmax(axis=1)

    acc = accuracy_score(y_val, preds_cls)
    f1 = f1_score(y_val, preds_cls, average="weighted")
    cm = confusion_matrix(y_val, preds_cls)
    report = classification_report(y_val, preds_cls, target_names=le.classes_)

    print("\nAttack Classifier Accuracy:", acc)
    print("F1 Score:", f1)

    # ------------------------------
    # 9. Save artifacts
    # ------------------------------
    model.save_model(str(ROOT / "models/artifacts/attack_lightgbm_model.txt"))
    joblib.dump(le, ROOT / "models/artifacts/attack_label_encoder.joblib")

    print("Saved attack classifier model + encoder")

    # ------------------------------
    # 10. Save report
    # ------------------------------
    report_path = ROOT / "docs/attack_training_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Attack Classifier Training Report\n\n")
        f.write(f"**Accuracy:** {acc}\n\n")
        f.write(f"**F1 Score:** {f1}\n\n")
        f.write("## Confusion Matrix\n")
        f.write(str(cm))
        f.write("\n\n## Classification Report\n")
        f.write("```\n" + report + "\n```\n")

    print("Wrote report →", report_path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    df = load_transformed_data()
    train_attack_classifier(df)
