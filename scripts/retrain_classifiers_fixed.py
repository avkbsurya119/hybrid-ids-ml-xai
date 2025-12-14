"""
Retrain Binary and Attack classifiers with CORRECT preprocessing.
Fixes the preprocessing mismatch that caused 0% DDoS detection accuracy.

This script uses the SAME preprocessing as inference (no percentile clipping).
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def apply_transforms_and_scale(df, transforms, scaler, feature_order):
    """Apply same transforms as csv_preprocessor (NO percentile clipping)"""
    df = df.copy()

    # Strip column names
    df.columns = df.columns.str.strip()

    # Apply log1p transforms
    log1p_cols = transforms.get('log1p_columns', [])
    for col in log1p_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Enforce feature order
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_order]

    # Clean infinities
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Clip extreme values (same as csv_preprocessor)
    df = df.clip(lower=-1e9, upper=1e9)

    # Convert to float32
    df = df.astype(np.float32)

    # Scale
    X = scaler.transform(df.values)

    return X


def load_and_preprocess_data():
    """Load all raw CSV files and preprocess consistently"""

    print_section("LOADING RAW CSV DATA")

    # Load artifacts
    ART = ROOT / "models" / "artifacts"

    with open(ART / "transforms.json") as f:
        transforms = json.load(f)

    with open(ART / "attack_feature_order.json") as f:
        feature_order = json.load(f)

    scaler = joblib.load(ART / "robust_scaler.joblib")

    # Load all raw CSV files
    raw_dir = ROOT / "data" / "raw"
    csv_files = list(raw_dir.glob("*.csv"))

    print(f"\nFound {len(csv_files)} CSV files")

    all_data = []
    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}")
        df = pd.read_csv(csv_file)
        df = df.replace([float("inf"), float("-inf")], 0).fillna(0)
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows: {len(df_all):,}")

    # Check label column
    if ' Label' in df_all.columns:
        label_col = ' Label'
    elif 'Label' in df_all.columns:
        label_col = 'Label'
    else:
        raise ValueError("No Label column found!")

    print(f"\nLabel distribution:")
    print(df_all[label_col].value_counts())

    # Separate features and labels
    labels = df_all[label_col].copy()
    df_features = df_all.drop(columns=[label_col])

    # Apply transforms and scaling
    print("\nApplying preprocessing (same as inference)...")
    X = apply_transforms_and_scale(df_features, transforms, scaler, feature_order)

    print(f"\nPreprocessed shape: {X.shape}")
    print(f"Value range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Mean: {X.mean():.4f}, Std: {X.std():.4f}")

    return X, labels, feature_order


def train_binary_classifier(X, labels, feature_order):
    """Train binary classifier: BENIGN (0) vs ATTACK (1)"""

    print_section("TRAINING BINARY CLASSIFIER")

    # Create binary labels
    y_binary = (labels != "BENIGN").astype(int)

    print(f"\nBinary distribution:")
    print(f"  BENIGN (0): {(y_binary == 0).sum():,}")
    print(f"  ATTACK (1): {(y_binary == 1).sum():,}")

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    print(f"\nTrain samples: {len(X_train):,}")
    print(f"Val samples: {len(X_val):,}")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Training parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    # Train
    print("\nTraining binary model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=20)
        ]
    )

    # Evaluate
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f"\n{'='*70}")
    print(f"BINARY MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"F1 Score: {f1:.6f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['BENIGN', 'ATTACK']))

    # Save model
    model.save_model(str(ROOT / "models/artifacts/binary_lightgbm_model.txt"))

    # Save feature order
    with open(ROOT / "models/artifacts/binary_feature_order.json", "w") as f:
        json.dump(feature_order, f, indent=2)

    # Save label encoder
    le = LabelEncoder()
    le.fit([0, 1])
    joblib.dump(le, ROOT / "models/artifacts/binary_label_encoder.joblib")

    print("\n[OK] Binary model saved")

    return model


def train_attack_classifier(X, labels, feature_order):
    """Train attack classifier (multiclass)"""

    print_section("TRAINING ATTACK CLASSIFIER")

    # Filter out BENIGN (attack classifier only sees attacks)
    attack_mask = labels != "BENIGN"
    X_attacks = X[attack_mask]
    y_attacks = labels[attack_mask]

    print(f"\nAttack samples: {len(X_attacks):,}")
    print(f"\nAttack distribution:")
    print(y_attacks.value_counts())

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_attacks)

    print(f"\nNumber of attack classes: {len(le.classes_)}")
    print("\nClasses:")
    for i, cls in enumerate(le.classes_):
        print(f"  {i:2d}. {cls}")

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_attacks, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nTrain samples: {len(X_train):,}")
    print(f"Val samples: {len(X_val):,}")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Training parameters
    params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    # Train
    print("\nTraining attack classifier...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=20)
        ]
    )

    # Evaluate
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')

    print(f"\n{'='*70}")
    print(f"ATTACK CLASSIFIER PERFORMANCE")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"F1 Score: {f1:.6f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # Check DDoS performance specifically
    if 'DDoS' in le.classes_:
        ddos_idx = list(le.classes_).index('DDoS')
        ddos_mask = y_val == ddos_idx
        ddos_accuracy = accuracy_score(y_val[ddos_mask], y_pred[ddos_mask])

        print(f"\n{'='*70}")
        print(f"DDoS SPECIFIC PERFORMANCE")
        print(f"{'='*70}")
        print(f"DDoS Accuracy: {ddos_accuracy:.6f}")
        print(f"DDoS samples in validation: {ddos_mask.sum()}")
        print(f"DDoS correctly predicted: {(y_pred[ddos_mask] == ddos_idx).sum()}")

    # Save model
    model.save_model(str(ROOT / "models/artifacts/attack_lightgbm_model.txt"))

    # Save feature order
    with open(ROOT / "models/artifacts/attack_feature_order.json", "w") as f:
        json.dump(feature_order, f, indent=2)

    # Save label encoder
    joblib.dump(le, ROOT / "models/artifacts/attack_label_encoder.joblib")

    print("\n[OK] Attack classifier saved")

    return model


def main():
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  RETRAINING CLASSIFIERS WITH CORRECT PREPROCESSING".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Load and preprocess data
    X, labels, feature_order = load_and_preprocess_data()

    # Train both models
    binary_model = train_binary_classifier(X, labels, feature_order)
    attack_model = train_attack_classifier(X, labels, feature_order)

    # Final summary
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  RETRAINING COMPLETE!".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    print("\nUpdated models:")
    print("  - models/artifacts/binary_lightgbm_model.txt")
    print("  - models/artifacts/binary_feature_order.json")
    print("  - models/artifacts/binary_label_encoder.joblib")
    print("  - models/artifacts/attack_lightgbm_model.txt")
    print("  - models/artifacts/attack_feature_order.json")
    print("  - models/artifacts/attack_label_encoder.joblib")

    print("\nNext steps:")
    print("  1. Test DDoS detection with: python test_ddos_predictions.py")
    print("  2. Restart Docker containers: docker-compose restart")
    print("  3. Test full pipeline with Streamlit")


if __name__ == "__main__":
    main()
