"""
Retrain BOTH the scaler and autoencoder without percentile clipping.
This ensures training and inference use the EXACT same preprocessing.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
import json


# ============================================================
# AUTOENCODER MODEL
# ============================================================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48), nn.ReLU(),
            nn.Linear(48, 24), nn.ReLU(),
            nn.Linear(24, 12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 24), nn.ReLU(),
            nn.Linear(24, 48), nn.ReLU(),
            nn.Linear(48, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ============================================================
# PREPROCESSING (same as csv_preprocessor but without scaler)
# ============================================================
def apply_transforms(df, transforms):
    """Apply log1p transforms to specified columns"""
    df = df.copy()

    # Strip column names
    df.columns = df.columns.str.strip()

    # Drop label
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])

    # Apply log1p
    log1p_cols = transforms.get('log1p_columns', [])
    for col in log1p_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Clean infinities
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Clip extreme values
    df = df.clip(lower=-1e9, upper=1e9)

    return df


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
def load_data():
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Load feature order and transforms
    ART = ROOT / "models" / "artifacts"

    with open(ART / "transforms.json") as f:
        transforms = json.load(f)

    with open(ART / "attack_feature_order.json") as f:
        feature_order = json.load(f)

    # Load raw CSV
    raw_dir = ROOT / "data" / "raw"
    csv_files = list(raw_dir.glob("*.csv"))

    print(f"\nFound {len(csv_files)} CSV files")

    dfs = []
    for csv_file in csv_files[:3]:  # Use first 3 files for faster processing
        print(f"  Loading: {csv_file.name}")
        df = pd.read_csv(csv_file)
        df = df.replace([float("inf"), float("-inf")], 0).fillna(0)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df_all)}")

    # Apply transforms
    print("\nApplying transforms...")
    df_transformed = apply_transforms(df_all, transforms)

    # Enforce feature order
    for col in feature_order:
        if col not in df_transformed.columns:
            df_transformed[col] = 0

    df_transformed = df_transformed[feature_order]
    df_transformed = df_transformed.astype(np.float32)

    print(f"Transformed shape: {df_transformed.shape}")

    return df_transformed, feature_order


# ============================================================
# RETRAIN SCALER
# ============================================================
def retrain_scaler(df):
    print("\n" + "=" * 60)
    print("RETRAINING ROBUST SCALER")
    print("=" * 60)

    # Sample for scaler fitting (100k samples)
    if len(df) > 100000:
        df_sample = df.sample(n=100000, random_state=42)
    else:
        df_sample = df

    print(f"Fitting scaler on {len(df_sample)} samples...")

    scaler = RobustScaler()
    scaler.fit(df_sample.values)

    # Save new scaler
    scaler_path = ROOT / "models/artifacts/robust_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    print(f"[OK] Saved new scaler to {scaler_path}")

    # Show scaler stats
    print(f"\nScaler statistics:")
    print(f"  Center (median): min={scaler.center_.min():.2f}, max={scaler.center_.max():.2f}")
    print(f"  Scale (IQR): min={scaler.scale_.min():.6f}, max={scaler.scale_.max():.2f}")

    return scaler


# ============================================================
# TRAIN AUTOENCODER
# ============================================================
def train_autoencoder(df, scaler):
    print("\n" + "=" * 60)
    print("TRAINING AUTOENCODER")
    print("=" * 60)

    # Filter BENIGN traffic only
    # Since we already dropped Label, we need to reload to filter
    ART = ROOT / "models" / "artifacts"
    with open(ART / "transforms.json") as f:
        transforms = json.load(f)
    with open(ART / "attack_feature_order.json") as f:
        feature_order = json.load(f)

    # Reload with labels
    raw_dir = ROOT / "data" / "raw"
    csv_files = list(raw_dir.glob("*.csv"))

    dfs_benign = []
    for csv_file in csv_files[:3]:
        df_temp = pd.read_csv(csv_file)
        df_temp = df_temp.replace([float("inf"), float("-inf")], 0).fillna(0)

        if 'Label' in df_temp.columns:
            df_temp = df_temp[df_temp['Label'] == 'BENIGN']

        dfs_benign.append(df_temp)

    df_benign = pd.concat(dfs_benign, ignore_index=True)
    print(f"BENIGN samples: {len(df_benign)}")

    # Sample
    max_samples = 100000
    if len(df_benign) > max_samples:
        df_benign = df_benign.sample(n=max_samples, random_state=42)
        print(f"Sampled to {max_samples} for faster training")

    # Transform and scale
    df_benign = apply_transforms(df_benign, transforms)

    for col in feature_order:
        if col not in df_benign.columns:
            df_benign[col] = 0

    df_benign = df_benign[feature_order].astype(np.float32)

    # Scale
    X = scaler.transform(df_benign.values)
    input_dim = X.shape[1]

    print(f"\nPreprocessed BENIGN data: {X.shape}")
    print(f"Value range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Mean: {X.mean():.2f}, Std: {X.std():.2f}")

    # Train/Val split
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=1024,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=1024
    )

    # Model
    model = AutoEncoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    patience = 5
    wait = 0
    EPOCHS = 15

    print(f"\nTraining for up to {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0]
                out = model(x)
                loss = criterion(out, x)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        status = ""
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), ROOT / "models/artifacts/autoencoder_model.pth")
            status = " [BEST]"
        else:
            wait += 1

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}{status}")

        if wait >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    # Compute threshold
    print("\n" + "=" * 60)
    print("COMPUTING THRESHOLD")
    print("=" * 60)

    model.load_state_dict(torch.load(ROOT / "models/artifacts/autoencoder_model.pth"))
    model.eval()

    errors = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0]
            out = model(x)
            mse = torch.mean((x - out) ** 2, dim=1).numpy()
            errors.extend(mse)

    errors = np.array(errors)
    threshold = float(errors.mean() + 3 * errors.std())

    print(f"\nReconstruction Error Statistics:")
    print(f"  Mean: {errors.mean():.6f}")
    print(f"  Std:  {errors.std():.6f}")
    print(f"  Threshold (mean + 3*std): {threshold:.6f}")

    # Save
    with open(ROOT / "models/artifacts/autoencoder_threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    print(f"\n[OK] Saved autoencoder and threshold")

    # Validation
    anomalies = (errors > threshold).sum()
    print(f"\nValidation: {anomalies}/{len(errors)} ({100*anomalies/len(errors):.1f}%) flagged as anomalies")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df, feature_order = load_data()
    scaler = retrain_scaler(df)
    train_autoencoder(df, scaler)

    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE!")
    print("=" * 60)
    print("\nUpdated artifacts:")
    print("  - models/artifacts/robust_scaler.joblib")
    print("  - models/artifacts/autoencoder_model.pth")
    print("  - models/artifacts/autoencoder_threshold.json")
    print("\nNOTE: You may need to RETRAIN the binary and attack classifiers")
    print("      with the new scaler for consistency!")
