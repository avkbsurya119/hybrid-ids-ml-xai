# scripts/train_autoencoder.py
"""
Train an unsupervised Autoencoder on BENIGN traffic only.
Outputs:
 - models/artifacts/autoencoder_model.pth
 - models/artifacts/autoencoder_threshold.json
 - docs/autoencoder_report.md
"""

# ------------------------------------------------------------
# IMPORT & PATH FIX
# ------------------------------------------------------------
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from preprocessor.helpers import load_config, save_json

# ------------------------------------------------------------
# AUTOENCODER MODEL
# ------------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# ------------------------------------------------------------
# LOAD TRANSFORMED DATA
# ------------------------------------------------------------
def load_transformed():
    cfg = load_config(ROOT / "preprocessor" / "config.yaml")
    processed_dir = Path(cfg["processed_dir"]).resolve()

    parts = sorted(processed_dir.glob("transformed_part*.parquet"))
    if not parts:
        raise SystemExit("No transformed parquet found.")

    dfs = []
    for p in parts:
        print("Loading", p)
        df = pd.read_parquet(p)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print("Loaded:", df.shape)
    return df


# ------------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------------
def train_autoencoder(df):

    # Use only BENIGN traffic
    df_benign = df[df["Label"] == "BENIGN"]
    print("BENIGN data:", df_benign.shape)

    df_benign = df_benign.drop(columns=["Label"])
    X = df_benign.values.astype(np.float32)

    input_dim = X.shape[1]

    # Train/Val split
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=1024, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=1024)

    model = AutoEncoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    patience = 5
    wait = 0

    EPOCHS = 15

    for epoch in range(EPOCHS):
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

        # Validation
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

        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), ROOT / "models/artifacts/autoencoder_model.pth")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # ------------------ Compute Reconstruction Error Threshold ------------------
    model.load_state_dict(torch.load(ROOT / "models/artifacts/autoencoder_model.pth"))

    model.eval()
    errors = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch[0]
            out = model(x)
            loss = ((x - out) ** 2).mean(dim=1)  # per sample MSE
            errors.extend(loss.cpu().numpy())

    # threshold = mean + 3 * std deviation
    errors = np.array(errors)
    threshold = float(errors.mean() + 3 * errors.std())

    save_json(
        {"threshold": threshold},
        ROOT / "models/artifacts/autoencoder_threshold.json"
    )

    print("Saved threshold:", threshold)

    # ------------------ Report ------------------
    report_path = ROOT / "docs/autoencoder_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Autoencoder Anomaly Detector Report\n\n")
        f.write(f"**Validation Reconstruction Loss Mean:** {errors.mean()}\n")
        f.write(f"**Threshold (mean + 3σ):** {threshold}\n")

    print("Report saved →", report_path)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    df = load_transformed()
    train_autoencoder(df)
