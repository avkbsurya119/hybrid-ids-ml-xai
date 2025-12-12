# api/model_loader.py

import joblib
import json
from pathlib import Path
import torch
import torch.nn as nn
import shap
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "models" / "artifacts"


# -----------------------------------------------------
# AutoEncoder
# -----------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=78):
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
        return self.decoder(self.encoder(x))


# -----------------------------------------------------
# Model Bundle
# -----------------------------------------------------
class ModelBundle:
    def __init__(self):

        # ---------------- LIGHTGBM ----------------
        print("✓ Loading LightGBM model…")
        self.model = lgb.Booster(model_file=str(ART / "lightgbm_model.txt"))

        # ---------------- SCALER -------------------
        print("✓ Loading scaler…")
        self.scaler = joblib.load(ART / "scaler.joblib")

        # ---------------- LABEL ENCODER ------------
        print("✓ Loading label encoder…")
        self.label_encoder = joblib.load(ART / "label_encoder.joblib")

        # ---------------- TRANSFORMS ---------------
        print("✓ Loading transforms.json…")
        with open(ART / "transforms.json", "r") as f:
            self.transforms = json.load(f)

        # ---------------- AUTOENCODER --------------
        print("✓ Loading autoencoder + threshold…")

        self.autoencoder = None
        self.autoencoder_threshold = None

        try:
            ae_state = ART / "autoencoder_state.pth"
            ae_thr = ART / "autoencoder_threshold.pkl"

            if ae_state.exists() and ae_thr.exists():

                thr = joblib.load(ae_thr)["threshold"]
                state = torch.load(ae_state, map_location="cpu")

                # resolve input dim
                first_key = list(state.keys())[0]
                input_dim = state[first_key].shape[1]

                # autoencoder class
                model = AutoEncoder(input_dim=input_dim)
                model.load_state_dict(state)
                model.eval()

                self.autoencoder = model
                self.autoencoder_threshold = thr

                print("✓ Autoencoder loaded successfully!")
            else:
                print("⚠ Autoencoder files missing.")

        except Exception as e:
            print("⚠ Autoencoder failed to load:", e)

        # ---------------- SHAP BACKGROUND ----------
        print("✓ Loading SHAP background…")
        try:
            self.shap_background = joblib.load(ART / "shap_background_sample.pkl")
        except:
            import pandas as pd
            self.shap_background = pd.read_parquet(ART / "shap_background_sample.parquet")

        # ---------------- SHAP EXPLAINER -----------
        print("✓ Loading SHAP explainer…")
        try:
            self.shap_explainer = joblib.load(ART / "shap_explainer.pkl")
        except Exception as e:
            print("⚠ Failed to load SHAP explainer:", e)
            self.shap_explainer = None

    # -----------------------------------------------------
    # TORCH CONVERSION (THIS WAS MISSING)
    # -----------------------------------------------------
    def to_tensor(self, X):
        return torch.tensor(X, dtype=torch.float32)


# global instance
model_bundle = ModelBundle()
