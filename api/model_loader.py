import joblib
import json
from pathlib import Path
import torch
import torch.nn as nn
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "models" / "artifacts"


# ---------------- AUTOENCODER ----------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=78):
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


# ---------------- MODEL BUNDLE ----------------
class ModelBundle:
    def __init__(self):

        # -------- Transforms (CRITICAL) --------
        with open(ART / "transforms.json") as f:
            self.transforms = json.load(f)

        # -------- Binary model --------
        print("[OK] Loading binary model")
        self.binary_model = lgb.Booster(
            model_file=str(ART / "binary_lightgbm_model.txt")
        )

        with open(ART / "binary_feature_order.json") as f:
            self.binary_features = json.load(f)


        # -------- Attack classifier --------
        print("[OK] Loading attack classifier")
        self.attack_model = lgb.Booster(
            model_file=str(ART / "attack_lightgbm_model.txt")
        )

        self.attack_label_encoder = joblib.load(
            ART / "attack_label_encoder.joblib"
        )

        with open(ART / "attack_feature_order.json") as f:
            self.attack_features = json.load(f)


        # -------- Shared scaler --------
        self.scaler = joblib.load(ART / "robust_scaler.joblib")

        # -------- Autoencoder --------
        self.autoencoder = None
        self.autoencoder_threshold = None

        ae_state = ART / "autoencoder_model.pth"
        ae_thr = ART / "autoencoder_threshold.json"

        if ae_state.exists() and ae_thr.exists():
            state = torch.load(ae_state, map_location="cpu")
            input_dim = list(state.values())[0].shape[1]

            ae = AutoEncoder(input_dim)
            ae.load_state_dict(state)
            ae.eval()

            self.autoencoder = ae

            # Load threshold from JSON
            with open(ae_thr) as f:
                self.autoencoder_threshold = json.load(f)["threshold"]

        print("[OK] Model bundle ready")

    def to_tensor(self, X):
        return torch.tensor(X, dtype=torch.float32)


# Global singleton
model_bundle = ModelBundle()
