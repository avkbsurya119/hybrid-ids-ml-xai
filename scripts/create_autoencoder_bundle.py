import json
import torch
import joblib
from pathlib import Path

# ---------------------------
# Path Setup
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "models" / "artifacts"

state_dict_path = ART / "autoencoder_model.pth"
threshold_json_path = ART / "autoencoder_threshold.json"

print("Loading state_dict:", state_dict_path)
state_dict = torch.load(state_dict_path, map_location="cpu")

print("Loading threshold:", threshold_json_path)
with open(threshold_json_path, "r") as f:
    threshold = json.load(f)["threshold"]

# ---------------------------
# Define AutoEncoder (same architecture used during training)
# ---------------------------
import torch.nn as nn

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

# ---------------------------
# Detect Input Dimension Automatically
# ---------------------------
first_key = list(state_dict.keys())[0]
input_dim = state_dict[first_key].shape[1]

print("Detected input_dim =", input_dim)

model = AutoEncoder(input_dim)
model.load_state_dict(state_dict)
model.eval()

print("✓ Autoencoder weights loaded successfully!")

# ---------------------------
# Save Correct Deployment Files
# ---------------------------

torch.save(model.state_dict(), ART / "autoencoder_state.pth")
joblib.dump({"threshold": threshold}, ART / "autoencoder_threshold.pkl")

print("\n🎉 DONE — AUTOENCODER BUNDLE CREATED!")
print("Saved:")
print(" →", ART / "autoencoder_state.pth")
print(" →", ART / "autoencoder_threshold.pkl")
