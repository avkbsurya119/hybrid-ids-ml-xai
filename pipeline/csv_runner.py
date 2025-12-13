# pipeline/csv_runner.py
# pipeline/csv_runner.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import json
import joblib
import lightgbm as lgb
from pathlib import Path

from pipeline.csv_inference_2stage import run_2stage_csv_inference
from api.model_loader import model_bundle   # reuses autoencoder safely


ROOT = Path.cwd()
ART = ROOT / "models" / "artifacts"

# ---------------- Load models ----------------
bin_model = lgb.Booster(model_file=str(ART / "binary_lightgbm_model.txt"))
atk_model = lgb.Booster(model_file=str(ART / "attack_lightgbm_model.txt"))

atk_encoder = joblib.load(ART / "attack_label_encoder.joblib")
scaler = joblib.load(ART / "scaler.joblib")

with open(ART / "binary_feature_order.json") as f:
    bin_feats = json.load(f)

with open(ART / "attack_feature_order.json") as f:
    atk_feats = json.load(f)

with open(ART / "transforms.json") as f:
    transforms = json.load(f)

# ---------------- Load RAW CSV ----------------
CSV_PATH = "data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

df = (
    pd.read_csv(CSV_PATH)
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

# Random 20 rows
df = df.sample(20, random_state=42)

# ---------------- Run inference ----------------
out = run_2stage_csv_inference(
    df_raw=df,
    bin_model=bin_model,
    atk_model=atk_model,
    atk_encoder=atk_encoder,
    scaler=scaler,
    transforms=transforms,
    bin_features=bin_feats,
    atk_features=atk_feats,
    autoencoder=model_bundle.autoencoder,
    ae_threshold=model_bundle.autoencoder_threshold
)

# Detect label column safely (handles CIC datasets)
label_col = None
for c in df.columns:
    if c.strip().lower() == "label":
        label_col = c
        break

# Attach actual labels if present
if label_col is not None:
    result_df = pd.concat(
        [
            df[[label_col]]
            .reset_index(drop=True)
            .rename(columns={label_col: "Actual"}),
            out
        ],
        axis=1
    )
else:
    result_df = out


print("\n===== CSV 2-STAGE RESULTS =====")
print(result_df.head(20))


