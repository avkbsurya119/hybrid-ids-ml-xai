import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
"""
scripts/validate_pipeline.py

Quick validation run that:
 - loads percentiles & transforms metadata & scaler
 - applies transforms to a small sample from processed parquet
 - prints basic sanity checks
"""

from pathlib import Path
import pandas as pd
import numpy as np
from preprocessor.helpers import load_json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
cfg_path = ROOT / "preprocessor" / "config.yaml"
cfg = None
import yaml
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

processed_dir = Path(cfg["processed_dir"]).resolve()
percentiles_path = Path(cfg.get("percentiles_path", "preprocessor/percentiles.json"))
transforms_path = Path(cfg.get("transforms_path", "preprocessor/transforms.json"))
scaler_path = Path(cfg.get("scaler_artifact", "models/artifacts/robust_scaler.joblib"))

for p in [percentiles_path, transforms_path, scaler_path]:
    if not p.exists():
        raise SystemExit(f"Required artifact missing: {p}")

print("Loaded artifacts exist.")
percentiles = load_json(percentiles_path)
transforms = load_json(transforms_path)
scaler = joblib.load(scaler_path)
print("Loaded scaler (type):", type(scaler))

# sample a few rows
parts = sorted(processed_dir.glob("*.parquet"))
if not parts:
    raise SystemExit("No processed parquet parts to validate.")

sample_df = pd.read_parquet(parts[0]).head(10)
print("Sample rows shape:", sample_df.shape)

# basic checks
log_cols = transforms.get("log1p_columns", [])
print("log1p columns from transforms:", log_cols)
print("Any NaNs in sample before transform?:", sample_df.isna().any().any())

# If scaler exists, run transform
try:
    numeric_cols = [c for c in sample_df.columns if pd.api.types.is_numeric_dtype(sample_df[c])]
    # fill NA
    sample_filled = sample_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    scaled = scaler.transform(sample_filled.values)
    print("Scaled shape:", scaled.shape)
    print("Scaled sample mean (per-col):", np.round(np.nanmean(scaled, axis=0)[:5], 4))
except Exception as e:
    print("Error applying scaler:", e)

print("Validation run complete.")
