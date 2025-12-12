# scripts/compute_percentiles.py
"""
Compute per-column percentiles from CSV(s) in data/raw.
Uses sampling to reduce RAM usage.

Outputs:
 - preprocessor/percentiles.json
"""

# ------------------------------------------------------------
# Fix import path
# ------------------------------------------------------------
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import random
from preprocessor.helpers import load_config, save_json


def sample_from_chunk(series, current_samples, max_samples):
    """
    Reservoir-like sampling to maintain up to max_samples values per column.
    """
    arr = series.dropna().values
    n = len(arr)
    if n == 0:
        return current_samples

    # Case 1: still filling buffer
    if len(current_samples) < max_samples:
        need = max_samples - len(current_samples)
        if n <= need:
            current_samples.extend(arr.tolist())
        else:
            idx = np.random.choice(n, size=need, replace=False)
            current_samples.extend(arr[idx].tolist())
        return current_samples

    # Case 2: replace a few existing values
    k = min(max_samples // 100, n)
    if k <= 0:
        return current_samples

    new_vals = arr[np.random.choice(n, size=k, replace=False)]
    for v in new_vals:
        i = random.randrange(len(current_samples))
        current_samples[i] = float(v)

    return current_samples


def main():
    cfg = load_config(ROOT / "preprocessor" / "config.yaml")

    raw_dir = Path(cfg["raw_data_dir"]).resolve()
    per_limit = int(cfg.get("percentile_sample_limit", 200000))
    percentiles = cfg.get("percentiles", [])

    if not raw_dir.exists():
        raise SystemExit(f"raw_data_dir not found: {raw_dir}")

    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        print("No CSVs found in", raw_dir)
        return

    samples = {}   # col → list of sampled values
    processed_files = []

    chunk_size = int(cfg.get("chunk_size", 100000))

    for csv in csvs:
        print("Reading", csv)
        for chunk in pd.read_csv(csv, chunksize=chunk_size):
            # Clean column names
            chunk.columns = [c.strip() for c in chunk.columns]

            for col in chunk.columns:
                if pd.api.types.is_numeric_dtype(chunk[col]):
                    lst = samples.get(col)
                    if lst is None:
                        lst = []
                        samples[col] = lst
                    sample_from_chunk(chunk[col], lst, per_limit)

        processed_files.append(csv.name)

    # ---------- Compute percentiles ----------
    result = {
        "files_processed": processed_files,
        "percentiles": {}
    }

    for col, vals in samples.items():
        if len(vals) == 0:
            continue
        arr = np.array(vals, dtype=float)
        qs = np.quantile(arr, percentiles)
        pct_map = {str(int(p * 100)): float(q) for p, q in zip(percentiles, qs)}
        result["percentiles"][col] = pct_map

    out_path = ROOT / cfg.get("percentiles_path", "preprocessor/percentiles.json")
    save_json(result, out_path)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
