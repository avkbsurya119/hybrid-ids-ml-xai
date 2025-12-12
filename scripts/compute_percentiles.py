# scripts/compute_percentiles.py
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
"""
Compute per-column percentiles from CSV(s) in data/raw.
This uses per-column sampling to avoid using too much memory on large datasets.

Outputs: preprocessor/percentiles.json
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
import random
from preprocessor.helpers import load_config, save_json

def sample_from_chunk(series, current_samples, max_samples):
    """
    Keep at most max_samples per column by reservoir-like sampling.
    (Simpler: if current len < max, add random subsample from chunk;
     else perform replacement sampling.)
    """
    arr = series.dropna().values
    n = len(arr)
    if n == 0:
        return current_samples
    if len(current_samples) < max_samples:
        need = max_samples - len(current_samples)
        # sample up to 'need' from arr
        if n <= need:
            current_samples.extend(arr.tolist())
        else:
            idx = np.random.choice(n, size=need, replace=False)
            current_samples.extend(arr[idx].tolist())
    else:
        # replacement: random replace some existing samples with chunk samples
        # pick k replacements proportional to chunk size but small
        k = min(max_samples // 100, n)  # tiny number to slowly refresh sample
        if k <= 0:
            return current_samples
        # sample k from arr
        new_vals = arr[np.random.choice(n, size=k, replace=False)]
        # replace random positions in current_samples
        for v in new_vals:
            i = random.randrange(len(current_samples))
            current_samples[i] = float(v)
    return current_samples

def main():
    cfg = load_config(Path(__file__).resolve().parents[1] / "preprocessor" / "config.yaml")
    raw_dir = Path(cfg["raw_data_dir"]).resolve()
    per_limit = int(cfg.get("percentile_sample_limit", 200000))
    percentiles = cfg.get("percentiles", [0.01,0.05,0.25,0.5,0.75,0.95,0.99])

    if not raw_dir.exists():
        raise SystemExit(f"raw_data_dir not found: {raw_dir}")

    # Dict: col -> sample list
    samples = {}
    processed_files = []
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        print("No CSVs found in", raw_dir)
        return

    for csv in csvs:
        print("Reading", csv)
        for chunk in pd.read_csv(csv, chunksize=int(cfg.get("chunk_size", 100000))):
            # for each numeric column in chunk, update sample list
            for col in chunk.columns:
                try:
                    if pd.api.types.is_numeric_dtype(chunk[col]):
                        # ensure key exists
                        lst = samples.get(col)
                        if lst is None:
                            lst = []
                            samples[col] = lst
                        # sample values from chunk
                        sample_from_chunk(chunk[col], lst, per_limit)
                except Exception as e:
                    # skip any column that errors
                    print("Warning: skipping col", col, "error:", e)
        processed_files.append(csv.name)

    # compute percentiles
    result = {"files_processed": processed_files, "percentiles": {}}
    for col, vals in samples.items():
        if len(vals) == 0:
            continue
        arr = np.array(vals, dtype=float)
        qs = np.quantile(arr, percentiles)
        # map percent values to numbers
        pct_map = {str(int(p*100)) : float(q) for p,q in zip(percentiles, qs)}
        result["percentiles"][col] = pct_map

    out_path = Path(cfg.get("percentiles_path", "preprocessor/percentiles.json"))
    save_json(result, out_path)
    print("Wrote percentiles to", out_path)

if __name__ == "__main__":
    main()
