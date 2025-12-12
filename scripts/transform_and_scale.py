# scripts/transform_and_scale.py
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
"""
Apply clipping, heavy-tail detection + log1p transforms, and fit RobustScaler.
Reads the percentiles file created earlier.

Outputs:
 - preprocessor/transforms.json  (records which columns got log1p)
 - models/artifacts/robust_scaler.joblib
 - data/processed/<transformed>_part*.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
from preprocessor.helpers import load_config, load_json, save_json, save_joblib
from sklearn.preprocessing import RobustScaler
import joblib
import math

def safe_apply_log1p(s: pd.Series):
    """Apply log1p safely — only on positive numeric values; keep zeros as 0."""
    # if any value <0, do not apply
    if (s.dropna() < 0).any():
        return None
    # if too many zeros, still ok; log1p handles 0 -> 0
    return np.log1p(s)

def main():
    cfg = load_config(Path(__file__).resolve().parents[1] / "preprocessor" / "config.yaml")
    raw_dir = Path(cfg["raw_data_dir"]).resolve()
    processed_dir = Path(cfg["processed_dir"]).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    per_path = Path(cfg.get("percentiles_path", "preprocessor/percentiles.json"))
    transforms_path = Path(cfg.get("transforms_path", "preprocessor/transforms.json"))
    scaler_path = Path(cfg.get("scaler_artifact", "models/artifacts/robust_scaler.joblib"))

    if not per_path.exists():
        raise SystemExit("Percentiles file not found. Run compute_percentiles.py first.")

    per_data = load_json(per_path)
    percentiles_map = per_data.get("percentiles", {})

    clip_lo_pct = str(int(cfg.get("clip_lower_pct", 0.01)*100))
    clip_hi_pct = str(int(cfg.get("clip_upper_pct", 0.99)*100))

    # Prepare metadata
    transforms_meta = {"log1p_columns": [], "clipping": {"lower_pct": clip_lo_pct, "upper_pct": clip_hi_pct}}

    numeric_cols = sorted(percentiles_map.keys())

    # Determine heavy-tail candidates
    heavy_candidates = []
    for col in numeric_cols:
        try:
            cmap = percentiles_map[col]
            if (clip_hi_pct not in cmap) or ("50" not in cmap):
                continue
            p99 = float(cmap.get(clip_hi_pct))
            p50 = float(cmap.get("50"))
            # avoid division by zero
            if p50 == 0:
                ratio = float("inf")
            else:
                ratio = p99 / p50 if p50 != 0 else float("inf")
            # we will compute skew from a small sample later. For now test ratio.
            if ratio >= float(cfg.get("heavytail_ratio_threshold", 10.0)):
                heavy_candidates.append(col)
        except Exception:
            continue

    print("Heavy-tail candidates by ratio:", heavy_candidates)

    # We'll do two passes:
    # 1) Read processed parquet parts or raw CSVs, apply clipping and detect skew per column (sample).
    # 2) Decide which candidates get log1p and then apply transforms + fit scaler.
    sample_for_scaler = []
    sample_rows_limit = 200000  # limit for scaler fitting sample
    current_sample_count = 0

    # helper to maybe read parquet parts first
    data_sources = sorted(list(processed_dir.glob("*.parquet")) + list(Path(cfg["raw_data_dir"]).glob("*.csv")))
    if not data_sources:
        raise SystemExit("No data files found in processed or raw locations for transform.")

    # gather skew statistics
    skew_map = {}
    for src in data_sources:
        print("Scanning for skew from", src)
        if src.suffix == ".parquet":
            # read in chunks by partitioning rows (parquet slices will fit)
            df_iter = [pd.read_parquet(src)]
        else:
            df_iter = pd.read_csv(src, chunksize=int(cfg.get("chunk_size", 100000)))
        for df in df_iter:
            # ensure numeric cols are numeric
            for col in numeric_cols:
                if col not in df.columns:
                    continue
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # try convert
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                s = df[col].dropna()
                if s.shape[0] == 0:
                    continue
                # update skew map with incremental estimation using sample
                prev = skew_map.get(col, [])
                # take a small sample from s
                take = min(1000, s.shape[0])
                sampled = s.sample(n=take, random_state=42).values
                prev.extend(sampled.tolist())
                # cap stored sample to 5000 per col
                if len(prev) > 5000:
                    prev = prev[-5000:]
                skew_map[col] = prev

            # collect rows for scaler sample
            if current_sample_count < sample_rows_limit:
                # sample a fraction of this chunk
                rows_to_take = min(2000, sample_rows_limit - current_sample_count)
                if df.shape[0] <= rows_to_take:
                    sample_for_scaler.append(df[numeric_cols].copy())
                    current_sample_count += df.shape[0]
                else:
                    sample_for_scaler.append(df[numeric_cols].sample(n=rows_to_take, random_state=42).copy())
                    current_sample_count += rows_to_take

    # compute skew metrics and finalize heavy-tail selection
    selected_log_cols = []
    for col in heavy_candidates:
        vals = skew_map.get(col, [])
        if not vals:
            continue
        try:
            svals = pd.Series(vals).dropna()
            skew_val = float(svals.skew())
            p50 = float(percentiles_map[col].get("50", 0.0))
            p99 = float(percentiles_map[col].get(clip_hi_pct, p50))
            ratio = float("inf") if p50 == 0 else p99 / p50
            if skew_val >= float(cfg.get("heavytail_skew_threshold", 1.0)) and ratio >= float(cfg.get("heavytail_ratio_threshold", 10.0)):
                # further check: values must be non-negative or mostly positive
                # we require at least min_unique_for_log unique positive values
                col_values_sample = svals
                pos_unique = col_values_sample[col_values_sample > 0].nunique() if not col_values_sample.empty else 0
                if pos_unique >= int(cfg.get("min_unique_for_log", 3)):
                    selected_log_cols.append(col)
        except Exception as e:
            print("Error computing skew for", col, e)

    print("Columns selected for log1p:", selected_log_cols)
    transforms_meta["log1p_columns"] = selected_log_cols

    # Fit RobustScaler on combined sample
    if sample_for_scaler:
        sample_df = pd.concat(sample_for_scaler, ignore_index=True).astype(float)
        # apply clipping to sample before fitting
        for col in numeric_cols:
            if col not in sample_df.columns:
                continue
            cmap = percentiles_map.get(col, {})
            lo = cmap.get(clip_lo_pct)
            hi = cmap.get(clip_hi_pct)
            if lo is not None and hi is not None:
                sample_df[col] = sample_df[col].clip(lower=float(lo), upper=float(hi))
            # if selected for log1p, apply
            if col in selected_log_cols:
                sample_df[col] = np.log1p(sample_df[col].clip(lower=0))
        # replace inf/nan
        sample_df = sample_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        # fit scaler
        scaler = RobustScaler()
        scaler.fit(sample_df.values)
        save_joblib(scaler, scaler_path)
        print("Saved RobustScaler to", scaler_path)
    else:
        print("Warning: no data sampled to fit scaler; skipping scaler save.")
        scaler = None

    # Now apply transforms (clipping, log1p) and scale -> write parquet parts
    part_idx = 0
    for src in data_sources:
        print("Transforming", src)
        if src.suffix == ".parquet":
            df_iter = [pd.read_parquet(src)]
        else:
            df_iter = pd.read_csv(src, chunksize=int(cfg.get("chunk_size", 100000)))
        for df in df_iter:
            # ensure numeric columns exist
            for col in numeric_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    cmap = percentiles_map.get(col, {})
                    lo = cmap.get(clip_lo_pct)
                    hi = cmap.get(clip_hi_pct)
                    if lo is not None and hi is not None:
                        df[col] = df[col].clip(lower=float(lo), upper=float(hi))
                    # apply log1p if selected
                    if col in selected_log_cols:
                        # ensure non-negative before log
                        df[col] = df[col].clip(lower=0)
                        df[col] = np.log1p(df[col])
            # scale numeric columns if scaler present
            if scaler is not None:
                # fill NA with 0 before transform
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                # select numeric columns in df preserving order
                cols_present = [c for c in numeric_cols if c in df.columns]
                if cols_present:
                    arr = df[cols_present].values.astype(float)
                    scaled = scaler.transform(arr)
                    df[cols_present] = scaled
            # write parquet part
            out_file = processed_dir / f"{cfg.get('transformed_prefix','transformed')}_part{part_idx}.parquet"
            df.to_parquet(out_file, index=False)
            print("Wrote", out_file)
            part_idx += 1

    # Save transforms metadata
    save_json(transforms_meta, transforms_path)
    print("Saved transforms metadata to", transforms_path)

if __name__ == "__main__":
    main()
