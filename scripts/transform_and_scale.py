# scripts/transform_and_scale.py
"""
Apply clipping, heavy-tail detection + log1p, fit RobustScaler,
and write transformed parquet files.

Outputs:
 - preprocessor/transforms.json
 - models/artifacts/robust_scaler.joblib
 - data/processed/transformed_part*.parquet
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
import joblib
from sklearn.preprocessing import RobustScaler
from preprocessor.helpers import load_config, load_json, save_json, save_joblib


def safe_apply_log1p(s: pd.Series):
    """Return log1p(s) only if all values are >= 0."""
    if (s.dropna() < 0).any():
        return None
    return np.log1p(s)


def main():

    # ------------- Load config -------------
    cfg = load_config(ROOT / "preprocessor" / "config.yaml")

    raw_dir       = Path(cfg["raw_data_dir"]).resolve()
    processed_dir = Path(cfg["processed_dir"]).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    percentiles_path = ROOT / cfg.get("percentiles_path", "preprocessor/percentiles.json")
    transforms_path  = ROOT / cfg.get("transforms_path",  "preprocessor/transforms.json")
    scaler_path      = ROOT / cfg.get("scaler_artifact", "models/artifacts/robust_scaler.joblib")

    if not percentiles_path.exists():
        raise SystemExit("Percentiles file missing. Run compute_percentiles.py first.")

    # ------------- Load & CLEAN percentile keys -------------
    raw_per = load_json(percentiles_path)
    percentiles_map = {}
    for col, pctdict in raw_per.get("percentiles", {}).items():
        percentiles_map[col.strip()] = pctdict

    numeric_cols = list(percentiles_map.keys())

    clip_lo_pct = str(int(cfg.get("clip_lower_pct", 0.01) * 100))
    clip_hi_pct = str(int(cfg.get("clip_upper_pct", 0.99) * 100))

    transforms_meta = {
        "log1p_columns": [],
        "clipping": {"lower_pct": clip_lo_pct, "upper_pct": clip_hi_pct}
    }

    # ------------- Heavy-tail detection by percentile ratio -------------
    ratio_threshold = float(cfg.get("heavytail_ratio_threshold", 10.0))
    heavy_candidates = []

    for col in numeric_cols:
        pct = percentiles_map[col]
        if clip_hi_pct not in pct or "50" not in pct:
            continue

        p99 = float(pct[clip_hi_pct])
        p50 = float(pct["50"])
        ratio = float("inf") if p50 == 0 else p99 / p50

        if ratio >= ratio_threshold:
            heavy_candidates.append(col)

    print("Heavy-tail candidates by ratio:", heavy_candidates)

    # ------------- Gather skew samples & scaler samples -------------
    skew_map = {}
    sample_rows_limit = 200000
    current_sample = 0
    sample_for_scaler = []

    data_sources = sorted(raw_dir.glob("*.csv")) + sorted(processed_dir.glob("*.parquet"))
    if not data_sources:
        raise SystemExit("No data files found.")

    chunk_size = int(cfg.get("chunk_size", 100000))

    for src in data_sources:
        print("Scanning for skew from", src)

        # load df blocks
        if src.suffix == ".parquet":
            blocks = [pd.read_parquet(src)]
        else:
            blocks = pd.read_csv(src, chunksize=chunk_size)

        for df in blocks:
            df.columns = [c.strip() for c in df.columns]

            # numeric conversion
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                    # SKew sampling
                    s = df[col].dropna()
                    if len(s) > 0:
                        prev = skew_map.get(col, [])
                        take = min(1000, len(s))
                        prev.extend(s.sample(n=take, random_state=42).tolist())
                        if len(prev) > 5000:
                            prev = prev[-5000:]
                        skew_map[col] = prev

            # ----- sample for scaler -----
            if current_sample < sample_rows_limit:
                rows_to_take = min(2000, sample_rows_limit - current_sample)
                subset = df[numeric_cols].dropna()

                if len(subset) == 0:
                    continue

                if len(subset) <= rows_to_take:
                    sample_for_scaler.append(subset.copy())
                    current_sample += len(subset)
                else:
                    part = subset.sample(n=rows_to_take, random_state=42).copy()
                    sample_for_scaler.append(part)
                    current_sample += rows_to_take

    # ------------- Decide final log1p columns -------------
    final_log_cols = []
    skew_threshold = float(cfg.get("heavytail_skew_threshold", 1.0))
    min_unique = int(cfg.get("min_unique_for_log", 3))

    for col in heavy_candidates:
        vals = skew_map.get(col, [])
        if not vals:
            continue

        svals = pd.Series(vals).dropna()
        skew_val = svals.skew()

        pct = percentiles_map[col]
        p99 = float(pct.get(clip_hi_pct, 0))
        p50 = float(pct.get("50", 0))
        ratio = float("inf") if p50 == 0 else p99 / p50

        pos_unique = svals[svals > 0].nunique()

        if skew_val >= skew_threshold and ratio >= ratio_threshold and pos_unique >= min_unique:
            final_log_cols.append(col)

    transforms_meta["log1p_columns"] = final_log_cols
    print("Columns selected for log1p:", final_log_cols)

    # ------------- Fit RobustScaler -------------
    scaler = None
    if sample_for_scaler:
        df_s = pd.concat(sample_for_scaler, ignore_index=True)
        df_s.columns = [c.strip() for c in df_s.columns]

        # apply clipping + log1p before fitting
        for col in numeric_cols:
            pct = percentiles_map[col]
            lo = pct.get(clip_lo_pct)
            hi = pct.get(clip_hi_pct)

            if lo is not None and hi is not None:
                df_s[col] = df_s[col].clip(float(lo), float(hi))

            if col in final_log_cols:
                df_s[col] = np.log1p(df_s[col].clip(lower=0))

        df_s = df_s.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

        scaler = RobustScaler()
        scaler.fit(df_s.values)
        save_joblib(scaler, scaler_path)
        print("Saved scaler →", scaler_path)

    # ------------- Apply transforms + write parquet -------------
    part = 0

    for src in data_sources:
        print("Transforming", src)

        if src.suffix == ".parquet":
            blocks = [pd.read_parquet(src)]
        else:
            blocks = pd.read_csv(src, chunksize=chunk_size)

        for df in blocks:
            df.columns = [c.strip() for c in df.columns]

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                    pct = percentiles_map[col]
                    lo, hi = pct.get(clip_lo_pct), pct.get(clip_hi_pct)
                    if lo is not None and hi is not None:
                        df[col] = df[col].clip(float(lo), float(hi))

                    if col in final_log_cols:
                        df[col] = np.log1p(df[col].clip(lower=0))

            # scaling
            if scaler is not None:
                present = [c for c in numeric_cols if c in df.columns]
                arr = df[present].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
                df[present] = scaler.transform(arr.values)

            out_file = processed_dir / f"transformed_part{part}.parquet"
            df.to_parquet(out_file, index=False)
            print("Wrote", out_file)
            part += 1

    # ------------- Save metadata -------------
    save_json(transforms_meta, transforms_path)
    print("Saved transforms →", transforms_path)


if __name__ == "__main__":
    main()
