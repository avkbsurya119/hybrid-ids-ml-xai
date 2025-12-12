import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
"""
scripts/generate_feature_report.py

Generates per-column statistics from transformed data (data/processed/*.parquet).
Outputs:
 - docs/feature_report.csv
 - docs/feature_report.md (human summary)
 - docs/top_correlations.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from preprocessor.helpers import load_config, save_json
import math

ROOT = Path(__file__).resolve().parents[1]
cfg = load_config(ROOT / "preprocessor" / "config.yaml")

processed_dir = Path(cfg["processed_dir"]).resolve()
out_dir = ROOT / "docs"
out_dir.mkdir(parents=True, exist_ok=True)

# gather parquet parts
parts = sorted(processed_dir.glob("*.parquet"))
if not parts:
    raise SystemExit(f"No processed parquet files found in {processed_dir}. Run transform_and_scale.py first.")

# read up to N rows for the report to keep memory low
max_rows = 200000
frames = []
rows = 0
for p in parts:
    if rows >= max_rows:
        break
    df = pd.read_parquet(p)
    take = min(len(df), max_rows - rows)
    if take < len(df):
        df = df.sample(n=take, random_state=42)
    frames.append(df)
    rows += take

df = pd.concat(frames, ignore_index=True)
print("Loaded rows for report:", df.shape)

# build report
report = []
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
for col in df.columns:
    s = df[col]
    dtype = str(s.dtype)
    miss_pct = s.isna().mean()
    unique = s.nunique(dropna=True)
    entry = {
        "feature": col,
        "dtype": dtype,
        "missing_pct": float(miss_pct),
        "unique_count": int(unique)
    }
    if col in numeric_cols:
        ser = s.dropna().astype(float)
        entry.update({
            "mean": float(ser.mean()) if len(ser) else None,
            "std": float(ser.std()) if len(ser) else None,
            "skew": float(ser.skew()) if len(ser) else None,
            "kurtosis": float(ser.kurtosis()) if len(ser) else None,
            "p25": float(np.nanpercentile(ser, 25)) if len(ser) else None,
            "p50": float(np.nanpercentile(ser, 50)) if len(ser) else None,
            "p75": float(np.nanpercentile(ser, 75)) if len(ser) else None,
            "p99": float(np.nanpercentile(ser, 99)) if len(ser) else None,
        })
    report.append(entry)

rdf = pd.DataFrame(report).sort_values(by="missing_pct", ascending=False)
rdf.to_csv(out_dir / "feature_report.csv", index=False)
print("Wrote docs/feature_report.csv")

# write a short markdown summary
md_lines = []
md_lines.append("# Feature Report Summary")
md_lines.append("")
md_lines.append(f"Generated from {len(parts)} parquet parts. Rows sampled: {rows}.")
md_lines.append("")
md_lines.append("## Top 10 features by missing percentage")
top_missing = rdf.sort_values("missing_pct", ascending=False).head(10)
md_lines.append(top_missing[["feature", "missing_pct", "dtype"]].to_markdown(index=False))
md_lines.append("")
md_lines.append("## Numeric feature summary (top skewed)")
numeric_df = rdf.dropna(subset=["skew"]).sort_values("skew", ascending=False).head(10)
md_lines.append(numeric_df[["feature","skew","kurtosis","mean","std","p99"]].to_markdown(index=False))
md_lines.append("")
# save md
with open(out_dir / "feature_report.md", "w", encoding="utf8") as f:
    f.write("\n".join(md_lines))
print("Wrote docs/feature_report.md")

# compute correlation matrix for numeric columns and save top absolute pairs
corr = df[numeric_cols].corr().abs()
# take upper triangle and unstack
tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pairs = tri.unstack().dropna().sort_values(ascending=False)
pairs = pairs.reset_index()
pairs.columns = ["feature_a","feature_b","abs_corr"]
pairs.to_csv(out_dir / "top_correlations.csv", index=False)
print("Wrote docs/top_correlations.csv")
