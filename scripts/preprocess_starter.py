"""
Starter preprocessing script (minimal).
Purpose: demonstrate where to load raw CSV(s) in chunks
and write a single parquet output to data/processed/.

Comments and meaningful sections are added so commits are useful.
"""

from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg_path = ROOT / "preprocessor" / "config.yaml"

def load_config(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def preprocess_chunk(df):
    # Example simple cleaning:
    #  - strip column names
    #  - drop listed columns if present
    df.columns = [c.strip() for c in df.columns]
    drop_cols = cfg.get("drop_columns", [])
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    # Replace inf with nan, then fillna with 0 (placeholder)
    df = df.replace([float("inf"), -float("inf")], pd.NA).fillna(0)
    return df

if __name__ == "__main__":
    cfg = load_config(cfg_path)
    raw_dir = Path(cfg["raw_data_dir"]).resolve()
    out_dir = Path(cfg["processed_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Process all CSVs in raw directory
    csvs = list(raw_dir.glob("*.csv"))
    if not csvs:
        print("No CSVs found in", raw_dir)
    else:
        # Example: process first CSV only and write parquet
        first = csvs[0]
        print("Processing", first)
        chunks = pd.read_csv(first, chunksize=cfg.get("chunk_size", 100000))
        out_parts = []
        for i, ch in enumerate(chunks):
            chp = preprocess_chunk(ch)
            part_file = out_dir / f"{first.stem}_part{i}.parquet"
            chp.to_parquet(part_file, index=False)
            print("Wrote", part_file)
