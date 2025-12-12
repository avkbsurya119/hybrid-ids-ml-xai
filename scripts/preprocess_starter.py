"""
Starter preprocessing script (minimal).
Purpose: demonstrate where to load raw CSV(s) in chunks
and write a single parquet output to data/processed/.

Comments and meaningful sections are added so commits are useful.
"""

from pathlib import Path
import pandas as pd
import yaml
pd.set_option('future.no_silent_downcasting', True)
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
    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.fillna(0)
    df = df.infer_objects()   # convert object columns to better dtypes where possible

    return df

if __name__ == "__main__":
    cfg = load_config(cfg_path)

    # Resolve raw and processed directories relative to project root (ROOT)
    # Note: config should use paths relative to project root (e.g., "data/raw"),
    # not paths with .. which move outside the project folder.
    raw_dir = (ROOT / cfg["raw_data_dir"]).resolve()
    out_dir = (ROOT / cfg["processed_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Debug info to help diagnose path mismatch problems
    print("DEBUG: Project ROOT resolved to:", ROOT)
    print("DEBUG: Config raw_data_dir:", cfg["raw_data_dir"])
    print("DEBUG: Resolved raw_dir (ROOT / cfg):", raw_dir)
    print("DEBUG: Resolved out_dir (ROOT / cfg):", out_dir)

    # Fallback: if the resolved raw_dir doesn't exist, try interpreting the config
    # path as an absolute/relative system path (maybe you ran script from another CWD).
    if not raw_dir.exists():
        alt = Path(cfg["raw_data_dir"]).expanduser().resolve()
        print("DEBUG: Primary raw_dir does not exist. Trying alternative:", alt)
        if alt.exists():
            raw_dir = alt
            print("DEBUG: Using alternative raw_dir:", raw_dir)
        else:
            print("No CSVs found in", raw_dir)
            print("No alternative path found at", alt)
            # list actual files in project data/raw for extra help:
            sample_dir = ROOT / "data" / "raw"
            print("Listing expected folder:", sample_dir)
            try:
                files = list(sample_dir.glob("*"))
                if files:
                    print("Files present in", sample_dir, "->")
                    for f in files[:20]:
                        print("  -", f.name)
                else:
                    print("Directory exists but is empty or not present.")
            except Exception as e:
                print("Could not list sample_dir:", e)
            # Exit early because there's nothing to process
            raise SystemExit(1)

    # Process all CSVs in the final raw directory
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
