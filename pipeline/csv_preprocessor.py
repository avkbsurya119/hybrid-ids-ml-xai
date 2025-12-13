import pandas as pd
import numpy as np


def preprocess_csv(
    df: pd.DataFrame,
    feature_order: list,
    transforms: dict,
    scaler
):
    """
    Convert RAW CSV dataframe → model-ready numpy array.
    - Applies SAME transforms as training
    - Enforces feature order
    - Cleans infinities / extreme values
    - Scales exactly ONCE
    """

    df = df.copy()

    # ---------------- Strip leading/trailing spaces from column names ----------------
    df.columns = df.columns.str.strip()

    # ---------------- Drop label ----------------
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # ---------------- Apply transforms ----------------
    log1p_cols = transforms.get("log1p_columns", [])
    for col in log1p_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # ---------------- Enforce feature order ----------------
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_order]

    # ---------------- HARD SAFETY CLEAN ----------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # clip extreme values (VERY IMPORTANT)
    df = df.clip(lower=-1e9, upper=1e9)

    # force numeric dtype
    df = df.astype(np.float32)

    # ---------------- Scale ONCE ----------------
    X = scaler.transform(df.values)

    return X
