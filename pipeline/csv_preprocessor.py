# pipeline/csv_preprocessor.py

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
    - Scales exactly ONCE
    """

    df = df.copy()

    # Drop label if present
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # Apply feature transforms
    for col, rule in transforms.items():
        if col not in df.columns:
            df[col] = 0

        if rule == "log1p":
            df[col] = np.log1p(df[col].clip(lower=0))

    # Enforce training feature order
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_order]

    # Scale ONCE
    X = scaler.transform(df.values)

    return X
