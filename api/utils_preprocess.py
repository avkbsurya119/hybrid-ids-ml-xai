import pandas as pd
import numpy as np

def preprocess_input(records, feature_order, transforms, scaler):
    df = pd.DataFrame(records)

    # Apply training-time transforms
    for col, rule in transforms.items():
        if col not in df.columns:
            df[col] = 0
        if rule == "log1p":
            df[col] = np.log1p(df[col].clip(lower=0))

    # Enforce feature order
    df = df.reindex(columns=feature_order, fill_value=0)

    # Scale
    X = scaler.transform(df.values)
    return X
