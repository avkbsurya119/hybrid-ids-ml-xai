import pandas as pd
import numpy as np

def preprocess_input(data: dict, bundle):
    df = pd.DataFrame([data])

    # Apply transformations from transforms.json
    for col, rule in bundle.transforms.items():
        if col not in df.columns:
            df[col] = 0

        if rule == "log1p":
            df[col] = df[col].apply(lambda x: np.log1p(x if x > 0 else 0))
        elif rule == "none":
            pass

    # Ensure column order matches LightGBM
    model_cols = bundle.model.feature_name()

    # Add missing columns
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[model_cols]

    # Convert to numpy
    X = df.values.astype(float)

    # Scale
    X = bundle.scaler.transform(X)

    # If scaler complains about feature names, ignore
    return X
