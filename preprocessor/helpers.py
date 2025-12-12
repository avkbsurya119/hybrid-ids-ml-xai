# preprocessor/helpers.py
"""
Small helper utilities used by preprocessing scripts.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

def load_config(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    p = Path(path)
    with open(p, "r") as f:
        return json.load(f)

def save_joblib(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def is_numeric_series(s: pd.Series):
    return pd.api.types.is_numeric_dtype(s)
