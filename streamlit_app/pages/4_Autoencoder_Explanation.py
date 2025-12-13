import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import torch

from api.model_loader import model_bundle
from pipeline.csv_preprocessor import preprocess_csv

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Autoencoder Explanation", layout="wide")
st.title("🧠 Autoencoder Anomaly Explanation")

# -------------------------------------------------
# Safety checks
# -------------------------------------------------
if (
    "uploaded_df" not in st.session_state
    or "inference_df" not in st.session_state
):
    st.warning("Please upload a dataset and run inference first.")
    st.stop()

if model_bundle.autoencoder is None:
    st.error("Autoencoder model not available.")
    st.stop()

df = st.session_state["uploaded_df"]
results = st.session_state["inference_df"]

# -------------------------------------------------
# Select flow
# -------------------------------------------------
st.subheader("Select Flow")

flow_idx = st.number_input(
    "Flow index",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

flow = df.iloc[flow_idx]
prediction = results.iloc[flow_idx]

# -------------------------------------------------
# Show summary
# -------------------------------------------------
st.subheader("Prediction Summary")

c1, c2, c3 = st.columns(3)

c1.metric("Final Prediction", prediction["Final_Prediction"])
c2.metric("Anomaly", "YES" if prediction["Anomaly"] else "NO")
c3.metric(
    "AE Threshold",
    round(model_bundle.autoencoder_threshold, 6)
)

# -------------------------------------------------
# Preprocess flow for AE
# -------------------------------------------------
flow_df = pd.DataFrame([flow])

X_ae = preprocess_csv(
    flow_df,
    model_bundle.attack_features,
    model_bundle.transforms,
    model_bundle.scaler
)

x_tensor = torch.tensor(X_ae, dtype=torch.float32)

# -------------------------------------------------
# Reconstruction
# -------------------------------------------------
with torch.no_grad():
    recon = model_bundle.autoencoder(x_tensor).numpy()

# Feature-wise reconstruction error
feature_errors = (X_ae - recon) ** 2
mse = float(np.mean(feature_errors))

# -------------------------------------------------
# MSE display
# -------------------------------------------------
st.subheader("Reconstruction Error")

c1, c2 = st.columns(2)

c1.metric("Reconstruction MSE", round(mse, 6))
c2.metric(
    "Above Threshold?",
    "YES" if mse > model_bundle.autoencoder_threshold else "NO"
)

# -------------------------------------------------
# Feature contribution table
# -------------------------------------------------
error_df = pd.DataFrame({
    "feature": model_bundle.attack_features,
    "reconstruction_error": feature_errors.flatten()
})

error_df = error_df.sort_values(
    "reconstruction_error",
    ascending=False
)

st.subheader("🔝 Top Contributing Features (AE)")

st.dataframe(
    error_df.head(15),
    use_container_width=True
)

# -------------------------------------------------
# Visualization
# -------------------------------------------------
st.subheader("📊 Feature-wise Reconstruction Error")

st.bar_chart(
    error_df.head(15)
    .set_index("feature")["reconstruction_error"]
)

# -------------------------------------------------
# Explanation text
# -------------------------------------------------
st.info(
    """
**How to interpret this:**
- Autoencoder is trained only on **normal traffic**
- High reconstruction error = unusual behavior
- Features with highest error contribute most to anomaly
"""
)
