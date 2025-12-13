import sys
from pathlib import Path

# -------------------------------------------------
# Fix imports for Streamlit
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from api.model_loader import model_bundle
from pipeline.csv_preprocessor import preprocess_csv
from streamlit_app.utils.shap_utils import (
    explain_binary_flow,
    explain_attack_flow
)

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Flow Explanation", layout="wide")
st.title("🔍 Flow Explanation Panel")

# -------------------------------------------------
# Safety checks
# -------------------------------------------------
if "uploaded_df" not in st.session_state or "inference_df" not in st.session_state:
    st.warning("Please upload a dataset and run inference first.")
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
# Prediction summary
# -------------------------------------------------
st.subheader("Prediction Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Binary Route", prediction["Binary_Route"])
c2.metric("Final Prediction", prediction["Final_Prediction"])
c3.metric("Confidence", prediction["Confidence"])
c4.metric("Anomaly", "YES" if prediction["Anomaly"] else "NO")

# -------------------------------------------------
# Raw features
# -------------------------------------------------
with st.expander("🔎 Raw Flow Features"):
    st.dataframe(flow.to_frame(name="value"), use_container_width=True)

# -------------------------------------------------
# Background data (shared)
# -------------------------------------------------
bg_df = df.sample(min(200, len(df)), random_state=42)

# -------------------------------------------------
# BINARY SHAP
# -------------------------------------------------
st.subheader("🧠 Binary Model Explanation (Why ATTACK / BENIGN)")

flow_df = pd.DataFrame([flow])

X_bin_single = preprocess_csv(
    flow_df,
    model_bundle.binary_features,
    model_bundle.transforms,
    model_bundle.scaler
)

X_bin_bg = preprocess_csv(
    bg_df,
    model_bundle.binary_features,
    model_bundle.transforms,
    model_bundle.scaler
)

with st.spinner("Computing Binary SHAP explanation..."):
    shap_bin_df = explain_binary_flow(
        model=model_bundle.binary_model,
        X_background=X_bin_bg,
        X_single=X_bin_single,
        feature_names=model_bundle.binary_features
    )

st.markdown("### 🔝 Top Binary Feature Contributions")
st.dataframe(shap_bin_df.head(15), use_container_width=True)

st.bar_chart(
    shap_bin_df.head(15)
    .set_index("feature")["shap_value"]
)

# -------------------------------------------------
# ATTACK SHAP (ONLY IF ATTACK)
# -------------------------------------------------
if prediction["Binary_Route"] == "ATTACK":

    st.subheader("🧠 Attack-Type Explanation (Why this attack)")

    X_atk_single = preprocess_csv(
        flow_df,
        model_bundle.attack_features,
        model_bundle.transforms,
        model_bundle.scaler
    )

    X_atk_bg = preprocess_csv(
        bg_df,
        model_bundle.attack_features,
        model_bundle.transforms,
        model_bundle.scaler
    )

    class_idx = model_bundle.attack_label_encoder.transform(
        [prediction["Final_Prediction"]]
    )[0]

    with st.spinner("Computing Attack SHAP explanation..."):
        shap_atk_df = explain_attack_flow(
            model=model_bundle.attack_model,
            X_background=X_atk_bg,
            X_single=X_atk_single,
            feature_names=model_bundle.attack_features,
            class_index=class_idx
        )

    st.markdown(
        f"### 🔝 Features driving **{prediction['Final_Prediction']}** prediction"
    )

    st.dataframe(shap_atk_df.head(15), use_container_width=True)

    st.bar_chart(
        shap_atk_df.head(15)
        .set_index("feature")["shap_value"]
    )

else:
    st.info("ℹ️ Attack-type SHAP is shown only for ATTACK flows.")
