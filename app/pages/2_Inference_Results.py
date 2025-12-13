import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Inference Results", layout="wide")

st.title("📊 Inference Results")

# -------------------------
# CHECK UPLOADED DATA
# -------------------------
if "uploaded_df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["uploaded_df"]

# -------------------------
# JSON SAFE CLEANING
# -------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0.0)
st.session_state["uploaded_df"] = df

st.info(f"Running inference on **{len(df)} flows**")

# -------------------------
# CONFIG
# -------------------------
BATCH_SIZE = 1000
API_URL = "http://127.0.0.1:8000/predict_batch"

# -------------------------
# RUN INFERENCE
# -------------------------
if st.button("🚀 Run Inference"):

    results = []
    progress = st.progress(0.0)

    for i in range(0, len(df), BATCH_SIZE):

        batch_df = df.iloc[i:i + BATCH_SIZE]

        payload = {
            "data": batch_df.to_dict(orient="records")
        }

        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()

        batch_results = resp.json()["results"]

        for idx, r in enumerate(batch_results):
            results.append({
                "Flow_ID": i + idx,
                "Prediction": r["label"],
                "Anomaly": r["is_anomaly"],
                "Anomaly_Score": r["anomaly_score"],
                "Confidence": max(r["probabilities"].values())
            })

        progress.progress(min((i + BATCH_SIZE) / len(df), 1.0))

    st.session_state["inference_results"] = pd.DataFrame(results)
    st.success("✅ Inference completed")

# -------------------------
# DISPLAY RESULTS
# -------------------------
if "inference_results" in st.session_state:

    res_df = st.session_state["inference_results"]

    # -------------------------
    # METRICS
    # -------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows", len(res_df))
    col2.metric("Anomalies", int(res_df["Anomaly"].sum()))
    col3.metric("Normal", int((~res_df["Anomaly"]).sum()))

    # -------------------------
    # DISTRIBUTION
    # -------------------------
    st.subheader("📈 Prediction Distribution")
    pred_counts = res_df["Prediction"].value_counts()
    st.bar_chart(pred_counts)
    st.write(pred_counts)

    # -------------------------
    # TABLE
    # -------------------------
    st.subheader("🧾 Inference Results Table")
    st.dataframe(res_df, use_container_width=True)

    # -------------------------
    # DOWNLOAD
    # -------------------------
    st.download_button(
        "⬇️ Download Results CSV",
        res_df.to_csv(index=False),
        file_name="nids_inference_results.csv",
        mime="text/csv"
    )
