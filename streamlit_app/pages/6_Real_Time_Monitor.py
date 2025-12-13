import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd

from pipeline.csv_inference_2stage import run_2stage_csv_inference
from api.model_loader import model_bundle

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Real-Time Monitor", layout="wide")
st.title("🔴 Real-Time Intrusion Monitor")

st.markdown("""
**Live-style monitoring using streaming flow batches**  
(SOC-style network telemetry simulation)
""")

# -------------------------------------------------
# Safety check
# -------------------------------------------------
if "uploaded_df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["uploaded_df"]

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.header("⚙️ Monitor Controls")

BATCH_SIZE = st.sidebar.selectbox(
    "Batch size",
    [50, 100, 200, 500],
    index=1
)

DELAY = st.sidebar.slider(
    "Update delay (seconds)",
    0.5, 3.0, 1.0, 0.5
)

start_btn = st.sidebar.button("▶ Start Monitor")

# -------------------------------------------------
# Session init
# -------------------------------------------------
if "rt_pointer" not in st.session_state:
    st.session_state.rt_pointer = 0

if "rt_log" not in st.session_state:
    st.session_state.rt_log = []

# -------------------------------------------------
# 🔒 STATIC LAYOUT (CREATED ONCE)
# -------------------------------------------------

# ---- KPI Row ----
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

kpi_total = kpi_col1.empty()
kpi_attack = kpi_col2.empty()
kpi_benign = kpi_col3.empty()
kpi_anomaly = kpi_col4.empty()

st.divider()

# ---- Graphs ----
graph_col1, graph_col2 = st.columns(2)

attack_ts_placeholder = graph_col1.empty()
attack_dist_placeholder = graph_col2.empty()

st.divider()

# ---- Live Table ----
st.subheader("🧾 Live Flow Stream")
table_placeholder = st.empty()

# -------------------------------------------------
# ▶ RUN MONITOR
# -------------------------------------------------
if start_btn:

    while st.session_state.rt_pointer < len(df):

        batch = df.iloc[
            st.session_state.rt_pointer :
            st.session_state.rt_pointer + BATCH_SIZE
        ]

        out = run_2stage_csv_inference(
            df_raw=batch,
            bin_model=model_bundle.binary_model,
            atk_model=model_bundle.attack_model,
            atk_encoder=model_bundle.attack_label_encoder,
            scaler=model_bundle.scaler,
            transforms=model_bundle.transforms,
            bin_features=model_bundle.binary_features,
            atk_features=model_bundle.attack_features,
            autoencoder=model_bundle.autoencoder,
            ae_threshold=model_bundle.autoencoder_threshold
        )

        st.session_state.rt_log.append(out)
        st.session_state.rt_pointer += BATCH_SIZE

        live_df = pd.concat(st.session_state.rt_log, ignore_index=True)

        # ---------------- KPIs ----------------
        total = len(live_df)
        attacks = (live_df["Binary_Route"] == "ATTACK").sum()
        benign = total - attacks
        anomalies = live_df["Anomaly"].sum()

        kpi_total.metric("Total Flows", total)
        kpi_attack.metric("Attacks", attacks)
        kpi_benign.metric("Benign", benign)
        kpi_anomaly.metric("Anomalies", anomalies)

        # ---------------- Attack rate (fixed graph) ----------------
        ts = (
            live_df["Binary_Route"]
            .eq("ATTACK")
            .groupby(live_df.index // BATCH_SIZE)
            .sum()
        )

        attack_ts_placeholder.line_chart(ts)

        # ---------------- Attack distribution ----------------
        atk_dist = (
            live_df[live_df["Binary_Route"] == "ATTACK"]
            ["Final_Prediction"]
            .value_counts()
        )

        if not atk_dist.empty:
            attack_dist_placeholder.bar_chart(atk_dist)

        # ---------------- Log-style table (append downward) ----------------
        table_placeholder.dataframe(
            live_df.tail(100),   # show last N rows only
            use_container_width=True,
            height=600
        )

        time.sleep(DELAY)
