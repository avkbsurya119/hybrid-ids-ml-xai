import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd

from pipeline.csv_inference_2stage import run_2stage_csv_inference
from api.model_loader import model_bundle

st.title("📊 Inference Results")

# -----------------------------
# Safety check
# -----------------------------
if "uploaded_df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["uploaded_df"]
st.write(f"Total rows: **{len(df):,}**")

# -----------------------------
# Batch settings
# -----------------------------
BATCH_SIZE = st.selectbox(
    "Batch size",
    options=[500, 1000, 5000, 10000],
    index=1
)

run_btn = st.button("🚀 Run Inference")

# -----------------------------
# Run inference
# -----------------------------
if run_btn:
    # Calculate total batches
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    st.info(f"Processing {len(df):,} rows in {total_batches} batch(es) of {BATCH_SIZE:,} rows each...")

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    batch_num = 0

    for start in range(0, len(df), BATCH_SIZE):
        batch_num += 1
        batch = df.iloc[start:start + BATCH_SIZE]

        status_text.text(f"⏳ Processing batch {batch_num}/{total_batches} ({start:,} to {min(start + BATCH_SIZE, len(df)):,})...")

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

        results.append(out)

        # Update progress
        progress_bar.progress(batch_num / total_batches)

    status_text.text("✅ Concatenating results...")
    final_df = pd.concat(results, ignore_index=True)

    progress_bar.empty()
    status_text.empty()

    # Attach ground truth if available
    if "Label" in df.columns or " Label" in df.columns:
        label_col = "Label" if "Label" in df.columns else " Label"
        final_df.insert(0, "Actual", df[label_col].values)

    # ✅ SINGLE KEY
    st.session_state["inference_df"] = final_df

    st.success("Inference completed!")

# -----------------------------
# Always show results if present
# -----------------------------
if "inference_df" in st.session_state:
    result_df = st.session_state["inference_df"]

    # Summary Statistics
    st.subheader("📈 Prediction Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        benign_count = (result_df["Final_Prediction"] == "BENIGN").sum()
        st.metric("BENIGN", f"{benign_count:,}",
                  delta=f"{benign_count/len(result_df)*100:.1f}%")

    with col2:
        attack_count = (result_df["Final_Prediction"] != "BENIGN").sum()
        st.metric("ATTACK", f"{attack_count:,}",
                  delta=f"{attack_count/len(result_df)*100:.1f}%")

    with col3:
        anomaly_count = result_df["Anomaly"].sum() if "Anomaly" in result_df.columns else 0
        st.metric("Anomalies", f"{anomaly_count:,}")

    # Attack type breakdown
    if attack_count > 0:
        attack_types = result_df[result_df["Final_Prediction"] != "BENIGN"]["Final_Prediction"].value_counts()
        st.write("**Attack Type Distribution:**")
        for attack_type, count in attack_types.items():
            st.write(f"- {attack_type}: {count:,} ({count/len(result_df)*100:.1f}%)")

    st.subheader("📋 Inference Results Table")

    st.dataframe(
        st.session_state["inference_df"],
        use_container_width=True,
        height=600
    )

    st.download_button(
        "⬇️ Download Results CSV",
        st.session_state["inference_df"].to_csv(index=False),
        file_name="ids_inference_results.csv",
        mime="text/csv"
    )
