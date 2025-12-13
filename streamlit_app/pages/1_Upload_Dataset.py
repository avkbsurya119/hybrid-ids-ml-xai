import streamlit as st
import pandas as pd

st.title("📂 Upload Dataset")

st.markdown("""
Upload a **raw network traffic CSV** file.  
The file will be processed in **batches** for safe inference.
""")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    with st.spinner("Reading CSV..."):
        df = pd.read_csv(uploaded_file)

    st.success(f"Loaded {len(df):,} rows")

    # ✅ SINGLE SOURCE OF TRUTH
    st.session_state["uploaded_df"] = df

    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Dataset Summary")
    st.write("Columns:", len(df.columns))
    st.write("Rows:", len(df))

    if "Label" in df.columns or " Label" in df.columns:
        st.warning("⚠️ Label column detected (ground truth available)")
    else:
        st.info("ℹ️ No label column (pure inference mode)")
