import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("📤 Upload Network Flow Dataset")

st.markdown("""
Upload a **CSV file** containing network flow features  
(Format must match the trained model input schema)
""")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("✅ Dataset uploaded successfully")

        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.info(f"""
        **Rows:** {df.shape[0]}  
        **Columns:** {df.shape[1]}
        """)

        # Store dataset globally
        st.session_state["uploaded_df"] = df

    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
