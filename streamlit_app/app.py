import streamlit as st

st.set_page_config(
    page_title="Hybrid NIDS",
    layout="wide"
)

# 👇 FORCE SIDEBAR
with st.sidebar:
    st.header("Navigation")
    st.write("Use pages to navigate")

st.title("🛡️ Hybrid Network Intrusion Detection System")

st.markdown("""
Welcome to the **2-Stage Network Intrusion Detection System**.

Use the sidebar to navigate through the pipeline.
""")
