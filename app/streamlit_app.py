import streamlit as st

st.set_page_config(
    page_title="Hybrid NIDS Dashboard",
    layout="wide"
)

st.title("🚨 Hybrid Network Intrusion Detection System")
st.markdown("""
**Models**: LightGBM + Autoencoder  
**Explainability**: SHAP  
**Backend**: FastAPI  
""")
