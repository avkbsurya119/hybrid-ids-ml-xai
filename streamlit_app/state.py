import streamlit as st

def init_state():
    defaults = {
        "csv_path": None,
        "batch_size": 5000,
        "results_df": None,
        "inference_done": False
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
