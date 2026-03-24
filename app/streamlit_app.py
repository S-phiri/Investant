"""
Investant — Streamlit app.
Run with: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# Add project root so src is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Investant", page_icon="📊", layout="wide")
st.title("Investant")
st.markdown("Data pipeline, analysis, and LLM-assisted insights.")

# Placeholder for data, analysis, and LLM sections
tab1, tab2, tab3 = st.tabs(["Data", "Analysis", "LLM Assistant"])
with tab1:
    st.info("Connect to data pipeline and raw/processed data here.")
with tab2:
    st.info("Exploratory analysis and ML results here.")
with tab3:
    st.info("LLM assistant integration here.")
