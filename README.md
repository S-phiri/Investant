# Investant

Data pipeline, analysis, and LLM-assisted insights.

## Structure

```
investant/
├── data/raw/          # Raw data files
├── notebooks/         # Jupyter notebooks (pipeline, analysis, LLM)
├── app/               # Streamlit app
├── src/               # Shared code (API, processor, LLM)
├── .env               # Secrets (not committed)
├── .gitignore
└── README.md
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install pandas streamlit python-dotenv
   ```

2. Copy `.env` and add your API keys if needed.

## Run

- **Streamlit app:** `streamlit run app/streamlit_app.py`
- **Notebooks:** Open `notebooks/*.ipynb` in Jupyter or VS Code.

## Notebooks

- `01_data_pipeline.ipynb` — Load and prepare raw data
- `02_analysis_ml.ipynb` — Exploratory analysis and ML
- `03_llm_integration.ipynb` — LLM assistant integration
