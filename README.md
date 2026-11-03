# Investant

Groups JSE and US stocks into behaviour regimes from recent returns and volatility, then packages the result as structured JSON for an LLM to explain in plain language.

The goal is an assistant that tells a retail investor "these names have been moving together, and here's what that means for your risk appetite", grounded in numbers rather than vibes. **Educational only, not investment advice.**

## How it works

```
Alpha Vantage (US) ─┐
                    ├─► cached OHLCV ─► aligned closes ─► daily returns ─► features ─► k-means + PCA ─► payload.json ─► LLM
yfinance (JSE) ─────┘
```

1. **Ingest** (`src/api_client.py`): daily prices for 8 names. Alpha Vantage for US listings (IBM, Apple, Microsoft), yfinance for JSE (MTN, Naspers, FirstRand, Shoprite, Anglo American), because free US APIs cover JSE tickers poorly. Results are cached to `data/raw/` to respect the 5-calls-per-minute limit.
2. **Process** (`src/data_processor.py`): align closes to the dates every name traded, compute daily % returns, and build a 20-day snapshot of mean return and volatility per stock.
3. **Model** (`notebooks/02_analysis_ml.ipynb`): standardise features, run k-means (k=3), label clusters by their centres (e.g. `weaker_trend_high_volatility`), and plot a PCA projection.
4. **Package** (`build_llm_payload`): export per-stock metrics, cluster labels and a narrative blurb to `data/payload.json`, plus a risk-posture mapping (conservative ⊆ balanced ⊆ aggressive).

### Sample run (as of 18 Mar 2026)

98 aligned trading days across 8 tickers, 97 return rows, k=3, PCA variance split roughly 50/50 across the first two components.

## Stack

Python · pandas · scikit-learn · matplotlib/seaborn · Streamlit · dbt (scaffolded)

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn yfinance requests python-dotenv streamlit
echo "ALPHA_VANTAGE_KEY=your_key" > .env
jupyter notebook notebooks/          # run 01 then 02
streamlit run app/streamlit_app.py
```

## Roadmap

- [ ] LLM layer (`src/llm_assistant.py`): answer questions over `payload.json` with the Anthropic API, citing the metrics it used
- [ ] Wire analysis and assistant into the Streamlit tabs
- [ ] Move transforms into dbt models (staging prices → returns → features)
- [ ] Tests for the processing functions
