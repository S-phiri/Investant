"""
Data loading, cleaning, and preparation for analysis.

Investant — data loading and feature prep for analysis, clustering, and LLM summaries.
Universe: CSVs produced by src.api_client (US + JSE). LuSE/Zambia can be added later.
Educational use only — not investment advice.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd


# Default paths relative to project root (works no mata where in project folder you are)
DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_stocks_from_raw_dir(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Load every *.csv in data/raw into {company_name: DataFrame}.
    Expects columns: open, high, low, close, volume and a date index (or first column as dates).
    """
    # Getting the raw directory.
    folder = raw_dir or DATA_RAW

    # Initialising an empty dictionary to store the stocks.
    stocks: dict[str, pd.DataFrame] = {}
    
    # Looping through the CSV files in the raw directory.
    for path in sorted(folder.glob("*.csv")):

        # Getting the name of the stock.
        name = path.stem

        # Reading the CSV file into a DataFrame.
        df = pd.read_csv(path, index_col=0, parse_dates=True)

        # Sorting the DataFrame by the index.
        df.sort_index(inplace=True)
        # Expected columns.
        expected = {"open", "high", "low", "close", "volume"}

        # Checking if the columns are missing.
        # “which required names are not present in this file?”
        missing = expected - set[Any](df.columns.str.lower())
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")
        df.columns = [c.lower() for c in df.columns]
        stocks[name] = df
    return stocks

# Aligning the close prices of the stocks.
def align_closes(stocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a wide DataFrame of close prices, inner-join on dates (only dates where all names trade).
    """

    # Creating a wide DataFrame of close prices.
    closes = pd.DataFrame({name: df["close"] for name, df in stocks.items()})

    # Dropping the rows where any of the columns are NaN.
    return closes.dropna(how="any")

# Calculating the daily returns percentage.
def daily_returns_pct(closes: pd.DataFrame) -> pd.DataFrame:
    """Percentage daily returns from close prices (same columns as closes)."""
    return closes.pct_change() * 100.0

# Calculating the snapshot features for clustering.
def snapshot_features_for_clustering(
    returns_pct: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    One row per stock: features from the last `window` trading days of returns.
    Columns:
      - mean_daily_return_pct
      - volatility_daily_pct (std of daily returns)
    """
    # Getting the last `window` trading days of returns.
    tail = returns_pct.tail(window)
    # Creating a DataFrame with the mean and standard deviation of the daily returns.
    return pd.DataFrame(
        {
            "mean_daily_return_pct": tail.mean(),
            "volatility_daily_pct": tail.std(),
        }
    )

# Labeling the clusters by the centers.
def label_clusters_by_centers(
    feature_df: pd.DataFrame,
    cluster_ids: pd.Series,
    centers: Any,
) -> dict[int, str]:
    """
    Map cluster id -> short human label from feature means (educational / interpretive).
    centers: KMeans.cluster_centers_ (scaled space) — we use cluster_id + rank of mean return
    in the *original* feature_df for simple wording.
    """
    # Copying the feature DataFrame.
    df = feature_df.copy()

    # Adding the cluster IDs to the DataFrame.
    df["cluster"] = cluster_ids.values

    # Initialising an empty dictionary to store the labels.
    labels: dict[int, str] = {}

    # Looping through the unique cluster IDs.
    for cid in sorted(df["cluster"].unique()):
        # Getting the sub DataFrame for the current cluster.
        sub = df[df["cluster"] == cid]


        m = sub["mean_daily_return_pct"].mean()
        v = sub["volatility_daily_pct"].mean()
        if m < 0 and v >= df["volatility_daily_pct"].median():
            labels[cid] = "weaker_trend_high_volatility"
        elif m < 0:
            labels[cid] = "weaker_trend_moderate_volatility"
        elif m >= 0 and v >= df["volatility_daily_pct"].median():
            labels[cid] = "firmer_trend_high_volatility"
        else:
            labels[cid] = "firmer_trend_moderate_volatility"
    return labels
 

# Building the LLM payload.
def build_llm_payload(
    *,
    as_of,
    window: int,
    feature_df: pd.DataFrame,
    cluster_series: pd.Series,
    cluster_label_map: dict[int, str],
    classifier_note: str,
) -> dict[str, Any]:
    """
    Structured dict for Claude / other LLMs: numbers come only from your pipeline.
    classifier_note: short text you computed in the notebook (e.g. accuracy + disclaimer).
    """
    # Initialising an empty dictionary to store the stocks payload.
    stocks_payload: dict[str, Any] = {}

    # Looping through the feature DataFrame.
    for name in feature_df.index:
        # Getting the cluster ID for the current stock.
        cid = int(cluster_series.loc[name])
        # Adding the stock to the stocks payload.
        stocks_payload[name] = {
            "mean_daily_return_pct_window": round(float(feature_df.loc[name, "mean_daily_return_pct"]), 4),
            "volatility_daily_pct_window": round(float(feature_df.loc[name, "volatility_daily_pct"]), 4),
            "kmeans_cluster_id": cid,
            "cluster_label": cluster_label_map.get(cid, "unknown"),
        }

    # Returning the LLM payload.
    return {
        "as_of": str(as_of),
        "window_trading_days": window,
        "universe_note": "Sample US + JSE names from cached OHLCV; educational only, not advice.",
        "stocks": stocks_payload,
        "classifier_note": classifier_note,
    }

# Building the narrative blurb.
def build_narrative_blurb(payload: dict[str, Any]) -> str:
    """
    One paragraph template the LLM can tighten — still grounded in payload numbers.
    """
    # Initialising an empty list to store the lines.
    lines = []
    
    # Looping through the stocks in the payload.
    for name, info in payload["stocks"].items():
        # Adding the stock to the lines.
        lines.append(
            f"{name}: last {payload['window_trading_days']}d avg daily return "
            f"{info['mean_daily_return_pct_window']}%, "
            f"volatility (std of daily returns) {info['volatility_daily_pct_window']}%, "
            f"cluster {info['kmeans_cluster_id']} ({info['cluster_label']})."
        )
    # Joining the lines into a single string.
    body = " ".join(lines)      
    # Returning the narrative blurb with the payload information    .
    return (
        f"As of {payload['as_of']}, {payload['universe_note']} "
        f"{body} Classifier note: {payload['classifier_note']}"
    )

