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



