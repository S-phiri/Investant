"""
Financial OHLCV data client for Investant.

**Why two sources?**

- **Alpha Vantage** is the primary source for **US-listed equities** (e.g. IBM, Apple,
  Microsoft). It offers a stable, key-authenticated API suited to major US symbols.

- **yfinance** backs **JSE (Johannesburg) African listings** (e.g. ``MTN.JO``, ``NPN.JO``).
  JSE tickers are often thin or missing on free-tier US APIs; yfinance typically has
  better JSE symbol coverage and does not impose the same per-minute rate limits,
  so it is used here without an API key while Alpha Vantage handles US names under
  the free-tier 5-calls-per-minute cap.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# Project root (parent of src/) for .env and data/raw regardless of CWD
_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _ROOT / ".env"
_RAW_DIR = _ROOT / "data" / "raw"

load_dotenv(_ENV_PATH)
API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
BASE_URL = "https://www.alphavantage.co/query"

US_STOCKS: dict[str, str] = {
    "IBM": "IBM",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
}

AFRICAN_STOCKS: dict[str, str] = {
    "MTN Group": "MTN.JO",
    "Naspers": "NPN.JO",
    "Shoprite": "SHP.JO",
    "FirstRand": "FSR.JO",
    "Anglo American": "AGL.JO",
}


def _cache_path(company_name: str) -> Path:
    """Return the CSV cache path for a company (safe filename)."""
    safe = company_name.replace("/", "-").replace("\\", "-")
    return _RAW_DIR / f"{safe}.csv"


def search_symbol(keywords: str) -> None:
    """
    Search Alpha Vantage for symbols matching the given keywords (SYMBOL_SEARCH).

    Args:
        keywords: Search string (e.g. company or ticker fragment).
    """
    if not API_KEY:
        print("ALPHA_VANTAGE_KEY is not set in .env")
        return
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": keywords,
        "apikey": API_KEY,
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        matches = data.get("bestMatches", [])
        for match in matches:
            print(f"{match['2. name']} — {match['1. symbol']}")
    except requests.exceptions.RequestException as e:
        print(f"Symbol search request failed: {e}")


def fetch_alpha_vantage_daily(symbol: str, company_name: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Alpha Vantage for a US symbol; cache under data/raw/.

    Args:
        symbol: Alpha Vantage ticker (e.g. ``IBM``, ``AAPL``).
        company_name: Human-readable name used for the cache filename.

    Returns:
        DataFrame indexed by date with columns open, high, low, close, volume.
    """
    cache_path = _cache_path(company_name)
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    if not API_KEY:
        print(f"No ALPHA_VANTAGE_KEY; cannot fetch {company_name}.")
        return pd.DataFrame()

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": API_KEY,
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        time_series = data.get("Time Series (Daily)", {})

        if not time_series:
            print(
                f"No data returned for {company_name} ({symbol}). "
                "Check symbol, API limit, or note field in response."
            )
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(time_series, orient="index")
        rename_map = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
        df = df.rename(columns=rename_map)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                print(f"Unexpected Alpha Vantage columns for {company_name}: {list(df.columns)}")
                return pd.DataFrame()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        os.makedirs(str(_RAW_DIR), exist_ok=True)
        df.to_csv(cache_path)
        print(f"Saved {company_name} to cache (Alpha Vantage).")

        return df
    except requests.exceptions.RequestException as e:
        print(f"Alpha Vantage request failed for {company_name}: {e}")
        return pd.DataFrame()
    except (KeyError, ValueError, TypeError) as e:
        print(f"Failed to parse Alpha Vantage data for {company_name}: {e}")
        return pd.DataFrame()


def fetch_yfinance_daily(symbol: str, company_name: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV from yfinance for a JSE (or other) symbol; cache to data/raw/.

    Args:
        symbol: yfinance ticker (e.g. ``MTN.JO``).
        company_name: Human-readable name used for the cache filename.

    Returns:
        DataFrame indexed by date with columns open, high, low, close, volume.
    """
    cache_path = _cache_path(company_name)
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y", interval="1d", auto_adjust=False)

        if hist is None or hist.empty:
            print(f"No yfinance data for {company_name} ({symbol}).")
            return pd.DataFrame()

        cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in cols if c not in hist.columns]
        if missing:
            print(f"yfinance missing columns for {company_name}: {missing}")
            return pd.DataFrame()

        df = hist[cols].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        # yfinance often returns tz-aware index; normalize to naive dates for CSV/cache parity
        df.index = pd.to_datetime(df.index.strftime("%Y-%m-%d"))
        df = df.astype({"open": float, "high": float, "low": float, "close": float})
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(float)
        df.sort_index(inplace=True)

        os.makedirs(str(_RAW_DIR), exist_ok=True)
        df.to_csv(cache_path)
        print(f"Saved {company_name} to cache (yfinance).")

        return df
    except Exception as e:
        print(f"yfinance failed for {company_name} ({symbol}): {e}")
        return pd.DataFrame()


def fetch_all_stocks() -> dict[str, pd.DataFrame]:
    """
    Fetch US stocks via Alpha Vantage and African JSE stocks via yfinance.

    Alpha Vantage calls are spaced by 13 seconds to stay within 5 calls/minute.
    yfinance calls are not delayed.

    Returns:
        Combined mapping ``{company_name: DataFrame}`` with columns
        open, high, low, close, volume.
    """
    all_data: dict[str, pd.DataFrame] = {}

    us_list = list(US_STOCKS.items())
    for i, (company_name, symbol) in enumerate(us_list):
        if i > 0:
            time.sleep(13)
        df = fetch_alpha_vantage_daily(symbol, company_name)
        if not df.empty:
            all_data[company_name] = df

    for company_name, symbol in AFRICAN_STOCKS.items():
        df = fetch_yfinance_daily(symbol, company_name)
        if not df.empty:
            all_data[company_name] = df

    return all_data


if __name__ == "__main__":
    stocks = fetch_all_stocks()
    print("\n" + "=" * 60)
    print("Investant — combined fetch summary")
    print("=" * 60)

    for name, df in stocks.items():
        if df.empty:
            print(f"\n{name}: (no data)")
            continue
        start = df.index.min()
        end = df.index.max()
        n_days = len(df)
        print(f"\n{name}")
        print(f"  Days: {n_days}")
        print(f"  Date range: {start.date()} → {end.date()}")
        print("  Last 3 rows:")
        print(df.tail(3).to_string())

    if not stocks:
        print("\nNo stock data retrieved.")
