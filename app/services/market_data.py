"""
Real market data via yfinance (free, no key). TCS on NSE = ticker 'TCS.NS'.
Swap `fetch_ohlcv` internals for Kite Connect / Upstox later without touching callers.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
from app.core.config import get_settings

settings = get_settings()


def fetch_ohlcv(ticker: str = None, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    period: '1d','5d','1mo','6mo','1y','5y','max'
    interval: '1m','5m','15m','1h','1d' (intraday intervals only available for short periods on yfinance)
    Returns DataFrame indexed by datetime with columns: Open, High, Low, Close, Volume
    """
    ticker = ticker or settings.default_ticker
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol or network access.")
    df = df.rename(columns=str.lower)
    df.index.name = "ts"
    return df[["open", "high", "low", "close", "volume"]]


def fetch_latest_quote(ticker: str = None) -> dict:
    ticker = ticker or settings.default_ticker
    df = fetch_ohlcv(ticker, period="5d", interval="1d")
    last, prev = df.iloc[-1], df.iloc[-2]
    change = last["close"] - prev["close"]
    return {
        "ticker": ticker,
        "price": round(float(last["close"]), 2),
        "change": round(float(change), 2),
        "change_pct": round(float(change / prev["close"] * 100), 2),
        "volume": int(last["volume"]),
        "as_of": datetime.utcnow().isoformat(),
    }


def fetch_training_history(ticker: str = None, years: str = "5y") -> pd.DataFrame:
    """Daily bars for XGBoost training — 5 years is yfinance's practical free ceiling for daily data."""
    return fetch_ohlcv(ticker, period=years, interval="1d")
