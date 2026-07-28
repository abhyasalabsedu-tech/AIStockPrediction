"""
Real market data via yfinance (free, no key). TCS on NSE = ticker 'TCS.NS'.

IMPORTANT: Yahoo Finance actively blocks plain requests from cloud/datacenter IPs
(the kind Render, Railway, etc. use) -- this is why a raw yfinance call that works
fine on a home/office connection can fail with "possibly delisted; no price data
found" once deployed. curl_cffi impersonates a real Chrome browser's TLS/HTTP
fingerprint, which is the standard fix as of 2024+ Yahoo anti-bot changes.

Swap `fetch_ohlcv` internals for Kite Connect / Upstox later without touching callers.
"""
import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests
from datetime import datetime
from app.core.config import get_settings

settings = get_settings()

_session = curl_requests.Session(impersonate="chrome")


def fetch_ohlcv(ticker: str = None, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    period: '1d','5d','1mo','6mo','1y','5y','max'
    interval: '1m','5m','15m','1h','1d' (intraday intervals only available for short periods on yfinance)
    Returns DataFrame indexed by datetime with columns: Open, High, Low, Close, Volume
    """
    ticker = ticker or settings.default_ticker
    df = yf.Ticker(ticker, session=_session).history(period=period, interval=interval)
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
