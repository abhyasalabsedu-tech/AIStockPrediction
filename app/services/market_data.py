"""
Real market data via yfinance (free, no key). TCS on NSE = ticker 'TCS.NS'.

RESILIENCE STRATEGY (self-healing, no manual intervention needed):
Yahoo Finance actively blocks plain requests from cloud/datacenter IPs (the kind
Render, Railway, etc. use) -- this is why a raw yfinance call that works fine on a
home/office connection can fail once deployed. Rather than requiring a redeploy
every time Yahoo's blocking behavior shifts, this module:
  1. Tries multiple browser-impersonation fingerprints in sequence (chrome, then
     safari, then edge) -- if one gets blocked, the next often still works.
  2. Falls back to a plain, unimpersonated request as a last live attempt.
  3. If ALL live attempts fail, falls back to the most recent data this backend
     has successfully fetched before (cached in Postgres) -- so the app keeps
     working with slightly stale data instead of hard-erroring on every request.
  4. Every successful live fetch updates that cache, so the fallback data is
     always as fresh as the last time Yahoo actually responded.

Swap `fetch_ohlcv` internals for Kite Connect / Upstox later without touching callers.
"""
import pandas as pd
import yfinance as yf
import httpx
from curl_cffi import requests as curl_requests
from datetime import datetime
from app.core.config import get_settings

settings = get_settings()

_IMPERSONATIONS = ["chrome", "safari", "edge"]


def _try_fetch_yahoo(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Attempts the live Yahoo fetch through several browser fingerprints, returns the first success."""
    last_error = None

    for impersonate in _IMPERSONATIONS:
        try:
            session = curl_requests.Session(impersonate=impersonate)
            df = yf.Ticker(ticker, session=session).history(period=period, interval=interval)
            if not df.empty:
                return df
        except Exception as e:
            last_error = e

    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if not df.empty:
            return df
    except Exception as e:
        last_error = e

    raise ValueError(f"Yahoo Finance: all fetch attempts failed: {last_error}")


def _try_fetch_alpha_vantage(ticker: str) -> pd.DataFrame:
    """
    PLAN B data source -- only runs if ALPHA_VANTAGE_API_KEY is set in Render's environment.
    Get a free key (no credit card) at: https://www.alphavantage.co/support/#api-key
    Then add it as an env var in Render named ALPHA_VANTAGE_API_KEY and redeploy -- no code
    changes needed, this tier activates automatically.

    Note: Alpha Vantage lists Indian equities under the .BSE suffix rather than .NS, and its
    free tier is DAILY granularity only (no intraday) -- this tier is skipped automatically
    for intraday requests and only engages for daily/multi-day series.
    """
    if not settings.alpha_vantage_api_key:
        raise ValueError("Alpha Vantage not configured (no ALPHA_VANTAGE_API_KEY set) -- skipping this tier.")

    av_symbol = ticker.replace(".NS", ".BSE").replace(".BO", ".BSE")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={av_symbol}&outputsize=full&apikey={settings.alpha_vantage_api_key}"
    )
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    series = data.get("Time Series (Daily)")
    if not series:
        raise ValueError(f"Alpha Vantage returned no data for {av_symbol}: {data.get('Note') or data.get('Error Message') or data}")

    rows = []
    for date_str, values in series.items():
        rows.append({
            "ts": pd.Timestamp(date_str),
            "open": float(values["1. open"]), "high": float(values["2. high"]),
            "low": float(values["3. low"]), "close": float(values["4. close"]),
            "volume": float(values["5. volume"]),
        })
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    return df


def _cache_bars(ticker: str, interval: str, df: pd.DataFrame):
    """Best-effort write to the OHLCVBar cache table. Never lets a caching failure break the live path."""
    try:
        from app.models.schema import SessionLocal, OHLCVBar
        db = SessionLocal()
        try:
            db.query(OHLCVBar).filter(OHLCVBar.ticker == ticker, OHLCVBar.interval == interval).delete()
            for ts, row in df.iterrows():
                db.add(OHLCVBar(
                    ticker=ticker, ts=ts.to_pydatetime().replace(tzinfo=None), interval=interval,
                    open=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]), volume=float(row["volume"]),
                ))
            db.commit()
        finally:
            db.close()
    except Exception:
        pass  # caching is best-effort; never block the live response on a cache write failure


def _read_cache(ticker: str, interval: str) -> pd.DataFrame:
    """Reads back whatever was last successfully cached, as a fallback when no live source works."""
    from app.models.schema import SessionLocal, OHLCVBar
    db = SessionLocal()
    try:
        rows = (
            db.query(OHLCVBar)
            .filter(OHLCVBar.ticker == ticker, OHLCVBar.interval == interval)
            .order_by(OHLCVBar.ts)
            .all()
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([{
            "ts": r.ts, "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume,
        } for r in rows]).set_index("ts")
        return df
    finally:
        db.close()


def fetch_ohlcv(ticker: str = None, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    period: '1d','5d','1mo','6mo','1y','5y','max'
    interval: '1m','5m','15m','1h','1d' (intraday intervals only available for short periods on yfinance)
    Returns DataFrame indexed by datetime with columns: open, high, low, close, volume.

    Tries, in order: (1) Yahoo Finance via yfinance with rotating browser fingerprints,
    (2) Alpha Vantage if configured, (3) last successfully cached data in Postgres.
    Only raises if all three produce nothing -- see module docstring for the full strategy.
    """
    ticker = ticker or settings.default_ticker
    errors = []

    try:
        df = _try_fetch_yahoo(ticker, period, interval)
        df = df.rename(columns=str.lower)
        df.index.name = "ts"
        df = df[["open", "high", "low", "close", "volume"]]
        _cache_bars(ticker, interval, df)
        return df
    except Exception as e:
        errors.append(f"Yahoo: {e}")

    if interval == "1d":
        try:
            df = _try_fetch_alpha_vantage(ticker)
            df.index.name = "ts"
            _cache_bars(ticker, interval, df)
            return df
        except Exception as e:
            errors.append(f"Alpha Vantage: {e}")

    cached = _read_cache(ticker, interval)
    if not cached.empty:
        return cached

    raise ValueError(
        f"No data available for {ticker}. All live sources failed: {' | '.join(errors)}. "
        f"No cached data exists yet either. This resolves automatically once any live source "
        f"succeeds once, or you can add a free ALPHA_VANTAGE_API_KEY (see market_data.py) for a second live source."
    )


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
