"""
Pure deterministic computation. No ML, no AI. Every value here is reproducible
from the same OHLCV input — this is what the 'Python Engine' page displays.
"""
import pandas as pd
import numpy as np


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series):
    ema12, ema26 = _ema(close, 12), _ema(close, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    return macd_line, signal


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()


def _bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma - std_mult * std, sma + std_mult * std


def _stochastic(df: pd.DataFrame, period: int = 14):
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    return 100 * (df["close"] - low_min) / (high_max - low_min)


def _vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return (typical * df["volume"]).cumsum() / df["volume"].cumsum()


def compute_all_indicators(df: pd.DataFrame) -> dict:
    """df must have columns: open, high, low, close, volume, sorted ascending by time."""
    close = df["close"]
    rsi = _rsi(close).iloc[-1]
    macd_line, macd_signal = _macd(close)
    ema20, ema50 = _ema(close, 20).iloc[-1], _ema(close, 50).iloc[-1]
    adx = _adx(df).iloc[-1]
    bb_low, bb_high = _bollinger(close)
    stoch = _stochastic(df).iloc[-1]
    atr = _atr(df).iloc[-1]
    vwap = _vwap(df).iloc[-1]

    support = float(df["low"].tail(30).min())
    resistance = float(df["high"].tail(30).max())
    volatility_10d = float(close.pct_change(fill_method=None).tail(10).std() * 100)

    def sig(cond_bull, cond_bear):
        return "Bullish" if cond_bull else ("Bearish" if cond_bear else "Neutral")

    return {
        "rsi_14": round(float(rsi), 2),
        "rsi_signal": sig(rsi > 55, rsi < 45),
        "macd": round(float(macd_line.iloc[-1]), 2),
        "macd_signal_line": round(float(macd_signal.iloc[-1]), 2),
        "macd_signal": sig(macd_line.iloc[-1] > macd_signal.iloc[-1], macd_line.iloc[-1] < macd_signal.iloc[-1]),
        "ema_20": round(float(ema20), 2),
        "ema_50": round(float(ema50), 2),
        "ema_signal": sig(ema20 > ema50, ema20 < ema50),
        "adx_14": round(float(adx), 2),
        "adx_signal": "Trending" if adx > 25 else "Range-bound",
        "bollinger_low": round(float(bb_low.iloc[-1]), 2),
        "bollinger_high": round(float(bb_high.iloc[-1]), 2),
        "stochastic": round(float(stoch), 2),
        "stochastic_signal": sig(stoch > 60, stoch < 40),
        "atr_14": round(float(atr), 2),
        "vwap": round(float(vwap), 2),
        "vwap_signal": sig(close.iloc[-1] > vwap, close.iloc[-1] < vwap),
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "volatility_10d_pct": round(volatility_10d, 2),
    }


def build_feature_row(df: pd.DataFrame) -> dict:
    """Feature engineering for the ML model — derived purely from Python-computed indicators."""
    close = df["close"]
    ema20, ema50 = _ema(close, 20), _ema(close, 50)
    features = {
        "ema_cross_strength": float((ema20.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1] * 100),
        "momentum_5d": float(close.pct_change(5).iloc[-1] * 100),
        "rsi_slope": float(_rsi(close).diff().tail(3).mean()),
        "volume_zscore": float((df["volume"].iloc[-1] - df["volume"].tail(20).mean()) / df["volume"].tail(20).std()),
        "atr_regime": float(_atr(df).tail(3).mean() / _atr(df).tail(10).mean()),
        "adx": float(_adx(df).iloc[-1]),
        "stochastic": float(_stochastic(df).iloc[-1]),
    }
    return features
