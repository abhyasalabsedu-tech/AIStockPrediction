"""
Trains an XGBoost classifier on real historical TCS data pulled via yfinance.
Label: did the stock go UP (>+0.5%), DOWN (<-0.5%), or FLAT over the next 5 sessions?
Run: python -m app.ml.train
Produces: app/ml/model.json + app/ml/feature_importance.json
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.services.market_data import fetch_training_history
from app.services.indicators import _ema, _rsi, _adx, _atr, _stochastic

MODEL_PATH = Path(__file__).parent / "model.json"
IMPORTANCE_PATH = Path(__file__).parent / "feature_importance.json"
LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

FEATURE_COLS = [
    "ema_cross_strength", "momentum_5d", "rsi_slope",
    "volume_zscore", "atr_regime", "adx", "stochastic",
]


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    ema20, ema50 = _ema(close, 20), _ema(close, 50)

    feat = pd.DataFrame(index=df.index)
    feat["ema_cross_strength"] = (ema20 - ema50) / ema50 * 100
    feat["momentum_5d"] = close.pct_change(5) * 100
    feat["rsi_slope"] = _rsi(close).diff().rolling(3).mean()
    feat["volume_zscore"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    feat["atr_regime"] = _atr(df).rolling(3).mean() / _atr(df).rolling(10).mean()
    feat["adx"] = _adx(df)
    feat["stochastic"] = _stochastic(df)

    # Forward-looking label: 5-session-ahead return, bucketed
    fwd_return = close.shift(-5) / close - 1
    feat["label"] = pd.cut(fwd_return, bins=[-np.inf, -0.005, 0.005, np.inf], labels=[0, 1, 2]).astype("float")

    return feat.dropna()


def train():
    df = fetch_training_history(years="5y")
    dataset = build_dataset(df)

    X, y = dataset[FEATURE_COLS], dataset["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, objective="multi:softprob",
        num_class=3, eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds) * 100, 2),
        "precision": round(precision_score(y_test, preds, average="macro", zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, preds, average="macro", zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, preds, average="macro", zero_division=0) * 100, 2),
        "sample_size": int(len(X_test)),
        "trained_rows": int(len(X_train)),
    }

    importance = model.feature_importances_
    ranked = sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])

    model.save_model(str(MODEL_PATH))
    IMPORTANCE_PATH.write_text(json.dumps({
        "importance": [{"feature": f, "value": round(float(v), 4)} for f, v in ranked],
        "metrics": metrics,
    }, indent=2))

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
