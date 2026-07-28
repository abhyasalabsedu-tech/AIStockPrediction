import json
import time
from pathlib import Path
import numpy as np
import xgboost as xgb

from app.services.indicators import build_feature_row
from app.ml.train import FEATURE_COLS, LABEL_MAP, MODEL_PATH, IMPORTANCE_PATH

_model = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "No trained model found. Run `python -m app.ml.train` first to train on real historical data."
            )
        _model = xgb.XGBClassifier()
        _model.load_model(str(MODEL_PATH))
    return _model


def predict(df) -> dict:
    """df: OHLCV dataframe, most recent row is 'now'. Returns full explainable ML output."""
    model = _load_model()
    features = build_feature_row(df)
    x = np.array([[features[c] for c in FEATURE_COLS]])

    t0 = time.perf_counter()
    proba = model.predict_proba(x)[0]
    pred_class = int(np.argmax(proba))
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    importance = json.loads(IMPORTANCE_PATH.read_text())["importance"] if IMPORTANCE_PATH.exists() else []

    return {
        "signal": LABEL_MAP[pred_class],
        "probabilities": {LABEL_MAP[i]: round(float(p) * 100, 1) for i, p in enumerate(proba)},
        "confidence": round(float(proba[pred_class]) * 100, 1),
        "features_used": features,
        "feature_importance": importance,
        "prediction_time_ms": latency_ms,
        "model_version": "xgb-v1.0",
    }
