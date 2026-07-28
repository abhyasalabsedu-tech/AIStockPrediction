from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session

from app.models.schema import get_db, Prediction, EvaluationRun
from app.services.market_data import fetch_latest_quote, fetch_ohlcv
from app.services.indicators import compute_all_indicators
from app.agents.graph import run_pipeline
from app.services.ai import generate_reasoning
from app.rag.retrieve import retrieve, build_query_for_ticker
from app.rag.ingest import ingest_document
from app.ml.predict import predict as ml_predict
from app.ml.train import train as train_ml_model
from app.core.security import require_admin

router = APIRouter()


@router.get("/quote")
def get_quote(ticker: str = Query(default="TCS.NS")):
    try:
        return fetch_latest_quote(ticker)
    except Exception as e:
        raise HTTPException(502, f"Market data fetch failed: {e}")


@router.get("/series")
def get_series(ticker: str = Query(default="TCS.NS"), period: str = Query(default="5d"), interval: str = Query(default="15m")):
    """Real OHLCV history for charting. Intraday intervals (e.g. 15m) only work for short periods on yfinance's free data."""
    try:
        df = fetch_ohlcv(ticker, period=period, interval=interval)
    except Exception as e:
        raise HTTPException(502, f"Market data fetch failed: {e}")

    df = df.tail(120)
    return [
        {
            "t": ts.strftime("%H:%M") if period in ("1d", "5d") else ts.strftime("%d %b"),
            "price": round(float(row["close"]), 2),
            "vwap": round(float((row["high"] + row["low"] + row["close"]) / 3), 2),
            "volume": int(row["volume"]),
        }
        for ts, row in df.iterrows()
    ]


@router.get("/indicators")
def get_indicators(ticker: str = Query(default="TCS.NS")):
    df = fetch_ohlcv(ticker, period="6mo", interval="1d")
    return compute_all_indicators(df)


@router.get("/ml/predict")
def get_ml_prediction(ticker: str = Query(default="TCS.NS")):
    df = fetch_ohlcv(ticker, period="6mo", interval="1d")
    return ml_predict(df)


@router.get("/rag/retrieve")
def get_rag(ticker: str = Query(default="TCS"), query: str = Query(default=None)):
    q = query or f"{ticker} outlook risk factors deal wins"
    return retrieve(q, ticker=ticker)


@router.post("/predict")
def run_full_prediction(ticker: str = Query(default="TCS.NS"), db: Session = Depends(get_db)):
    """Runs the entire LangGraph pipeline end to end and persists the auditable result."""
    t0 = datetime.utcnow()
    try:
        state = run_pipeline(ticker)
    except Exception as e:
        raise HTTPException(502, f"Pipeline execution failed: {e}")

    ind, ml, rag, ai = state["indicators"], state["ml_output"], state["rag_output"], state["ai_output"]
    row = Prediction(
        ticker=ticker, ts=t0,
        indicators=ind, support=ind["support"], resistance=ind["resistance"],
        ml_signal=ml["signal"], ml_probability=ml["confidence"],
        feature_importance=ml["feature_importance"], model_version=ml["model_version"],
        retrieved_doc_ids=[d["id"] for d in rag["documents"]], rag_context_used=str(rag["documents"]),
        ai_summary=ai["text"], ai_tokens_used=ai.get("tokens_used"), ai_latency_ms=ai["latency_ms"],
        agent_votes=state["agent_votes"],
        final_decision=state["final_decision"], final_confidence=state["final_confidence"],
        expected_price=None, stop_loss=ind["support"], target=ind["resistance"],
        execution_time_ms=int(sum(n["ms"] for n in state["node_log"])),
        was_correct=-1,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "prediction_id": row.id,
        "ticker": ticker,
        "indicators": ind,
        "ml": ml,
        "rag": rag,
        "ai": ai,
        "agent_votes": state["agent_votes"],
        "critic_flag": state.get("critic_flag"),
        "final_decision": state["final_decision"],
        "final_confidence": state["final_confidence"],
        "node_log": state["node_log"],
    }


@router.post("/ai/ask")
def ask_ai(ticker: str = Query(default="TCS.NS"), question: str = Query(...)):
    df = fetch_ohlcv(ticker, period="6mo", interval="1d")
    ind = compute_all_indicators(df)
    ml = ml_predict(df)
    query = build_query_for_ticker(ticker, ind, ml["signal"])
    rag = retrieve(query, ticker=ticker.replace(".NS", ""))
    return generate_reasoning(ticker, ind, ml, rag, question=question)


@router.get("/history")
def get_history(ticker: str = Query(default="TCS.NS"), limit: int = 30, db: Session = Depends(get_db)):
    rows = (
        db.query(Prediction)
        .filter(Prediction.ticker == ticker)
        .order_by(Prediction.ts.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id, "ts": r.ts.isoformat(), "final_decision": r.final_decision,
            "final_confidence": r.final_confidence, "ml_signal": r.ml_signal,
            "actual_outcome": r.actual_outcome, "was_correct": r.was_correct,
        }
        for r in rows
    ]


@router.get("/history/{prediction_id}")
def get_prediction_detail(prediction_id: int, db: Session = Depends(get_db)):
    row = db.query(Prediction).get(prediction_id)
    if not row:
        raise HTTPException(404, "Prediction not found")
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


@router.get("/evaluation/latest")
def get_latest_evaluation(ticker: str = Query(default="TCS.NS"), db: Session = Depends(get_db)):
    row = (
        db.query(EvaluationRun)
        .filter(EvaluationRun.ticker == ticker)
        .order_by(EvaluationRun.run_date.desc())
        .first()
    )
    if not row:
        raise HTTPException(404, "No evaluation runs yet. Run app/core/evaluation.py after a day of predictions.")
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


# ============================== ADMIN ENDPOINTS ==============================
# These let you trigger training/ingestion by clicking buttons in the dashboard's
# Admin tab, instead of needing a terminal or Render's (paid-only) Shell feature.

@router.post("/admin/train")
def admin_train_model(ticker: str = Query(default="TCS.NS"), _auth: bool = Depends(require_admin)):
    """Trains XGBoost on 5 years of real historical data. Takes ~10-30 seconds."""
    try:
        metrics = train_ml_model(ticker)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")


@router.post("/admin/ingest")
def admin_ingest_document(
    payload: dict = Body(...),
    _auth: bool = Depends(require_admin),
):
    """
    Body: { "title": "...", "doc_type": "Annual Report", "ticker": "TCS", "text": "...paste raw text..." }
    Embeds via Gemini and stores in pgvector. Costs one embedding API call per ~800-word chunk.
    """
    title = payload.get("title")
    doc_type = payload.get("doc_type")
    ticker = payload.get("ticker", "TCS")
    text = payload.get("text", "")

    if not title or not doc_type or not text.strip():
        raise HTTPException(400, "title, doc_type, and text are all required.")

    try:
        n_chunks = ingest_document(title, doc_type, text, ticker)
        return {"status": "success", "chunks_ingested": n_chunks}
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")
