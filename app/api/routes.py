from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.models.schema import get_db, Prediction, EvaluationRun
from app.services.market_data import fetch_latest_quote, fetch_ohlcv
from app.services.indicators import compute_all_indicators
from app.agents.graph import run_pipeline
from app.services.ai import generate_reasoning
from app.rag.retrieve import retrieve, build_query_for_ticker
from app.ml.predict import predict as ml_predict

router = APIRouter()


@router.get("/quote")
def get_quote(ticker: str = Query(default="TCS.NS")):
    try:
        return fetch_latest_quote(ticker)
    except Exception as e:
        raise HTTPException(502, f"Market data fetch failed: {e}")


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
