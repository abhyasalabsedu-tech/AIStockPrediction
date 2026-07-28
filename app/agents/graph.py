"""
LangGraph state machine that IS the pipeline: Market Data -> Python -> ML -> RAG -> AI ->
Agent Consensus -> Critic -> Final Decision. Each node writes into a shared state dict so
the full execution trace can be persisted to `Prediction` for audit/replay.
"""
import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.services.market_data import fetch_ohlcv
from app.services.indicators import compute_all_indicators
from app.ml.predict import predict as ml_predict
from app.rag.retrieve import retrieve, build_query_for_ticker
from app.services.ai import generate_reasoning


class PipelineState(TypedDict, total=False):
    ticker: str
    df: object
    indicators: dict
    ml_output: dict
    rag_output: dict
    ai_output: dict
    agent_votes: list
    final_decision: str
    final_confidence: float
    node_log: list
    errors: list


def _log(state: PipelineState, node: str, status: str, ms: float):
    state.setdefault("node_log", []).append({"node": node, "status": status, "ms": round(ms, 1)})


def node_ingest(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()
    df = fetch_ohlcv(state["ticker"], period="6mo", interval="1d")
    state["df"] = df
    _log(state, "ingest", "complete", (time.perf_counter() - t0) * 1000)
    return state


def node_python_engine(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()
    state["indicators"] = compute_all_indicators(state["df"])
    _log(state, "python_engine", "complete", (time.perf_counter() - t0) * 1000)
    return state


def node_ml(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()
    state["ml_output"] = ml_predict(state["df"])
    _log(state, "ml_inference", "complete", (time.perf_counter() - t0) * 1000)
    return state


def node_rag(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()
    query = build_query_for_ticker(state["ticker"], state["indicators"], state["ml_output"]["signal"])
    state["rag_output"] = retrieve(query, ticker=state["ticker"].replace(".NS", ""))
    _log(state, "rag_retrieval", "complete", (time.perf_counter() - t0) * 1000)
    return state


def node_ai_reasoning(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()
    state["ai_output"] = generate_reasoning(
        state["ticker"], state["indicators"], state["ml_output"], state["rag_output"]
    )
    _log(state, "ai_reasoning", "complete", (time.perf_counter() - t0) * 1000)
    return state


def node_agent_consensus(state: PipelineState) -> PipelineState:
    """Rule-based specialist agents voting off the SAME shared state — deterministic and auditable,
    each agent's 'reasoning' is templated from real upstream values, not separately hallucinated."""
    t0 = time.perf_counter()
    ind, ml = state["indicators"], state["ml_output"]

    def vote(cond_buy, cond_sell):
        return "BUY" if cond_buy else ("SELL" if cond_sell else "HOLD")

    technical_vote = vote(ind["ema_signal"] == "Bullish" and ind["adx_signal"] == "Trending",
                           ind["ema_signal"] == "Bearish")
    ml_vote = ml["signal"]
    rag_docs = state["rag_output"]["documents"]
    news_positive = any("deal" in d["chunk"].lower() or "win" in d["chunk"].lower() for d in rag_docs)
    news_vote = vote(news_positive, False)
    risk_vote = "BUY" if ind["volatility_10d_pct"] < 3.5 else "HOLD"

    votes = [
        {"name": "Technical Agent", "vote": technical_vote, "confidence": ind["adx_14"],
         "reasoning": f"EMA signal {ind['ema_signal']}, ADX {ind['adx_14']} ({ind['adx_signal']})."},
        {"name": "ML Agent", "vote": ml_vote, "confidence": ml["confidence"],
         "reasoning": f"XGBoost probability {ml['confidence']}% for {ml_vote}."},
        {"name": "News/RAG Agent", "vote": news_vote, "confidence": state["rag_output"]["mean_similarity"] * 100,
         "reasoning": f"{state['rag_output']['chunks_retrieved']} documents retrieved, positive deal-flow mentions: {news_positive}."},
        {"name": "Risk Agent", "vote": risk_vote, "confidence": 100 - ind["volatility_10d_pct"] * 10,
         "reasoning": f"10-day volatility {ind['volatility_10d_pct']}%."},
    ]

    buy_weight = sum(v["confidence"] for v in votes if v["vote"] == "BUY")
    sell_weight = sum(v["confidence"] for v in votes if v["vote"] == "SELL")
    hold_weight = sum(v["confidence"] for v in votes if v["vote"] == "HOLD")
    total = buy_weight + sell_weight + hold_weight or 1

    final = max([("BUY", buy_weight), ("SELL", sell_weight), ("HOLD", hold_weight)], key=lambda x: x[1])
    state["agent_votes"] = votes
    state["final_decision"] = final[0]
    state["final_confidence"] = round(final[1] / total * 100, 1)
    _log(state, "agent_consensus", "complete", (time.perf_counter() - t0) * 1000)
    return state


def node_critic(state: PipelineState) -> PipelineState:
    """Checks for contradictions between ML output and agent consensus; flags, doesn't block."""
    t0 = time.perf_counter()
    contradiction = state["ml_output"]["signal"] != state["final_decision"] and state["final_confidence"] < 60
    state["critic_flag"] = "Low-confidence contradiction between ML and consensus" if contradiction else "No contradiction found"
    _log(state, "critic", "complete", (time.perf_counter() - t0) * 1000)
    return state


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("ingest", node_ingest)
    g.add_node("python_engine", node_python_engine)
    g.add_node("ml", node_ml)
    g.add_node("rag", node_rag)
    g.add_node("ai_reasoning", node_ai_reasoning)
    g.add_node("agent_consensus", node_agent_consensus)
    g.add_node("critic", node_critic)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "python_engine")
    g.add_edge("python_engine", "ml")
    g.add_edge("ml", "rag")
    g.add_edge("rag", "ai_reasoning")
    g.add_edge("ai_reasoning", "agent_consensus")
    g.add_edge("agent_consensus", "critic")
    g.add_edge("critic", END)
    return g.compile()


_compiled_graph = None


def run_pipeline(ticker: str) -> PipelineState:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph.invoke({"ticker": ticker})
