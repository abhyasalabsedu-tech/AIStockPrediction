"""
This is the ONLY place Gemini is called. Everything else (indicators, ML, RAG retrieval)
is deterministic/local. Gemini receives all upstream outputs as grounded context and is
explicitly instructed not to invent facts beyond what's provided — this is what makes the
reasoning explainable rather than a free-floating hallucination.
"""
import time
import google.generativeai as genai
from app.core.config import get_settings

settings = get_settings()
genai.configure(api_key=settings.gemini_api_key)

SYSTEM_INSTRUCTION = """You are the reasoning layer of an explainable stock intelligence platform.
You are given: (1) deterministic technical indicators computed in Python, (2) an XGBoost model's
probability output and top features, (3) retrieved excerpts from real filings/news via RAG.
Your job is ONLY to synthesize and explain — never invent a fact, number, or document that
was not provided to you. If evidence conflicts, say so explicitly. Keep responses grounded,
concise, and structured. Never give unqualified financial advice — frame as analysis, not instruction."""


def _build_prompt(ticker, indicators, ml_output, rag_output, question=None):
    docs_block = "\n".join(
        f"- [{d['type']}] {d['title']} (similarity {d['similarity']}): {d['chunk']}"
        for d in rag_output["documents"]
    )
    base = f"""
Ticker: {ticker}

PYTHON ENGINE (deterministic):
RSI: {indicators['rsi_14']} ({indicators['rsi_signal']})
EMA 20/50: {indicators['ema_20']}/{indicators['ema_50']} ({indicators['ema_signal']})
ADX: {indicators['adx_14']} ({indicators['adx_signal']})
VWAP: {indicators['vwap']} ({indicators['vwap_signal']})
Support/Resistance: {indicators['support']} / {indicators['resistance']}

MACHINE LEARNING (XGBoost {ml_output['model_version']}):
Signal: {ml_output['signal']} | Confidence: {ml_output['confidence']}%
Probabilities: {ml_output['probabilities']}
Top features: {ml_output['feature_importance'][:3]}

RETRIEVED DOCUMENTS (RAG, {rag_output['chunks_retrieved']} chunks, mean similarity {rag_output['mean_similarity']}):
{docs_block or '(no documents retrieved — reason from technical/ML signal only, note the gap)'}
"""
    if question:
        return base + f"\nUSER QUESTION: {question}\nAnswer the question directly using only the evidence above."
    return base + "\nWrite a 3-4 sentence natural-language summary explaining the overall picture, in the voice of a market analyst."


def generate_reasoning(ticker, indicators, ml_output, rag_output, question=None) -> dict:
    model = genai.GenerativeModel(settings.gemini_model, system_instruction=SYSTEM_INSTRUCTION)
    prompt = _build_prompt(ticker, indicators, ml_output, rag_output, question)

    t0 = time.perf_counter()
    response = model.generate_content(prompt)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    usage = getattr(response, "usage_metadata", None)
    tokens = (usage.total_token_count if usage else None)

    return {
        "text": response.text,
        "latency_ms": latency_ms,
        "tokens_used": tokens,
        "model": settings.gemini_model,
    }
