import time
from app.models.schema import Document, SessionLocal
from app.rag.embed import embed_query


def retrieve(query: str, ticker: str = "TCS", k: int = 6) -> dict:
    """Cosine-similarity search over pgvector. Returns top-k chunks + latency + mean similarity."""
    t0 = time.perf_counter()
    query_vec = embed_query(query)

    db = SessionLocal()
    try:
        # pgvector cosine distance operator `<=>` ; similarity = 1 - distance
        results = (
            db.query(Document, Document.embedding.cosine_distance(query_vec).label("distance"))
            .filter(Document.ticker == ticker)
            .order_by("distance")
            .limit(k)
            .all()
        )
    finally:
        db.close()

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    docs = []
    for doc, distance in results:
        docs.append({
            "id": doc.id,
            "title": doc.title,
            "type": doc.doc_type,
            "similarity": round(1 - float(distance), 3),
            "chunk": doc.chunk_text[:400],
            "source_url": doc.source_url,
        })

    mean_sim = round(sum(d["similarity"] for d in docs) / len(docs), 3) if docs else 0.0
    return {
        "documents": docs,
        "chunks_retrieved": len(docs),
        "mean_similarity": mean_sim,
        "latency_ms": latency_ms,
    }


def build_query_for_ticker(ticker: str, indicators: dict, ml_signal: str) -> str:
    """Turns the current technical/ML state into a retrieval query — grounds RAG in *this* prediction cycle."""
    return (
        f"{ticker} stock outlook, deal wins, earnings guidance, risk factors. "
        f"Current technical bias: {ml_signal}. RSI {indicators.get('rsi_14')}, "
        f"EMA signal {indicators.get('ema_signal')}, ADX {indicators.get('adx_14')}."
    )
