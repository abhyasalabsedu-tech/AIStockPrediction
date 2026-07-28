"""
Embeddings run locally via sentence-transformers (all-MiniLM-L6-v2, 384-dim) —
no API key or per-call cost. Only the final AI *reasoning* step (app/services/ai.py)
calls Gemini. This keeps RAG cheap to run at ingestion scale.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache
def get_embedder():
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    return get_embedder().encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Simple sliding-window chunker. Swap for a semantic/recursive splitter later if needed."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
