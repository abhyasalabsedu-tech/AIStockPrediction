"""
Embeddings via Gemini's embedding API (text-embedding-004, 768-dim) -- NOT a local
model. This avoids bundling torch/sentence-transformers, which alone exceeds a
512MB free-tier RAM budget before even serving a request. Costs are minimal
(embedding calls are cheap on Gemini's free tier), and reuses the same
GEMINI_API_KEY you already set for reasoning -- no new account needed.
"""
import google.generativeai as genai
from app.core.config import get_settings

settings = get_settings()
genai.configure(api_key=settings.gemini_api_key)

EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    result = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
    return result["embedding"]


def embed_query(text: str) -> list[float]:
    return embed_text(text, task_type="RETRIEVAL_QUERY")


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Gemini's SDK embeds one string per call; looping is fine at ingestion-time volumes."""
    return [embed_text(t) for t in texts]


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
