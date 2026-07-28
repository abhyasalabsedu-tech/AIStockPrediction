from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime

from app.core.config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class OHLCVBar(Base):
    """Raw market data, populated from yfinance. TimescaleDB hypertable candidate on `ts`."""
    __tablename__ = "ohlcv_bars"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), index=True, nullable=False)
    ts = Column(DateTime, index=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    interval = Column(String(10), default="1d")


class Prediction(Base):
    """One row per prediction cycle — the full auditable output of the pipeline."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), index=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)

    # Python engine
    indicators = Column(JSON)          # dict of computed indicator values
    support = Column(Float)
    resistance = Column(Float)

    # ML
    ml_signal = Column(String(10))     # BUY/HOLD/SELL
    ml_probability = Column(Float)
    feature_importance = Column(JSON)
    model_version = Column(String(30))

    # RAG
    retrieved_doc_ids = Column(JSON)   # list of Document.id + similarity scores
    rag_context_used = Column(Text)

    # AI
    ai_summary = Column(Text)
    ai_tokens_used = Column(Integer)
    ai_latency_ms = Column(Integer)

    # Agents
    agent_votes = Column(JSON)         # list of {name, vote, confidence, reasoning}

    # Final
    final_decision = Column(String(10))
    final_confidence = Column(Float)
    expected_price = Column(Float)
    stop_loss = Column(Float)
    target = Column(Float)
    execution_time_ms = Column(Integer)

    # Outcome (filled in by evaluation job the next day)
    actual_outcome = Column(String(10), nullable=True)
    actual_return_pct = Column(Float, nullable=True)
    was_correct = Column(Integer, nullable=True)  # 1/0/-1(pending)


class Document(Base):
    """RAG corpus: annual reports, filings, news, investor decks — chunked + embedded."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), index=True)
    title = Column(String(300))
    doc_type = Column(String(50))      # Annual Report / Quarterly Results / News / Filing / ...
    source_url = Column(String(500), nullable=True)
    published_at = Column(DateTime, nullable=True)
    chunk_text = Column(Text)
    embedding = Column(Vector(384))    # all-MiniLM-L6-v2 dim
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), index=True)
    run_date = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1 = Column(Float)
    buy_accuracy = Column(Float)
    sell_accuracy = Column(Float)
    hold_accuracy = Column(Float)
    sample_size = Column(Integer)


def init_db():
    """Creates tables + pgvector extension. Call once at startup / via migration script."""
    with engine.connect() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
