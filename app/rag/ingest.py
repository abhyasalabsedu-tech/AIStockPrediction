"""
Ingests documents into the vector store. Point it at real filings/news for TCS:

    python -m app.rag.ingest --file path/to/tcs_annual_report.txt --type "Annual Report" --title "TCS Annual Report FY26"
    python -m app.rag.ingest --url https://www.bseindia.com/... --type "Corporate Filing" --title "..."

For news, wire in a news API (NewsAPI, GNews, or NSE/BSE RSS feeds) and call `ingest_document`
per article on a schedule (see app/core/scheduler.py).
"""
import argparse
from datetime import datetime
import httpx

from app.models.schema import Document, SessionLocal
from app.rag.embed import chunk_text, embed_batch


def ingest_document(title: str, doc_type: str, raw_text: str, ticker: str = "TCS", source_url: str = None):
    chunks = chunk_text(raw_text)
    if not chunks:
        return 0
    vectors = embed_batch(chunks)

    db = SessionLocal()
    try:
        for chunk, vec in zip(chunks, vectors):
            db.add(Document(
                ticker=ticker, title=title, doc_type=doc_type,
                source_url=source_url, published_at=datetime.utcnow(),
                chunk_text=chunk, embedding=vec,
            ))
        db.commit()
    finally:
        db.close()
    return len(chunks)


def ingest_from_url(url: str, title: str, doc_type: str, ticker: str = "TCS"):
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    return ingest_document(title, doc_type, resp.text, ticker, source_url=url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Local text file to ingest")
    parser.add_argument("--url", help="URL to fetch and ingest")
    parser.add_argument("--title", required=True)
    parser.add_argument("--type", required=True, dest="doc_type")
    parser.add_argument("--ticker", default="TCS")
    args = parser.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8").read()
        n = ingest_document(args.title, args.doc_type, text, args.ticker)
    elif args.url:
        n = ingest_from_url(args.url, args.title, args.doc_type, args.ticker)
    else:
        raise SystemExit("Provide --file or --url")

    print(f"Ingested {n} chunks for '{args.title}'")
