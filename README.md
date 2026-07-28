# Stock Intelligence Platform — Backend

Real pipeline: **yfinance** (market data) → **pandas/numpy** (Python indicators) →
**XGBoost** (trained on real 5y history) → **pgvector + Gemini embeddings** (RAG) →
**Gemini** (grounded reasoning) → **LangGraph** (orchestrates all of it, node-by-node,
auditable) → **Postgres** (every prediction persisted for replay/evaluation).

> **Free-tier note:** embeddings run through Gemini's API rather than a local
> sentence-transformers/torch model. Torch alone uses several hundred MB of RAM at
> idle, which exceeds Render's free 512MB instance before it can even serve a
> request. Using Gemini for both embeddings and reasoning keeps the whole backend
> comfortably inside 512MB.

## 1. Local setup

```bash
cp .env.example .env
# edit .env: add GEMINI_API_KEY

docker compose up -d db          # Postgres + pgvector
pip install -r requirements.txt
python -m app.ml.train            # trains XGBoost on 5y real TCS data, writes model.json
python -m app.rag.ingest --file docs/tcs_q1_fy27.txt --type "Quarterly Results" --title "TCS Q1 FY27 Results"
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/health` → `{"status":"ok"}`
Then `POST http://localhost:8000/api/v1/predict?ticker=TCS.NS` runs the full pipeline.

## 2. Feeding real RAG documents

`app/rag/ingest.py` accepts any `.txt`/plain-text file or URL. Practical sources for TCS:
- Annual reports / investor presentations: download PDFs from tcs.com/investors, convert to text (`pdftotext`), ingest
- Quarterly results transcripts: same site, "Financial Results" section
- Corporate filings: BSE/NSE announcement pages
- News: wire a news API (NewsAPI.org free tier, or GNews) into a scheduled ingestion job in `app/core/scheduler.py`

Each ingested doc is chunked (800 words, 120 overlap) and embedded via Gemini's
`text-embedding-004` model — one API call per chunk. This is the same key you already
set for reasoning; ingesting a few dozen documents costs negligible quota on Gemini's
free tier.

## 3. Training the ML model on real data

```bash
python -m app.ml.train
```
Pulls 5 years of daily TCS OHLCV via yfinance, builds features from Python-computed
indicators, labels each day by its realized 5-day-forward return (BUY/HOLD/SELL bucketed
at ±0.5%), trains XGBoost, saves `app/ml/model.json` + `feature_importance.json` with
real accuracy/precision/recall/F1 on a held-out test split. Re-run weekly/monthly (see
`app/core/scheduler.py` to automate).

## 4. Daily evaluation loop

```bash
python -m app.core.evaluation
```
Scores every prediction ≥5 trading sessions old against what the stock actually did,
writes an `EvaluationRun` row. Wire this into `app/core/scheduler.py` (already scheduled
via `EVAL_CRON_HOUR`/`EVAL_CRON_MINUTE` in `.env`) so it runs automatically after market close.

## 5. Deploy

**Backend → Render** (recommended, Docker-native, has `render.yaml` here):
```bash
# push this repo, connect it in Render dashboard, it reads render.yaml automatically
# set GEMINI_API_KEY and DATABASE_URL as secrets in Render's dashboard — never in code
```

**Database → Supabase** (free tier, pgvector built in):
1. Create a Supabase project
2. Enable the `vector` extension (SQL editor: `create extension if not exists vector;`)
3. Copy the connection string into `DATABASE_URL`

**Frontend → Netlify** (already covered separately): set `API_BASE_URL` in the React
artifact / Next.js app to your Render backend's public URL, add it to `CORS_ORIGINS` in
the backend `.env`.

## 6. What's mocked vs. real right now

| Component | Status |
|---|---|
| Market data (OHLCV) | **Real** — live yfinance pull |
| Python indicators (RSI/EMA/MACD/ADX/etc.) | **Real** — computed from actual data |
| XGBoost prediction | **Real**, once you run `train.py` once against real history |
| RAG retrieval | **Real** mechanism (pgvector cosine search), but the corpus is empty until you run `ingest.py` against actual filings/news |
| AI reasoning (Gemini) | **Real**, requires your `GEMINI_API_KEY` |
| Agent consensus | **Real** — rule-based, votes derived from the actual upstream values, not separately faked |
| Evaluation metrics | **Real**, but need ≥5 trading days of accumulated predictions before the first run produces output |
