# Corporate Layoff Risk Monitor

A multi-modal ML pipeline that predicts the probability of a company executing
a mass layoff within a 90-day forward window, fusing government filings,
market data, company fundamentals, news/social sentiment, and macro labor
data into a single feature store — with full MLOps tracking, drift
monitoring, and a closed prediction-vs-actual feedback loop.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/hongming111/layoff-risk-monitor-demo) · **Model:** [Hugging Face Hub](https://huggingface.co/hongming111/layoff-risk-xgboost)

---

## Highlights

- Designed and built a multi-source ETL pipeline ingesting 5 heterogeneous
  data domains (government WARN filings, market OHLCV, financial
  fundamentals, news/RSS sentiment, BLS macro series) via 8 parallel Airflow
  tasks into a unified daily feature store.
- Architected a temporal alignment and entity-resolution layer using
  dbt-core + DuckDB with ASOF joins to reconcile mismatched update
  frequencies (quarterly fundamentals, monthly macro, daily prices) and
  RapidFuzz-based fuzzy matching to map unstructured company names to stock
  tickers.
- Engineered pipeline reliability into every ingestion task, including
  tenacity-based retry with exponential backoff, incremental high-watermark
  loading, checkpoint and resume state tracking, and idempotent PostgreSQL
  upserts to guarantee safe reruns.
- Enforced data quality gates end-to-end with Pydantic v2 schema validation
  and automated dbt test suites, and deployed the full stack as
  containerised microservices (Postgres, Airflow, MLflow, Streamlit) via
  Docker Compose with versioned, hash-checked Parquet feature snapshots.

## Architecture

```
[ Daily WARN Scraping ]      [ NewsAPI/Reddit ]    [ yfinance/AlphaVantage ]
        │                           │                         │
   (Incremental)                 (Hourly)                  (Daily)
        ▼                           ▼                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Orchestration Layer (Apache Airflow)                  │
└────────────────────────────────────────────────────────────────────────┘
        │                           │                         │
        ▼                           ▼                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│             ETL Processing Layer (dbt-core + DuckDB / Pandas)          │
│  • Entity Resolution (Fuzzy matching Company Strings -> Stock Tickers) │
│  • Temporal Feature Alignment (ASOF joins across mismatched cadences)  │
│  • Stateful Rolling Windows (7d Sentiment Moving Avg, 14d Volatility)  │
└────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Storage & Feature Store Layer                        │
│       • Postgres (Structured Aggregates & Relational Metadata)         │
│       • Hash-versioned Local Parquet Feature Files                     │
└────────────────────────────────────────────────────────────────────────┘
```

An XGBoost classifier is trained on the resulting feature matrix, tracked
via MLflow (params, metrics, ROC/PR curves), and served through a Streamlit
dashboard with real-time distress-score alerts. A daily Airflow DAG closes
the loop: it checks predictions made 60–90 days ago against newly published
WARN data, updates live precision/recall, and gates auto-deployment of
retrained models below a precision threshold.

## Data sources

| Domain | Source | Cadence |
|---|---|---|
| Ground truth labels | State WARN Act notices (`warn-scraper`) | Incremental/daily |
| Market & price | `yfinance` OHLCV + volatility | Daily |
| Fundamentals | Alpha Vantage / Financial Modeling Prep | Quarterly |
| Text sentiment | NewsAPI + RSS feeds, VADER scoring | Hourly |
| Macro headwinds | US Bureau of Labor Statistics API | Monthly |

## Tech stack

Airflow · dbt-core · DuckDB · Pandas · XGBoost · scikit-learn · MLflow ·
PostgreSQL · Streamlit · Docker Compose · Pydantic v2 · pytest

## Running locally

```
docker compose up -d
```

Brings up Postgres, Airflow (webserver + scheduler), MLflow, and the
Streamlit dashboard. See `CLAUDE.md` for the full architecture rationale
and `huggingface_space/` for a standalone demo that runs without any of
the above services.

## Repository structure

```
├── airflow/dags/        # Ingestion & MLOps evaluation DAGs
├── core_etl/             # Entity resolution, validation, feature store
├── dbt/                  # Temporal alignment models (ASOF joins) + tests
├── ingestion/             # Per-source fetchers (retry, checkpoint/resume)
├── ml_engine/             # Training, inference, drift evaluation
├── dashboard/             # Streamlit monitoring app
├── huggingface_space/     # Standalone demo for Hugging Face Spaces
├── model_card/            # Model card for the Hugging Face Hub
└── scripts/               # Utility & publishing scripts
```
