"""Hourly sentiment ingestion DAG.

Schedule: top of every hour (:00)
Purpose:  Keeps the sentiment parquet fresh between daily full-pipeline runs so
          the dashboard's Data Freshness sidebar reflects near-real-time news
          coverage without triggering the expensive ETL / inference steps.

Tasks:
    [ingest_sentiment]  ──▶  [validate_sentiment]
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

default_args = {
    "owner": "retrenchment_pipeline",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
}

TICKERS = ["AMZN", "GOOG", "META", "MSFT", "AAPL", "NFLX", "SNAP", "LYFT", "UBER", "COIN"]
COMPANY_NAMES = {
    "AMZN": "Amazon",
    "GOOG": "Alphabet Google",
    "META": "Meta Platforms",
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "NFLX": "Netflix",
    "SNAP": "Snap Inc",
    "LYFT": "Lyft",
    "UBER": "Uber Technologies",
    "COIN": "Coinbase",
}

# NewsAPI free tier: 100 req/day total across all DAGs.
# The daily ingestion DAG already uses ~10 req (1 page × 10 tickers).
# This hourly DAG caps at 1 page per ticker (10 req/run × 24 runs = 240 req/day
# worst-case) so we restrict to RSS-only during the hour to stay within quota.
# NewsAPI is called only in the daily DAG; hourly uses RSS + rate-limit guard.
_RSS_ONLY_HOURLY = True


def _ingest_sentiment_hourly(**ctx):
    from pathlib import Path

    import pandas as pd
    from ingestion.rate_limits import remaining
    from ingestion.sentiment import fetch_rss_headlines, persist_to_parquet

    # Fetch only the last 2 hours of RSS headlines to minimise redundancy.
    lookback_days = 1  # RSS feeds don't support sub-day windowing; deduplicate below.

    try:
        rss_df = fetch_rss_headlines(TICKERS, COMPANY_NAMES, lookback_days=lookback_days)
    except Exception as exc:
        logger.warning(f"Hourly RSS fetch failed (non-fatal): {exc}")
        return

    if rss_df.empty:
        logger.info("Hourly RSS returned no new headlines — nothing to write.")
        return

    # ── Optionally append NewsAPI if daily budget is not exhausted ────────────
    if not _RSS_ONLY_HOURLY and remaining("news_api") > 20:
        try:
            from ingestion.sentiment import fetch_news_headlines
            news_df = fetch_news_headlines(TICKERS, COMPANY_NAMES, lookback_days=1)
            combined = news_df.merge(rss_df, on=["ticker", "date"], how="outer")
        except Exception as exc:
            logger.warning(f"Hourly NewsAPI fetch failed (non-fatal): {exc}")
            combined = rss_df.rename(columns={
                "rss_sentiment_score": "sentiment_score",
                "rss_article_count": "article_count",
            })
    else:
        combined = rss_df.rename(columns={
            "rss_sentiment_score": "sentiment_score",
            "rss_article_count": "article_count",
        })

    # ── Merge with prior history to preserve the 7-day rolling window ─────────
    prior_path = Path("data/processed/sentiment.parquet")
    new_rows = combined[["ticker", "date", "sentiment_score"]].copy()
    new_rows["mention_velocity"] = combined.get(
        "mention_velocity", pd.Series(0.0, index=combined.index)
    ).fillna(0.0)

    if prior_path.exists():
        try:
            prior = pd.read_parquet(prior_path)
            prior = prior.drop(columns=["sentiment_score_ma7d"], errors="ignore")
            new_rows = pd.concat([prior, new_rows], ignore_index=True)
        except Exception as exc:
            logger.warning(f"Could not read prior sentiment parquet ({exc}) — starting fresh")

    new_rows["date"] = pd.to_datetime(new_rows["date"])
    new_rows = new_rows.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    )
    new_rows["sentiment_score_ma7d"] = (
        new_rows.groupby("ticker")["sentiment_score"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )

    persist_to_parquet(
        new_rows[["ticker", "date", "sentiment_score", "sentiment_score_ma7d", "mention_velocity"]]
    )
    logger.info(f"Hourly sentiment update complete: {len(new_rows)} rows written.")


def _validate_sentiment_freshness(**ctx):
    """Warn in the Airflow log if sentiment data is stale (> 2 hours old)."""
    from pathlib import Path

    path = Path("data/processed/sentiment.parquet")
    if not path.exists():
        logger.warning("sentiment.parquet does not exist yet — skip freshness check.")
        return

    import time
    age_seconds = time.time() - path.stat().st_mtime
    age_hours = age_seconds / 3600
    if age_hours > 2:
        logger.warning(
            f"sentiment.parquet is {age_hours:.1f} h old — "
            "RSS sources may be unreachable or returning no new entries."
        )
    else:
        logger.info(f"Sentiment freshness OK: {age_hours:.2f} h since last write.")


with DAG(
    dag_id="sentiment_hourly",
    default_args=default_args,
    description="Hourly RSS sentiment refresh — keeps news feed fresh between daily pipeline runs",
    schedule="0 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "sentiment", "hourly"],
) as dag:

    t_sentiment = PythonOperator(
        task_id="ingest_sentiment_hourly",
        python_callable=_ingest_sentiment_hourly,
        execution_timeout=timedelta(minutes=10),
    )

    t_validate = PythonOperator(
        task_id="validate_sentiment_freshness",
        python_callable=_validate_sentiment_freshness,
        execution_timeout=timedelta(minutes=2),
    )

    t_sentiment >> t_validate
