"""Daily multi-modal ingestion DAG.

Schedule: 06:00 UTC daily
Tasks (fan-out then converge):
    [warn] [market] [fundamentals] [sentiment] [macro]  ──▶  [etl]  ──▶  [inference]
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

default_args = {
    "owner": "retrenchment_pipeline",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=60),
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


def _ingest_warn(**ctx):
    from ingestion.warn_scraper import fetch_warn_notices, persist_to_parquet, persist_to_postgres
    import pandas as pd
    from pathlib import Path

    try:
        df = fetch_warn_notices()
    except Exception as exc:
        logger.warning(f"WARN scraper failed (non-fatal): {exc}")
        df = pd.DataFrame(columns=["company", "state", "date", "employees_affected"])

    # Only overwrite existing data if the scraper returned something useful.
    # Preserves manually seeded or previously scraped rows when the live feed fails.
    existing_path = Path("data/processed/warn_raw.parquet")
    if df.empty and existing_path.exists() and existing_path.stat().st_size > 200:
        logger.warning("Scraper returned empty — keeping existing warn_raw.parquet")
        return

    persist_to_parquet(df)
    try:
        persist_to_postgres(df)
    except Exception as exc:
        logger.warning(f"WARN Postgres write failed (non-fatal): {exc}")


def _ingest_market(**ctx):
    import pandas as pd
    from ingestion.checkpoint import read_watermark
    from ingestion.market import compute_rolling_volatility, fetch_ohlcv, persist_to_parquet

    # Incremental: start from the last persisted date; first run falls back to 2 years.
    # Subtract 3 days to overlap and capture any late-arriving corrections.
    watermark = read_watermark("data/processed/market.parquet", fallback_days=730)
    from datetime import timedelta
    start = (watermark - timedelta(days=3)).isoformat()
    logger.info(f"Market incremental fetch: start={start} (watermark={watermark})")

    raw = fetch_ohlcv(TICKERS, start=start)
    # Flatten multi-ticker download and add volatility columns
    frames = []
    for ticker in TICKERS:
        try:
            close = raw[ticker]["Close"] if ticker in raw.columns.get_level_values(0) else raw["Close"][ticker]
            vol = compute_rolling_volatility(close)
            df = pd.DataFrame({"ticker": ticker, "close": close}).join(vol)
            frames.append(df)
        except KeyError:
            pass

    if frames:
        new_df = pd.concat(frames)
        # Reset the DatetimeIndex (named 'Date' by yfinance) to a plain column
        # before merging with prior data — prior parquet already has a flat 'date'
        # column, so both sides must be column-aligned before concat.
        if new_df.index.name and new_df.index.name.lower() == "date":
            new_df = new_df.reset_index()
            new_df.columns = [c.lower() for c in new_df.columns]
        # Merge with existing data: read prior parquet, append new rows, dedup in persist.
        from pathlib import Path
        prior_path = Path("data/processed/market.parquet")
        if prior_path.exists():
            prior = pd.read_parquet(prior_path)
            new_df = pd.concat([prior, new_df], ignore_index=True)
        persist_to_parquet(new_df)


def _ingest_fundamentals(**ctx):
    from ingestion.fundamentals import fetch_fundamentals, persist_to_parquet

    df = fetch_fundamentals(TICKERS)
    persist_to_parquet(df)


def _ingest_sentiment(**ctx):
    from datetime import date, timedelta
    from pathlib import Path

    import pandas as pd
    from ingestion.checkpoint import read_watermark
    from ingestion.sentiment import fetch_news_headlines, fetch_rss_headlines, persist_to_parquet

    # Incremental: fetch only since last persisted sentiment date.
    # Clamp to [1, 30] days so we never request more than the NewsAPI free tier
    # can serve or fewer than one day's worth of new headlines.
    watermark = read_watermark("data/processed/sentiment.parquet", fallback_days=7)
    lookback_days = max(1, min(30, (date.today() - watermark).days + 1))
    logger.info(f"Sentiment incremental fetch: lookback_days={lookback_days} (watermark={watermark})")

    try:
        news_df = fetch_news_headlines(TICKERS, COMPANY_NAMES, lookback_days=lookback_days)
    except Exception as exc:
        logger.warning(f"NewsAPI fetch failed (non-fatal): {exc}")
        news_df = pd.DataFrame(columns=["ticker", "date", "sentiment_score", "article_count"])

    try:
        rss_df = fetch_rss_headlines(TICKERS, COMPANY_NAMES, lookback_days=lookback_days)
    except Exception as exc:
        logger.warning(f"RSS fetch failed (non-fatal): {exc}")
        rss_df = pd.DataFrame(columns=["ticker", "date", "rss_sentiment_score", "rss_article_count"])

    if news_df.empty and rss_df.empty:
        logger.warning("Both NewsAPI and RSS returned no data — skipping sentiment write")
        return

    combined = news_df.merge(rss_df, on=["ticker", "date"], how="outer")
    combined["sentiment_score"] = (
        combined["sentiment_score"].fillna(0) * combined["article_count"].fillna(0)
        + combined["rss_sentiment_score"].fillna(0) * combined["rss_article_count"].fillna(0)
    ) / (combined["article_count"].fillna(0) + combined["rss_article_count"].fillna(0) + 1e-9)
    combined["mention_velocity"] = combined.get("mention_velocity", pd.Series(0.0, index=combined.index)).fillna(0.0)

    # Keep only the columns the feature engineer expects; merge with prior history
    # so the 7-day rolling mean has sufficient context from previous days.
    new_rows = combined[["ticker", "date", "sentiment_score", "mention_velocity"]].copy()
    prior_path = Path("data/processed/sentiment.parquet")
    if prior_path.exists():
        try:
            prior = pd.read_parquet(prior_path)
            # Drop the derived rolling column — it will be recomputed below.
            prior = prior.drop(columns=["sentiment_score_ma7d"], errors="ignore")
            new_rows = pd.concat([prior, new_rows], ignore_index=True)
        except Exception as exc:
            logger.warning(f"Could not read prior sentiment parquet ({exc}) — starting fresh")

    new_rows["date"] = pd.to_datetime(new_rows["date"])
    new_rows = new_rows.sort_values(["ticker", "date"])
    new_rows["sentiment_score_ma7d"] = (
        new_rows.groupby("ticker")["sentiment_score"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    persist_to_parquet(new_rows[["ticker", "date", "sentiment_score", "sentiment_score_ma7d", "mention_velocity"]])


def _ingest_macro(**ctx):
    from ingestion.macro import fetch_bls_series, persist_to_parquet

    df = fetch_bls_series()
    persist_to_parquet(df)


def _run_etl(**ctx):
    import shutil
    import subprocess
    from pathlib import Path

    import duckdb
    import pandas as pd

    from core_etl.entity_resolver import resolve_entities

    # ─── Step 1: Entity resolution (unchanged) ──────────────────────────────
    warn_raw_path = Path("data/processed/warn_raw.parquet")
    if warn_raw_path.exists() and warn_raw_path.stat().st_size > 200:
        warn_df = pd.read_parquet(warn_raw_path)
    else:
        logger.warning(
            "warn_raw.parquet missing or empty — warn labels will be all-zero. "
            "Check ingest_warn task logs for scraper errors."
        )
        warn_df = pd.DataFrame(columns=["company", "state", "date", "employees_affected"])

    resolved = resolve_entities(warn_df, COMPANY_NAMES)
    labels_path = Path("data/processed/warn_labels.parquet")
    if not resolved.empty:
        labels = resolved[["ticker", "date", "employees_affected"]].rename(
            columns={"date": "event_date"}
        )
        labels.to_parquet(labels_path, index=False)
        logger.info(f"Wrote {len(labels)} warn labels -> {labels_path}")
    elif labels_path.exists() and labels_path.stat().st_size > 200:
        logger.warning("Entity resolution returned nothing — keeping existing warn_labels.parquet")
    else:
        pd.DataFrame(columns=["ticker", "event_date", "employees_affected"]).to_parquet(
            labels_path, index=False
        )

    # ─── Step 2: Feature matrix via dbt ──────────────────────────────────────
    # dbt builds staging views (read_parquet), ASOF-joins them in the intermediate
    # model, and materialises the mart as a DuckDB table with the 90-day label.
    # Python then exports the table to the Parquet path the ML engine reads.
    Path("data/features").mkdir(parents=True, exist_ok=True)

    dbt_result = subprocess.run(
        [
            "dbt", "run",
            "--project-dir", "/opt/airflow/dbt",
            "--profiles-dir", "/opt/airflow/dbt",
        ],
        capture_output=True,
        text=True,
        cwd="/opt/airflow",
    )

    if dbt_result.returncode != 0:
        logger.error(
            f"dbt run failed (exit {dbt_result.returncode}):\n"
            f"STDOUT:\n{dbt_result.stdout}\nSTDERR:\n{dbt_result.stderr}"
        )
        logger.warning("Falling back to legacy build_feature_matrix()")
        from core_etl.feature_engineer import build_feature_matrix
        build_feature_matrix()
    else:
        logger.info(f"dbt run succeeded:\n{dbt_result.stdout}")
        con = duckdb.connect("/opt/airflow/data/dbt.duckdb", read_only=True)
        try:
            feature_df = con.execute(
                "SELECT * FROM main.mart_layoff_risk_features ORDER BY ticker, date"
            ).df()
        finally:
            con.close()
        feature_path = Path("data/features/feature_matrix.parquet")
        from core_etl.feature_store import write_versioned_matrix
        versioned = write_versioned_matrix(feature_df)
        logger.info(f"Exported dbt mart → {versioned.name} ({feature_df.shape})")

    # ─── Step 3: Baseline snapshot for drift detection ───────────────────────
    current = Path("data/features/feature_matrix.parquet")
    baseline = Path("data/features/feature_matrix_baseline.parquet")
    if current.exists() and not baseline.exists():
        shutil.copy(current, baseline)
        logger.info(f"Created drift baseline snapshot → {baseline}")


def _validate_etl(**ctx):
    """Run dbt data quality tests against the mart model."""
    import subprocess

    result = subprocess.run(
        [
            "dbt", "test",
            "--project-dir", "/opt/airflow/dbt",
            "--profiles-dir", "/opt/airflow/dbt",
            "--select", "mart_layoff_risk_features",
        ],
        capture_output=True,
        text=True,
        cwd="/opt/airflow",
    )
    if result.returncode != 0:
        logger.error(
            f"dbt test FAILED (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        raise RuntimeError("dbt data quality tests failed — check Airflow logs for details")
    logger.info(f"dbt test passed:\n{result.stdout}")


def _run_inference(**ctx):
    from ml_engine.predict import run_inference

    try:
        preds = run_inference()
        logger.info(f"Inference task complete: {len(preds)} tickers written to predictions_log")
    except RuntimeError as exc:
        # A missing model is expected on first run before training has been done.
        # Log clearly and let the task succeed so the rest of the DAG is not
        # blocked — the dashboard will show the "no data yet" message until
        # the user runs: python -m ml_engine.train
        logger.warning(f"Inference skipped: {exc}")


with DAG(
    dag_id="ingestion_pipeline",
    default_args=default_args,
    description="Multi-modal ingestion: WARN + market + fundamentals + sentiment + macro → feature matrix → predictions",
    schedule="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "etl", "inference"],
) as dag:

    t_warn        = PythonOperator(task_id="ingest_warn",        python_callable=_ingest_warn,        execution_timeout=timedelta(minutes=45))
    t_market      = PythonOperator(task_id="ingest_market",      python_callable=_ingest_market,      execution_timeout=timedelta(minutes=10))
    t_fundamentals= PythonOperator(task_id="ingest_fundamentals",python_callable=_ingest_fundamentals,execution_timeout=timedelta(minutes=20))
    t_sentiment   = PythonOperator(task_id="ingest_sentiment",   python_callable=_ingest_sentiment,   execution_timeout=timedelta(minutes=30))
    t_macro       = PythonOperator(task_id="ingest_macro",       python_callable=_ingest_macro,       execution_timeout=timedelta(minutes=10))
    t_etl         = PythonOperator(task_id="run_etl",            python_callable=_run_etl,            execution_timeout=timedelta(minutes=15))
    t_validate    = PythonOperator(task_id="validate_etl",       python_callable=_validate_etl,       execution_timeout=timedelta(minutes=5))
    t_inference   = PythonOperator(task_id="run_inference",      python_callable=_run_inference,      execution_timeout=timedelta(minutes=10))

    [t_warn, t_market, t_fundamentals, t_sentiment, t_macro] >> t_etl >> t_validate >> t_inference
