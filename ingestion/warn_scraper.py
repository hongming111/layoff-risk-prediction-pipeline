"""Ground truth label ingestion — WARN Act notifications via warn-scraper."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
def _scrape_state(state: str) -> pd.DataFrame:
    from warn import scraper  # warn-scraper package

    df = scraper.run(state)
    df["state"] = state
    return df


def fetch_warn_notices(states: Optional[list[str]] = None) -> pd.DataFrame:
    """Scrape WARN Act notices for all (or a subset of) US states.

    Returns DataFrame with columns: company, state, date, employees_affected.
    """
    try:
        from warn import scraper
    except ImportError:
        raise ImportError("Run: pip install warn-scraper")

    target_states = states or list(scraper.STATES.keys())
    results: list[pd.DataFrame] = []

    STATE_TIMEOUT_SECS = 60  # max seconds to wait for a single state scrape

    done = load_checkpoint("warn_scraper")
    for state in target_states:
        if state in done:
            logger.info(f"WARN [{state}]: skipping (checkpoint)")
            continue
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_scrape_state, state)
                df = future.result(timeout=STATE_TIMEOUT_SECS)
            results.append(df)
            done.add(state)
            save_checkpoint("warn_scraper", done)
            logger.info(f"WARN [{state}]: {len(df)} records")
        except FuturesTimeoutError:
            logger.warning(f"WARN [{state}]: timed out after {STATE_TIMEOUT_SECS}s — skipping")
        except Exception as exc:
            logger.warning(f"WARN [{state}]: failed — {exc}")

    # Load previously-checkpointed results from today's partial parquet (if any)
    # so a mid-run resume returns the full accumulated dataset, not just the
    # tail fetched in this invocation.
    from pathlib import Path as _Path
    _prior_path = _Path("data/processed/warn_raw.parquet")
    if _prior_path.exists() and _prior_path.stat().st_size > 200:
        try:
            prior_df = pd.read_parquet(_prior_path)
            if not prior_df.empty:
                results.insert(0, prior_df)
        except Exception:
            pass

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True).drop_duplicates()
    clear_checkpoint("warn_scraper")
    combined["date"] = pd.to_datetime(combined.get("date"), errors="coerce")
    return combined


def persist_to_parquet(df: pd.DataFrame, path: str = "data/processed/warn_raw.parquet") -> None:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="snappy")
    logger.info(f"WARN raw data → {out} ({len(df)} rows)")


def persist_to_postgres(df: pd.DataFrame, table: str = "warn_notices", db_url: Optional[str] = None) -> None:
    """Upsert WARN notices into Postgres (idempotent on company + state + event_date).

    ON CONFLICT DO UPDATE means re-scraping the same state on a retry does not
    produce duplicate rows — it refreshes employees_affected instead.
    """
    import os

    url = db_url or os.getenv("DATABASE_URL", "")
    if not url:
        logger.warning("DATABASE_URL not set — skipping Postgres write")
        return

    required = {"company", "state", "date", "employees_affected"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(f"WARN Postgres write skipped — missing columns: {missing}")
        return

    sub = (
        df[["company", "state", "date", "employees_affected"]]
        .rename(columns={"date": "event_date"})
    )
    records = sub.where(pd.notnull(sub), other=None).to_dict("records")

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO warn_notices (company, state, event_date, employees_affected)
                VALUES (:company, :state, :event_date, :employees_affected)
                ON CONFLICT (company, state, event_date)
                DO UPDATE SET employees_affected = EXCLUDED.employees_affected
            """),
            records,
        )
    logger.info(f"Upserted {len(records)} WARN rows → {table}")
