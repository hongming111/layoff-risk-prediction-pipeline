"""Temporal alignment and rolling-window feature matrix construction via DuckDB."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from core_etl.schema import PARQUET_SCHEMAS as _FALLBACK_SCHEMAS
from core_etl.feature_store import write_versioned_matrix

logger = logging.getLogger(__name__)

FEATURE_MATRIX_PATH = Path("data/features/feature_matrix.parquet")


def forward_fill_to_daily(
    df: pd.DataFrame,
    date_col: str = "date",
    group_col: str | None = "ticker",
    freq: str = "D",
) -> pd.DataFrame:
    """Upsample an irregular-frequency DataFrame to daily by forward-filling.

    Quarterly fundamentals and monthly macro data both go through this before
    being joined into the unified feature matrix.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if group_col and group_col in df.columns:
        parts = []
        for key, grp in df.groupby(group_col):
            grp = grp.set_index(date_col).sort_index().resample(freq).ffill()
            grp[group_col] = key
            parts.append(grp.reset_index())
        return pd.concat(parts, ignore_index=True)

    return df.set_index(date_col).sort_index().resample(freq).ffill().reset_index()


def add_rolling_windows(
    df: pd.DataFrame,
    col: str,
    windows: tuple[int, ...] = (7, 14, 21),
    group_col: str = "ticker",
) -> pd.DataFrame:
    """Append rolling mean columns for the specified windows."""
    df = df.copy()
    if group_col in df.columns:
        for w in windows:
            df[f"{col}_ma{w}d"] = (
                df.groupby(group_col)[col]
                .transform(lambda s: s.rolling(w, min_periods=1).mean())
            )
    else:
        for w in windows:
            df[f"{col}_ma{w}d"] = df[col].rolling(w, min_periods=1).mean()
    return df


def _ensure_valid_parquet(path: str, schema_key: str) -> None:
    """Replace an empty/schema-less parquet file with a properly-typed empty DataFrame.

    DuckDB raises 'Need at least one non-root column' when reading a parquet
    written from pd.DataFrame() with no columns. Writing a zero-row DataFrame
    with the correct schema lets the LEFT JOIN return NULLs instead of crashing.

    Validates by actually reading the file — a file can be hundreds of bytes of
    parquet metadata yet still have zero columns, so size is not a reliable check.
    """
    p = Path(path)
    needs_fix = False
    if not p.exists():
        needs_fix = True
    else:
        try:
            existing = pd.read_parquet(p)
            if existing.shape[1] == 0:
                needs_fix = True
        except Exception:
            needs_fix = True

    if not needs_fix:
        # Also check that all expected columns are present — a file with only
        # a 'date' column (all metric values were suppressed by the source API)
        # would pass the shape[1] == 0 check but still break the DuckDB JOIN.
        schema = _FALLBACK_SCHEMAS[schema_key]
        existing_cols = set(pd.read_parquet(p).columns)
        if not set(schema.keys()).issubset(existing_cols):
            needs_fix = True

    if needs_fix:
        schema = _FALLBACK_SCHEMAS[schema_key]
        empty = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})
        p.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(p, index=False, compression="snappy")
        logger.warning(f"Replaced incomplete/missing {p.name} with zero-row schema stub")


def build_feature_matrix(
    market_path: str = "data/processed/market.parquet",
    fundamentals_path: str = "data/processed/fundamentals.parquet",
    sentiment_path: str = "data/processed/sentiment.parquet",
    macro_path: str = "data/processed/macro.parquet",
    labels_path: str = "data/processed/warn_labels.parquet",
    output_path: Path = FEATURE_MATRIX_PATH,
) -> pd.DataFrame:
    """JOIN all five data sources on (ticker, date) into a daily feature matrix.

    Uses DuckDB for efficient in-process SQL over Parquet files — avoids
    loading all data into Python memory simultaneously.

    Label logic: a row gets label=1 if a confirmed WARN event falls within
    the next LOOKAHEAD_DAYS (90) calendar days for that ticker.
    """
    # Ensure auxiliary files have at least a schema so DuckDB LEFT JOINs return
    # NULLs rather than crashing on empty/schema-less parquet files.
    _ensure_valid_parquet(fundamentals_path, "fundamentals")
    _ensure_valid_parquet(sentiment_path,    "sentiment")
    _ensure_valid_parquet(macro_path,        "macro")
    _ensure_valid_parquet(labels_path,       "labels")

    # Build query as an explicit str before passing to DuckDB.
    # DuckDB's execute() is NOT SQLAlchemy — it accepts plain strings only.
    # Do not pass sqlalchemy.text() objects here; they are not string-compatible
    # with the DuckDB DBAPI and would raise "Query must be a string".
    query: str = f"""
        SELECT
            m.ticker,
            m.date AS date,
            -- Market features
            m.close,
            m.vol_7d,
            m.vol_14d,
            m.vol_21d,
            -- Fundamental features (forward-filled quarterly)
            f.debt_to_equity,
            f.current_ratio,
            f.profit_margin,
            -- Sentiment features (7-day rolling avg already applied upstream)
            s.sentiment_score,
            s.sentiment_score_ma7d,
            s.mention_velocity,
            -- Macro features (forward-filled monthly)
            mac.unemployment_rate_total,
            mac.layoff_rate_total,
            mac.layoff_rate_tech,
            -- Label: did a WARN event occur within the next 90 days?
            COALESCE(MAX(CASE
                WHEN l.event_date BETWEEN m.date AND m.date + INTERVAL '90 days'
                THEN 1 ELSE 0
            END), 0) AS label
        FROM read_parquet('{market_path}')        m
        LEFT JOIN read_parquet('{fundamentals_path}') f
            ON m.ticker = f.ticker AND m.date    = f.date
        LEFT JOIN read_parquet('{sentiment_path}')    s
            ON m.ticker = s.ticker AND m.date    = s.date
        LEFT JOIN read_parquet('{macro_path}')        mac
            ON m.date = mac.date
        LEFT JOIN read_parquet('{labels_path}')       l
            ON m.ticker = l.ticker
        GROUP BY ALL
        ORDER BY m.ticker, m.date
    """

    # Use an in-memory DuckDB connection scoped to this function call.
    # Closing explicitly prevents handle leaks across repeated Airflow task runs.
    con = duckdb.connect(database=":memory:")
    try:
        feature_df = con.execute(query).df()
    finally:
        con.close()

    # output_path kept for API compatibility; feature_store always writes to
    # data/features/feature_matrix.parquet + a versioned timestamped copy.
    versioned = write_versioned_matrix(feature_df)
    logger.info(f"Feature matrix built: {feature_df.shape} → {versioned.name}")
    return feature_df
