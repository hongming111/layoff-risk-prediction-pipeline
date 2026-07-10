"""Pydantic data-validation gate for the feature matrix.

Called before model training and inference to catch bad data early:
  - All required columns are present
  - The identifier field (ticker) contains no nulls
  - At least one positive label exists (training only)
  - Key numeric columns are not *entirely* null per ticker
  - Score column is within [0, 1] after inference (predictions gate)

Usage
-----
    from core_etl.validator import validate_feature_matrix, validate_predictions

    validate_feature_matrix(df)           # raises on failure
    validate_predictions(predictions_df)  # raises on failure
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# ── Required column sets ──────────────────────────────────────────────────────

FEATURE_COLS: list[str] = [
    "ticker", "date",
    "close", "vol_7d", "vol_14d", "vol_21d",
    "debt_to_equity", "current_ratio", "profit_margin",
    "sentiment_score", "sentiment_score_ma7d", "mention_velocity",
    "unemployment_rate_total", "layoff_rate_total", "layoff_rate_tech",
]

PREDICTION_COLS: list[str] = [
    "ticker", "prediction_date", "target_date", "score", "model_version",
]

# Columns where a fully-null column for every ticker signals broken ingestion.
# These are the most important market/fundamental signals; macro/sentiment are
# allowed to be sparse because their APIs have higher failure rates.
_CRITICAL_NUMERIC_COLS: list[str] = ["close", "vol_7d", "vol_14d"]


# ── Pydantic row model ────────────────────────────────────────────────────────

class FeatureRow(BaseModel):
    """Validate one row of the feature matrix."""

    ticker: str
    close: float | None = None
    vol_7d: float | None = None

    @field_validator("ticker")
    @classmethod
    def ticker_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ticker must be a non-empty string")
        return v.strip().upper()

    @field_validator("close", "vol_7d", mode="before")
    @classmethod
    def coerce_nan_to_none(cls, v: Any) -> Any:
        import math
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            return None


class PredictionRow(BaseModel):
    """Validate one row of the predictions output."""

    ticker: str
    score: float

    @field_validator("ticker")
    @classmethod
    def ticker_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ticker must be a non-empty string")
        return v

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {v}")
        return v


# ── DataFrame-level gates ─────────────────────────────────────────────────────

def validate_feature_matrix(
    df: pd.DataFrame,
    require_positive_labels: bool = False,
) -> None:
    """Raise ValueError if the feature matrix fails any validation gate.

    Gates (in order):
    1. All FEATURE_COLS are present.
    2. DataFrame is not empty.
    3. `ticker` column has no null or blank values.
    4. Critical numeric columns (close, vol_7d, vol_14d) are not entirely null.
    5. Row-level Pydantic validation on a sample (up to 500 rows).
    6. If require_positive_labels=True, at least one label==1 row exists.
    """
    # Gate 1: column presence
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[validator] Feature matrix is missing required columns: {missing_cols}. "
            "Re-run build_feature_matrix() to regenerate the feature store."
        )

    # Gate 2: non-empty
    if df.empty:
        raise ValueError(
            "[validator] Feature matrix is empty — no rows to train or score. "
            "Check that ingestion tasks completed successfully."
        )

    # Gate 3: ticker nulls
    null_tickers = df["ticker"].isna().sum()
    if null_tickers > 0:
        raise ValueError(
            f"[validator] 'ticker' column contains {null_tickers} null value(s). "
            "Entity resolution may have failed — check ingest_warn and run_etl logs."
        )

    # Gate 4: critical numeric columns entirely null
    for col in _CRITICAL_NUMERIC_COLS:
        if col in df.columns and df[col].isna().all():
            raise ValueError(
                f"[validator] Critical column '{col}' is null for every row. "
                "Market data ingestion likely failed — check ingest_market task logs."
            )

    # Gate 5: row-level Pydantic validation on a sample
    sample = df.sample(min(500, len(df)), random_state=42)
    errors: list[str] = []
    for idx, row in sample.iterrows():
        try:
            FeatureRow(ticker=row["ticker"], close=row.get("close"), vol_7d=row.get("vol_7d"))
        except Exception as exc:
            errors.append(f"row {idx}: {exc}")
    if errors:
        summary = errors[:5]
        raise ValueError(
            f"[validator] {len(errors)} row(s) failed Pydantic validation "
            f"(showing first 5):\n" + "\n".join(summary)
        )

    # Gate 6: positive labels (training only)
    if require_positive_labels:
        if "label" not in df.columns or int(df["label"].sum()) == 0:
            raise ValueError(
                "[validator] Feature matrix has zero positive labels — the WARN scraper "
                "has not collected any layoff notices yet, or seeded labels are missing. "
                "Run scripts/seed_historical_labels.py then re-trigger the DAG."
            )

    logger.info(
        f"[validator] Feature matrix passed all gates: "
        f"{len(df)} rows, {df['ticker'].nunique()} tickers"
        + (f", {int(df['label'].sum())} positive labels" if "label" in df.columns else "")
    )


def validate_predictions(df: pd.DataFrame) -> None:
    """Raise ValueError if the predictions DataFrame fails any validation gate.

    Gates:
    1. All PREDICTION_COLS are present.
    2. DataFrame is not empty.
    3. Row-level Pydantic validation (ticker not blank, score in [0,1]).
    """
    missing_cols = [c for c in PREDICTION_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[validator] Predictions DataFrame missing columns: {missing_cols}."
        )

    if df.empty:
        raise ValueError("[validator] Predictions DataFrame is empty — no tickers were scored.")

    errors: list[str] = []
    for idx, row in df.iterrows():
        try:
            PredictionRow(ticker=str(row["ticker"]), score=float(row["score"]))
        except Exception as exc:
            errors.append(f"row {idx}: {exc}")
    if errors:
        summary = errors[:5]
        raise ValueError(
            f"[validator] {len(errors)} prediction row(s) failed validation "
            f"(showing first 5):\n" + "\n".join(summary)
        )

    logger.info(
        f"[validator] Predictions passed all gates: "
        f"{len(df)} rows, score range [{df['score'].min():.4f}, {df['score'].max():.4f}]"
    )
