"""Batch inference — scores every tracked ticker against the production model
and persists results to both Postgres (for the dashboard) and a local Parquet
log (for the drift-evaluation DAG).

Called as the final task in Ingestion_Pipeline_DAG after the feature matrix
has been built by core_etl.feature_engineer.build_feature_matrix().
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import date, timedelta
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path as _Path
from sqlalchemy import create_engine, text

from core_etl.schema import FEATURE_COLS
from core_etl.validator import validate_feature_matrix, validate_predictions

_DEFAULT_MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(_Path(__file__).resolve().parent.parent / "data/mlflow/mlflow.db").replace("\\", "/"),
)

logger = logging.getLogger(__name__)

LOOKAHEAD_DAYS: int = int(os.getenv("LOOKAHEAD_DAYS", "90"))
MLFLOW_MODEL_NAME: str = "layoff_prediction_model"
PREDICTIONS_PARQUET: Path = Path("data/processed/predictions_log.parquet")
MODEL_DIR: Path = Path("data/models")


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_production_model() -> tuple:
    """Return (sklearn_model, version_label).

    Load order:
    1. data/models/xgboost.pkl  (best CV metrics; written by train.py)
    2. data/models/random_forest.pkl
    3. MLflow registry (alias champion → challenger → latest version)

    The local-pkl path is primary because MLflow 3.x artifact URIs are not
    reliably resolvable across Docker containers without extra configuration.
    """
    for pkl_name, label in [("xgboost", "xgboost_local"), ("random_forest", "rf_local")]:
        pkl_path = MODEL_DIR / f"{pkl_name}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as fh:
                model = pickle.load(fh)
            logger.info(f"Loaded model from {pkl_path}")
            return model, label

    # MLflow fallback (useful when running outside Docker with direct artifact access)
    mlflow.set_tracking_uri(_DEFAULT_MLFLOW_URI)
    client = mlflow.MlflowClient()

    for alias in ("champion", "challenger"):
        try:
            model = mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}@{alias}")
            logger.info(f"Loaded model from MLflow alias '@{alias}'")
            return model, f"{MLFLOW_MODEL_NAME}@{alias}"
        except Exception:
            pass

    try:
        versions = client.search_model_versions(
            filter_string=f"name='{MLFLOW_MODEL_NAME}'",
            order_by=["version_number DESC"],
            max_results=1,
        )
    except Exception:
        versions = []

    if versions:
        mv = versions[0]
        try:
            model = mlflow.sklearn.load_model(mv.source)
            logger.info(f"Loaded model v{mv.version} via MLflow source URI")
            return model, f"{MLFLOW_MODEL_NAME}/v{mv.version}"
        except Exception as exc:
            logger.warning(f"MLflow source load failed: {exc}")

    raise RuntimeError(
        f"No trained model found. Expected pickle at {MODEL_DIR}/xgboost.pkl or "
        f"a registered MLflow model named '{MLFLOW_MODEL_NAME}'. "
        "Run training first: python -m ml_engine.train"
    )


# ── Persistence helpers ───────────────────────────────────────────────────────

def _write_to_postgres(df: pd.DataFrame, db_url: str) -> None:
    """Upsert prediction rows into predictions_log (idempotent on ticker + prediction_date).

    ON CONFLICT DO UPDATE means re-running the same DAG day — e.g. a retry or
    a manual backfill — overwrites the existing row with the latest score rather
    than appending a duplicate.
    """
    engine = create_engine(db_url)
    records = df.to_dict("records")
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO predictions_log
                    (ticker, prediction_date, target_date, score, model_version)
                VALUES
                    (:ticker, :prediction_date, :target_date, :score, :model_version)
                ON CONFLICT (ticker, prediction_date)
                DO UPDATE SET
                    target_date   = EXCLUDED.target_date,
                    score         = EXCLUDED.score,
                    model_version = EXCLUDED.model_version
            """),
            records,
        )
    logger.info(f"Upserted {len(df)} prediction rows -> Postgres predictions_log")


def _append_to_parquet(df: pd.DataFrame) -> None:
    """Append prediction rows to the local Parquet log used by evaluate_drift."""
    PREDICTIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    if PREDICTIONS_PARQUET.exists():
        prior = pd.read_parquet(PREDICTIONS_PARQUET)
        df = pd.concat([prior, df], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "prediction_date"], keep="last")
    if len(df) < before:
        logger.info(f"Predictions dedup: dropped {before - len(df)} duplicate (ticker, prediction_date) rows")
    df.to_parquet(PREDICTIONS_PARQUET, index=False, compression="snappy")
    logger.info(f"Appended predictions → {PREDICTIONS_PARQUET} ({len(df)} total rows)")


# ── Public entry point ────────────────────────────────────────────────────────

def run_inference(
    feature_path: str = "data/features/feature_matrix.parquet",
    db_url: str | None = None,
    prediction_date: date | None = None,
) -> pd.DataFrame:
    """Score every ticker's latest feature row and write predictions.

    Takes the most recent date-row per ticker from the feature matrix,
    runs predict_proba, and writes one row per ticker to predictions_log
    (both Postgres and Parquet).

    Returns the predictions DataFrame so callers can inspect results.
    """
    pred_date = prediction_date or date.today()
    target_date = pred_date + timedelta(days=LOOKAHEAD_DAYS)

    # ── Load feature matrix: latest row per ticker ────────────────────────────
    feature_df = pd.read_parquet(feature_path)
    # Normalise column names — yfinance saves the date index as 'Date' (capital);
    # DuckDB preserves original case, so older feature matrices may have 'Date'.
    feature_df.columns = [c.lower() for c in feature_df.columns]
    feature_df["date"] = pd.to_datetime(feature_df["date"])

    latest_per_ticker = (
        feature_df
        .sort_values("date")
        .groupby("ticker", as_index=False)
        .last()
    )

    if latest_per_ticker.empty:
        raise RuntimeError(
            "Feature matrix is empty — the ETL step may not have completed. "
            "Check the run_etl task logs in Airflow."
        )

    # Pydantic validation gate — ensures feature matrix is structurally sound
    # before scoring. Does not require positive labels (inference, not training).
    validate_feature_matrix(latest_per_ticker, require_positive_labels=False)

    missing = [c for c in FEATURE_COLS if c not in latest_per_ticker.columns]
    if missing:
        raise RuntimeError(
            f"Feature matrix is missing columns required by the model: {missing}. "
            "Re-run build_feature_matrix() to regenerate the feature store."
        )

    X = latest_per_ticker[FEATURE_COLS].fillna(0).values

    # ── Score ─────────────────────────────────────────────────────────────────
    model, model_version = _load_production_model()
    scores = model.predict_proba(X)[:, 1]

    predictions = pd.DataFrame({
        "ticker":          latest_per_ticker["ticker"].values,
        "prediction_date": pred_date,
        "target_date":     target_date,
        "score":           scores.round(6),
        "model_version":   model_version,
    })

    validate_predictions(predictions)

    logger.info(
        f"Scored {len(predictions)} tickers | "
        f"prediction_date={pred_date} | target_date={target_date} | "
        f"top scorer: {predictions.nlargest(1, 'score')[['ticker', 'score']].to_dict('records')}"
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    url = db_url or os.getenv("DATABASE_URL", "")
    if url:
        _write_to_postgres(predictions, url)
    else:
        logger.warning("DATABASE_URL not set — skipping Postgres write. Set it in .env.")

    _append_to_parquet(predictions)

    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference()
