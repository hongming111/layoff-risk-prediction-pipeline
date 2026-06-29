"""Daily MLOps loop: KS drift detection + prediction vs actual evaluation + CD guardrails."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)

DRIFT_P_THRESHOLD = 0.05       # p < 0.05 → flag as drifted
PRECISION_DEPLOY_THRESHOLD = float(os.getenv("PRECISION_DEPLOY_THRESHOLD", "0.70"))
MLFLOW_MODEL_NAME = "layoff_prediction_model"

_DEFAULT_MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(Path(__file__).resolve().parent.parent / "data/mlflow/mlflow.db").replace("\\", "/"),
)

DRIFT_FEATURES = [
    "vol_14d", "vol_21d",
    "debt_to_equity", "current_ratio", "profit_margin",
    "sentiment_score", "mention_velocity",
    "unemployment_rate_total", "layoff_rate_total",
]


def run_ks_drift_detection(
    baseline_path: str = "data/features/feature_matrix_baseline.parquet",
    current_path: str = "data/features/feature_matrix.parquet",
) -> dict[str, float]:
    """Run per-feature KS tests against a baseline window. Log p-values to MLflow.

    The baseline snapshot is taken once after the first full training run.
    Sudden spikes in, e.g., sector-wide volatility will surface here before
    they silently distort model inputs.
    """
    mlflow.set_tracking_uri(_DEFAULT_MLFLOW_URI)
    baseline = pd.read_parquet(baseline_path)
    current = pd.read_parquet(current_path)

    drift_results: dict[str, float] = {}
    with mlflow.start_run(run_name="ks_drift_check"):
        for col in DRIFT_FEATURES:
            if col not in baseline.columns or col not in current.columns:
                continue
            b_vals = baseline[col].dropna().values
            c_vals = current[col].dropna().values
            if len(b_vals) < 30 or len(c_vals) < 30:
                logger.warning(
                    f"[DRIFT] Skipping '{col}': insufficient non-null samples "
                    f"(baseline={len(b_vals)}, current={len(c_vals)}) — "
                    "refresh baseline with: scripts/refresh_baseline.py"
                )
                continue
            stat, p_val = ks_2samp(b_vals, c_vals)
            drift_results[col] = p_val
            mlflow.log_metric(f"ks_pval_{col}", p_val)
            mlflow.log_metric(f"ks_stat_{col}", stat)

            if p_val < DRIFT_P_THRESHOLD:
                logger.warning(f"[DRIFT] '{col}': KS stat={stat:.4f}, p={p_val:.4f} — distribution shift detected")

    drifted = [k for k, v in drift_results.items() if v < DRIFT_P_THRESHOLD]
    logger.info(f"Drift check complete. {len(drifted)}/{len(drift_results)} features flagged.")
    return drift_results


def evaluate_past_predictions(
    predictions_path: str = "data/processed/predictions_log.parquet",
    warn_path: str = "data/processed/warn_labels.parquet",
    lookback_days: int = 90,
) -> dict[str, float]:
    """Compare predictions made ~90 days ago against newly confirmed WARN labels.

    This closes the 'Prediction vs Actual' loop described in the MLOps spec:
    precision/recall computed on predictions whose target window has now elapsed.
    """
    mlflow.set_tracking_uri(_DEFAULT_MLFLOW_URI)
    preds = pd.read_parquet(predictions_path)
    actuals = pd.read_parquet(warn_path)

    preds["prediction_date"] = pd.to_datetime(preds["prediction_date"])
    preds["target_date"] = pd.to_datetime(preds["target_date"])
    actuals["event_date"] = pd.to_datetime(actuals["event_date"])

    cutoff = preds["prediction_date"].max() - pd.Timedelta(days=lookback_days)
    past_preds = preds[preds["prediction_date"] <= cutoff].copy()

    merged = past_preds.merge(
        actuals[["ticker", "event_date"]].assign(label=1),
        left_on=["ticker", "target_date"],
        right_on=["ticker", "event_date"],
        how="left",
    )
    merged["label"] = merged["label"].fillna(0).astype(int)

    if merged.empty or merged["label"].sum() == 0:
        logger.warning("No confirmed WARN labels in the lookback window — skipping metric computation.")
        return {}

    y_true = merged["label"].values
    y_pred = (merged["score"] >= 0.5).astype(int).values
    y_score = merged["score"].values

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
    }

    with mlflow.start_run(run_name="prediction_vs_actual"):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    logger.info(f"Prediction vs Actual [{lookback_days}d]: {metrics}")
    return metrics


def check_and_deploy(precision_threshold: float = PRECISION_DEPLOY_THRESHOLD) -> bool:
    """CD guardrail: assign @champion alias only if precision >= threshold.

    MLflow 3.x removed model stages and transition_model_version_stage.
    The replacement is named aliases: @champion marks the serving model,
    @challenger marks a candidate under evaluation.
    """
    mlflow.set_tracking_uri(_DEFAULT_MLFLOW_URI)
    metrics = evaluate_past_predictions()
    precision = metrics.get("precision", 0.0)

    client = mlflow.MlflowClient()
    # MLflow 3.x: search_model_versions replaces get_latest_versions(stages=[...])
    try:
        versions = client.search_model_versions(
            filter_string=f"name='{MLFLOW_MODEL_NAME}'",
            order_by=["version_number DESC"],
            max_results=1,
        )
    except Exception:
        versions = []

    if not versions:
        logger.warning("[DEPLOY] No registered model versions found — skipping deployment check.")
        return False

    mv = versions[0]

    if precision >= precision_threshold:
        logger.info(f"[DEPLOY] Approved: precision={precision:.3f} >= {precision_threshold}")
        # Promote latest version to @champion (overwrites any prior assignment)
        client.set_registered_model_alias(
            name=MLFLOW_MODEL_NAME,
            alias="champion",
            version=str(mv.version),
        )
        return True

    logger.warning(
        f"[DEPLOY] BLOCKED: precision={precision:.3f} < {precision_threshold}. "
        "Retaining existing @champion weights."
    )
    return False
