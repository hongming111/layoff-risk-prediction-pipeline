"""
DEMO-ONLY backfill: seeds synthetic 'prediction_vs_actual' MLflow runs spread
over the past ~90 days, so the dashboard's "System Precision Over Time" chart
(app.py) has a trend line to show before 90 days of real production history
has accumulated (evaluate_drift.evaluate_past_predictions only logs a run once
predictions made >=90 days ago can be checked against confirmed WARN labels).

This does NOT touch predictions_log.parquet or warn_labels.parquet — it only
writes fake precision/recall/roc_auc metrics directly to MLflow, tagged
seeded=true so they can be told apart from real evaluation runs later.

Run once from project root (or inside the airflow container where
MLFLOW_TRACKING_URI is already set):
  python scripts/seed_precision_trend.py

To remove afterward once real evaluation runs exist, delete runs tagged
seeded=true from the MLflow UI (or via MlflowClient.delete_run).
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(PROJECT_ROOT / "data/mlflow/mlflow.db").replace("\\", "/"),
)
EXPERIMENT_NAME = "layoff_prediction"
NUM_POINTS = 13
LOOKBACK_DAYS = 90


def main() -> None:
    mlflow.set_tracking_uri(_DEFAULT_MLFLOW_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found — run training first.")

    random.seed(42)
    now = datetime.utcnow()
    step_days = LOOKBACK_DAYS / (NUM_POINTS - 1)
    precision = 0.62

    for i in range(NUM_POINTS):
        run_date = now - timedelta(days=LOOKBACK_DAYS - i * step_days)
        precision = min(0.92, max(0.50, precision + random.uniform(-0.03, 0.05)))
        recall = min(0.95, max(0.40, precision - random.uniform(0.0, 0.15)))
        roc_auc = min(0.97, max(0.55, precision + random.uniform(0.0, 0.08)))

        start_ms = int(run_date.timestamp() * 1000)
        run = client.create_run(
            experiment_id=experiment.experiment_id,
            start_time=start_ms,
            tags={"mlflow.runName": "prediction_vs_actual", "seeded": "true"},
        )
        client.log_metric(run.info.run_id, "precision", round(precision, 4), timestamp=start_ms)
        client.log_metric(run.info.run_id, "recall", round(recall, 4), timestamp=start_ms)
        client.log_metric(run.info.run_id, "roc_auc", round(roc_auc, 4), timestamp=start_ms)
        client.set_terminated(run.info.run_id, end_time=start_ms)
        print(f"{run_date.date()}  precision={precision:.3f}  recall={recall:.3f}  roc_auc={roc_auc:.3f}")

    print(f"\nSeeded {NUM_POINTS} synthetic 'prediction_vs_actual' runs into experiment '{EXPERIMENT_NAME}'.")


if __name__ == "__main__":
    main()
