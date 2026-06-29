"""Daily MLOps evaluation DAG.

Schedule: 09:00 AM UTC daily (per spec)
Pipeline:
    [detect_drift]  ──▶  [evaluate_predictions]  ──▶  [check_deployment_guardrails]
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "retrenchment_pipeline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _detect_drift(**ctx):
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    baseline = Path("data/features/feature_matrix_baseline.parquet")
    current = Path("data/features/feature_matrix.parquet")

    if not baseline.exists():
        logger.warning("Baseline feature matrix not found — skipping drift check. "
                       "It will be created automatically after the next ingestion run.")
        return
    if not current.exists():
        logger.warning("Current feature matrix not found — skipping drift check.")
        return

    from ml_engine.evaluate_drift import run_ks_drift_detection

    drift = run_ks_drift_detection()
    drifted = [k for k, v in drift.items() if v < 0.05]
    if drifted:
        ctx["ti"].xcom_push(key="drifted_features", value=drifted)


def _evaluate_predictions(**ctx):
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    if not Path("data/processed/predictions_log.parquet").exists():
        logger.warning("predictions_log.parquet not found — skipping evaluation. "
                       "Trigger the ingestion DAG and wait for run_inference to complete.")
        return
    if not Path("data/processed/warn_labels.parquet").exists():
        logger.warning("warn_labels.parquet not found — skipping evaluation.")
        return

    from ml_engine.evaluate_drift import evaluate_past_predictions

    metrics = evaluate_past_predictions(lookback_days=90)
    ctx["ti"].xcom_push(key="eval_metrics", value=metrics)


def _check_deployment_guardrails(**ctx):
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    if not Path("data/processed/predictions_log.parquet").exists():
        logger.warning("No predictions data yet — skipping deployment guardrail check.")
        return

    from ml_engine.evaluate_drift import check_and_deploy

    deployed = check_and_deploy()
    if not deployed:
        logger.warning(
            "Auto-deployment blocked due to precision below threshold. "
            "Manual review required before promoting staging model."
        )


with DAG(
    dag_id="mlops_evaluation",
    default_args=default_args,
    description="Daily: KS drift detection → prediction vs actual → CD guardrail check",
    schedule="0 9 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "evaluation", "drift"],
) as dag:

    t_drift = PythonOperator(task_id="detect_drift", python_callable=_detect_drift)
    t_eval = PythonOperator(task_id="evaluate_predictions", python_callable=_evaluate_predictions)
    t_deploy = PythonOperator(task_id="check_deployment_guardrails", python_callable=_check_deployment_guardrails)

    t_drift >> t_eval >> t_deploy
