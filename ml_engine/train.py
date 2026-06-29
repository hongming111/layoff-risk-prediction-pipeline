"""XGBoost / RandomForest training pipeline with full MLflow tracking."""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe inside Docker / Airflow
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from core_etl.schema import FEATURE_COLS, LABEL_COL
from core_etl.validator import validate_feature_matrix
from core_etl.feature_store import get_version_id

# When MLFLOW_TRACKING_URI is not set (i.e. Docker isn't running) fall back to
# a local file store under <project_root>/mlruns so training works without a server.
_DEFAULT_MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(Path(__file__).resolve().parent.parent / "data/mlflow/mlflow.db").replace("\\", "/"),
)

logger = logging.getLogger(__name__)

LOOKAHEAD_DAYS = 90
MLFLOW_MODEL_NAME = "layoff_prediction_model"
MODEL_DIR = Path("data/models")


def load_feature_matrix(path: str = "data/features/feature_matrix.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def _log_feature_importance(clf, feature_names: list[str]) -> None:
    if hasattr(clf, "feature_importances_"):
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": clf.feature_importances_}
        ).sort_values("importance", ascending=False)
        tmp = os.path.join(tempfile.gettempdir(), "feature_importance.csv")
        importance_df.to_csv(tmp, index=False)
        mlflow.log_artifact(tmp, artifact_path="diagnostics")


def _log_plots(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> None:
    """Log ROC curve and Precision-Recall curve PNGs as MLflow artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # ── ROC curve ─────────────────────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, color="#2F81F7", label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {model_name}")
        ax.legend(loc="lower right")
        fig.tight_layout()
        roc_path = os.path.join(tmpdir, "roc_curve.png")
        fig.savefig(roc_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # ── Precision-Recall curve ─────────────────────────────────────────────
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec_vals, prec_vals, lw=2, color="#3FB950", label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve — {model_name}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        pr_path = os.path.join(tmpdir, "pr_curve.png")
        fig.savefig(pr_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(pr_path, artifact_path="plots")


def _log_report(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> None:
    """Log a text classification report as an MLflow artifact."""
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(
        y_true, y_pred,
        target_names=["no_layoff", "layoff"],
        digits=4,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix=f"{model_name}_report_"
    ) as f:
        f.write(f"Classification Report — {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        tmp_path = f.name
    mlflow.log_artifact(tmp_path, artifact_path="reports")
    os.unlink(tmp_path)


def train(
    feature_path: str = "data/features/feature_matrix.parquet",
    experiment_name: str = "layoff_prediction",
    n_splits: int = 5,
    register_model: bool = True,
) -> None:
    """Train XGBoost and RandomForest with stratified CV; log everything to MLflow."""
    mlflow.set_tracking_uri(_DEFAULT_MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    df = load_feature_matrix(feature_path)

    # Pydantic validation gate — checks column presence, null identifiers,
    # critical numeric columns, and positive label count before any compute.
    validate_feature_matrix(df, require_positive_labels=True)

    try:
        _feature_version = get_version_id(feature_path)
    except Exception:
        _feature_version = "unknown"

    X = df[FEATURE_COLS].fillna(0).values
    y = df[LABEL_COL].values

    models: dict[str, object] = {
        "xgboost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        ),
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for model_name, clf in models.items():
        with mlflow.start_run(run_name=model_name):
            # ── Parameters ────────────────────────────────────────────────────
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("feature_matrix_version", _feature_version)
            mlflow.log_param("feature_matrix_path", feature_path)
            mlflow.log_params(clf.get_params())   # n_estimators, learning_rate, etc.
            mlflow.log_param("n_cv_splits", n_splits)
            mlflow.log_param("lookahead_days", LOOKAHEAD_DAYS)
            mlflow.log_param("n_features", len(FEATURE_COLS))
            mlflow.log_param("n_positive_labels", int(y.sum()))

            # ── Cross-validation ──────────────────────────────────────────────
            aucs, precisions, recalls, aps = [], [], [], []
            rmses, maes, accuracies = [], [], []
            oof_probs = np.zeros(len(y))   # out-of-fold probabilities for plots

            for _, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
                clf.fit(X[tr_idx], y[tr_idx])
                y_prob = clf.predict_proba(X[val_idx])[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                aucs.append(roc_auc_score(y[val_idx], y_prob))
                precisions.append(precision_score(y[val_idx], y_pred, zero_division=0))
                recalls.append(recall_score(y[val_idx], y_pred, zero_division=0))
                aps.append(average_precision_score(y[val_idx], y_prob))
                rmses.append(float(np.sqrt(mean_squared_error(y[val_idx], y_prob))))
                maes.append(mean_absolute_error(y[val_idx], y_prob))
                accuracies.append(accuracy_score(y[val_idx], y_pred))
                oof_probs[val_idx] = y_prob

            # ── Metrics ───────────────────────────────────────────────────────
            mlflow.log_metric("roc_auc_mean",       float(np.mean(aucs)))
            mlflow.log_metric("roc_auc_std",        float(np.std(aucs)))
            mlflow.log_metric("precision_mean",     float(np.mean(precisions)))
            mlflow.log_metric("recall_mean",        float(np.mean(recalls)))
            mlflow.log_metric("avg_precision_mean", float(np.mean(aps)))
            mlflow.log_metric("rmse_mean",          float(np.mean(rmses)))
            mlflow.log_metric("mae_mean",           float(np.mean(maes)))
            mlflow.log_metric("accuracy_mean",      float(np.mean(accuracies)))

            # ── Retrain on full dataset before saving ─────────────────────────
            clf.fit(X, y)

            # ── Artifacts: feature importance, plots, report ──────────────────
            _log_feature_importance(clf, FEATURE_COLS)
            _log_plots(y, oof_probs, model_name)
            _log_report(y, oof_probs, model_name)

            # ── Save model to shared volume (Airflow containers load from here) ─
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODEL_DIR / f"{model_name}.pkl"
            with open(model_path, "wb") as fh:
                pickle.dump(clf, fh)
            mlflow.log_param("local_model_path", str(model_path))
            logger.info(f"Saved {model_name} -> {model_path}")

            if register_model:
                try:
                    mlflow.sklearn.log_model(
                        clf,
                        artifact_path=model_name,
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                    )
                    mlflow.register_model(
                        f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
                        MLFLOW_MODEL_NAME,
                    )
                except Exception as exc:
                    logger.warning(f"MLflow artifact registration skipped: {exc}")

            logger.info(
                f"{model_name} | AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f} | "
                f"P={np.mean(precisions):.3f} | R={np.mean(recalls):.3f} | "
                f"RMSE={np.mean(rmses):.3f} | MAE={np.mean(maes):.3f} | "
                f"Acc={np.mean(accuracies):.3f}"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
