"""Portfolio demo of the Corporate Layoff Risk Monitor.

Standalone Streamlit app for Hugging Face Spaces — reads a bundled static
snapshot instead of the project's live Postgres + MLflow stack, since Spaces'
free tier has no persistent database. See dashboard/app.py in the source repo
for the production version this is adapted from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

GITHUB_URL = "https://github.com/hongming111/layoff-risk-prediction-pipeline"
_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Layoff Risk Monitor — Demo",
    page_icon="⚠️",
    layout="wide",
)


@st.cache_data
def load_sample_predictions() -> pd.DataFrame:
    df = pd.read_parquet(_DIR / "sample_predictions.parquet")
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df.sort_values("score", ascending=False)


@st.cache_data
def load_model_metrics() -> dict:
    with open(_DIR / "model_metrics.json") as f:
        return json.load(f)


def main() -> None:
    st.title("Corporate Layoff Risk Monitor")
    st.caption("Portfolio demo · 90-day forward distress probability")

    preds = load_sample_predictions()
    metrics = load_model_metrics()
    snapshot_date = preds["prediction_date"].max().date()

    st.info(
        f"**This is a static portfolio demo, not a live system.** Scores below are "
        f"a real snapshot from the pipeline's XGBoost model as of {snapshot_date}, "
        f"for a small set of well-known public companies chosen to illustrate the "
        f"output — they are not investment, HR, or legal advice, and are not "
        f"updated in real time. Full source, live architecture, and MLOps "
        f"tooling: [{GITHUB_URL}]({GITHUB_URL})."
    )

    st.divider()

    k1, k2, k3 = st.columns(3)
    k1.metric("Companies in Sample", preds["ticker"].nunique())
    k2.metric("Snapshot Date", str(snapshot_date))
    k3.metric("Avg Score", f"{preds['score'].mean():.4f}")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("90-Day Distress Score by Company")
        fig = px.bar(
            preds, x="score", y="ticker", orientation="h",
            color="score", color_continuous_scale="Reds",
        )
        fig.update_layout(yaxis_autorange="reversed", showlegend=False,
                           xaxis_title="Distress Score", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.subheader("Model Validation Metrics (5-fold Stratified CV)")
        st.caption(
            "From the training run logged to MLflow — real cross-validation "
            "performance on historical WARN-labeled data, not a live "
            "production monitoring feed."
        )
        m1, m2 = st.columns(2)
        m1.metric("ROC-AUC", f"{metrics['roc_auc_mean']:.3f}")
        m1.metric("Precision", f"{metrics['precision_mean']:.3f}")
        m2.metric("Recall", f"{metrics['recall_mean']:.3f}")
        m2.metric("Avg Precision", f"{metrics['avg_precision_mean']:.3f}")
        st.caption(
            f"{metrics['n_features']} features · {metrics['n_cv_splits']}-fold CV · "
            f"{metrics['n_positive_labels']} positive labels · "
            f"{metrics['lookahead_days']}-day lookahead window"
        )

    st.divider()
    st.subheader("Sample Predictions")
    st.dataframe(
        preds[["ticker", "prediction_date", "target_date", "score"]]
        .rename(columns={"prediction_date": "Predicted On", "target_date": "90-Day Target",
                         "score": "Distress Score"}),
        width="stretch", hide_index=True,
    )

    st.divider()
    st.markdown(
        "**Pipeline overview:** WARN Act notices (ground truth) fused with "
        "market data (yfinance), fundamentals (Alpha Vantage/FMP), news/social "
        "sentiment, and BLS macro indicators, aligned into a daily feature "
        "matrix and served through an XGBoost classifier with MLflow tracking "
        "and Airflow-orchestrated drift monitoring. "
        f"Full architecture and code: [{GITHUB_URL}]({GITHUB_URL})"
    )


if __name__ == "__main__":
    main()
