"""Streamlit layoff risk monitoring dashboard.

Features:
- Active incident flags for companies with distress score > 0.85
- Score distribution histogram
- Top-N at-risk companies bar chart
- Per-ticker score trend over time
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from streamlit_autorefresh import st_autorefresh

DISTRESS_THRESHOLD = float(os.getenv("DISTRESS_ALERT_THRESHOLD", "0.85"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://retrench:retrench_pass@localhost:5432/retrenchment_db")
# Interval must match or exceed the cache TTL (300 s) so a rerun actually sees
# fresh data rather than hitting the still-warm cache.
_AUTOREFRESH_INTERVAL_MS = int(os.getenv("DASHBOARD_REFRESH_MS", "300000"))  # 5 min default
_DEFAULT_MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(Path(__file__).resolve().parent.parent / "mlflow.db").replace("\\", "/"),
)

st.set_page_config(
    page_title="Layoff Risk Monitor",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    # pd.read_sql() has a SA-version detection path that silently falls
    # through to a raw-DBAPI branch under SQLAlchemy 1.4, stripping the
    # text() wrapper before it reaches the driver and raising
    # "Query must be a string unless using sqlalchemy".
    # Bypassing pd.read_sql() entirely and using SA 1.4's native
    # conn.execute(text(...)) avoids that compatibility layer completely.
    with _engine().connect() as conn:
        result = conn.execute(
            text("SELECT * FROM predictions_log ORDER BY prediction_date DESC LIMIT 2000")
        )
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


@st.cache_data(ttl=300)
def load_system_metrics(include_seeded: bool = False) -> pd.DataFrame:
    """Load historical precision/recall from MLflow (written by evaluate_drift DAG).

    Runs tagged seeded=true come from scripts/seed_precision_trend.py — synthetic
    points that stand in until 90 days of real prediction-vs-actual history exists.
    Excluded by default so the dashboard never shows fabricated metrics as real;
    pass include_seeded=True (demo mode) to show them, clearly labeled by the caller.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(_DEFAULT_MLFLOW_URI)
        client = mlflow.MlflowClient()
        runs = client.search_runs(
            experiment_ids=["1"],
            filter_string="tags.mlflow.runName = 'prediction_vs_actual'",
            order_by=["start_time DESC"],
            max_results=90,
        )
        rows = [
            {
                "date": pd.to_datetime(r.info.start_time, unit="ms"),
                "precision": r.data.metrics.get("precision"),
                "recall": r.data.metrics.get("recall"),
                "roc_auc": r.data.metrics.get("roc_auc"),
                "seeded": r.data.tags.get("seeded") == "true",
            }
            for r in runs
        ]
        df = pd.DataFrame(rows)
        if not df.empty and not include_seeded:
            df = df[~df["seeded"]]
        return df
    except Exception:
        return pd.DataFrame()


# Root of the project — dashboard/app.py lives one level below the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent

_DATA_SOURCES: dict[str, Path] = {
    "Stock prices":          _REPO_ROOT / "data/processed/market.parquet",
    "Company financials":    _REPO_ROOT / "data/processed/fundamentals.parquet",
    "News sentiment":        _REPO_ROOT / "data/processed/sentiment.parquet",
    "Labour market data":    _REPO_ROOT / "data/processed/macro.parquet",
    "Layoff notices":        _REPO_ROOT / "data/processed/warn_raw.parquet",
    "Prediction model feed": _REPO_ROOT / "data/features/feature_matrix.parquet",
}


@st.cache_data(ttl=300)
def load_data_freshness() -> dict:
    """Return last-modified timestamps for every upstream parquet file.

    Also records the wall-clock time this function was called so the dashboard
    can display 'last loaded at HH:MM UTC' that updates every cache TTL cycle.
    """
    result: dict[str, str] = {}
    for label, path in _DATA_SOURCES.items():
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            result[label] = mtime.strftime("%Y-%m-%d %H:%M UTC")
        else:
            result[label] = "not yet generated"
    result["_loaded_at"] = datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")
    return result


def _render_freshness_sidebar(freshness: dict) -> None:
    # Sources updated by the hourly sentiment DAG vs. the daily pipeline.
    _hourly_sources = {"News sentiment"}

    with st.sidebar:
        st.divider()
        st.subheader("Data Freshness")
        for label, ts in freshness.items():
            if label.startswith("_"):
                continue
            icon = "✅" if ts != "not yet generated" else "⚠️"
            cadence = " *(hourly)*" if label in _hourly_sources else " *(daily)*"
            st.markdown(f"{icon} **{label}**{cadence}  \n`{ts}`")
        st.divider()
        refresh_mins = _AUTOREFRESH_INTERVAL_MS // 60_000
        st.caption(f"Dashboard cache loaded at {freshness.get('_loaded_at', '—')}")
        st.caption(f"Auto-refreshes every {refresh_mins} min")


def _incident_banner(high_risk: pd.DataFrame) -> None:
    st.error(
        f"🚨  **ACTIVE INCIDENT FLAGS** — {len(high_risk)} "
        f"{'company' if len(high_risk) == 1 else 'companies'} above "
        f"{DISTRESS_THRESHOLD:.0%} distress threshold"
    )
    st.dataframe(
        high_risk[["ticker", "prediction_date", "target_date", "score", "model_version"]]
        .rename(columns={"prediction_date": "Predicted On", "target_date": "90-Day Target",
                         "score": "Distress Score", "model_version": "Model"}),
        use_container_width=True,
    )


def main() -> None:
    # Auto-refresh: triggers a Streamlit rerun via a JS component every N ms.
    # Returns the current run count (unused here but required by the API).
    st_autorefresh(interval=_AUTOREFRESH_INTERVAL_MS, key="dashboard_autorefresh")

    st.title("Corporate Layoff Risk Monitor")
    refresh_mins = _AUTOREFRESH_INTERVAL_MS // 60_000
    st.caption(f"90-day forward distress probability · auto-refreshes every {refresh_mins} minutes")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")
        top_n = st.slider("Top N companies", min_value=5, max_value=50, value=20, step=5)
        min_score = st.slider("Min score filter", 0.0, 1.0, 0.0, 0.05)
        st.divider()
        st.metric("Alert threshold", f"{DISTRESS_THRESHOLD:.0%}")
        st.divider()
        show_demo_trend = st.checkbox(
            "Show simulated precision trend (demo)",
            value=False,
            help="Overlays synthetic precision/recall points from "
                 "scripts/seed_precision_trend.py, used only until 90 days of "
                 "real prediction-vs-actual history has accumulated.",
        )

    freshness = load_data_freshness()
    _render_freshness_sidebar(freshness)

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        preds = load_predictions()
    except Exception as exc:
        st.error(f"Database connection failed: {exc}")
        st.stop()

    if preds.empty:
        st.info("No predictions in the database yet. Trigger the ingestion DAG first.")
        st.stop()

    preds = preds[preds["score"] >= min_score]

    # ── Incident flags ────────────────────────────────────────────────────────
    high_risk = preds[preds["score"] >= DISTRESS_THRESHOLD].sort_values("score", ascending=False)
    if not high_risk.empty:
        _incident_banner(high_risk)
    else:
        st.success(f"No companies currently exceed the {DISTRESS_THRESHOLD:.0%} distress threshold.")

    st.divider()

    # ── KPI tiles ─────────────────────────────────────────────────────────────
    latest_date = preds["prediction_date"].max()
    latest = preds[preds["prediction_date"] == latest_date]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Companies Tracked", preds["ticker"].nunique())
    k2.metric("Latest Prediction Date", str(latest_date.date()))
    k3.metric("High-Risk Today", int((latest["score"] >= DISTRESS_THRESHOLD).sum()))
    k4.metric("Avg Score (Latest)", f"{latest['score'].mean():.3f}")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Score Distribution")
        fig = px.histogram(preds, x="score", nbins=50, color_discrete_sequence=["#e63946"])
        fig.add_vline(x=DISTRESS_THRESHOLD, line_dash="dash", line_color="orange",
                      annotation_text=f"Alert ({DISTRESS_THRESHOLD:.0%})")
        fig.update_layout(xaxis_title="Distress Score", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader(f"Top {top_n} At-Risk Companies (Latest Run)")
        top = latest.nlargest(top_n, "score")
        fig = px.bar(
            top, x="score", y="ticker", orientation="h",
            color="score", color_continuous_scale="Reds",
        )
        fig.add_vline(x=DISTRESS_THRESHOLD, line_dash="dash", line_color="orange")
        fig.update_layout(yaxis_autorange="reversed", xaxis_range=[0, 1], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── System health: precision over time ────────────────────────────────────
    metrics_df = load_system_metrics(include_seeded=show_demo_trend)
    if not metrics_df.empty:
        st.subheader("System Precision Over Time (Prediction vs Actual Loop)")
        if show_demo_trend and metrics_df["seeded"].any():
            st.warning(
                "⚠️ Includes simulated demo data (seeded=true) — not real "
                "production evaluation history. Uncheck 'Show simulated "
                "precision trend' in the sidebar to see only real metrics."
            )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics_df["date"], y=metrics_df["precision"],
                                 mode="lines+markers", name="Precision"))
        fig.add_trace(go.Scatter(x=metrics_df["date"], y=metrics_df["recall"],
                                 mode="lines+markers", name="Recall"))
        fig.add_hline(y=0.70, line_dash="dash", line_color="red",
                      annotation_text="Deploy threshold (70%)")
        fig.update_layout(yaxis_range=[0, 1], xaxis_title="Date", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-ticker trend ──────────────────────────────────────────────────────
    st.subheader("Score Trend — Ticker Lookup")
    ticker_input = st.text_input("Enter ticker symbol (e.g. AMZN)", "").strip().upper()
    if ticker_input:
        ticker_df = preds[preds["ticker"] == ticker_input].sort_values("prediction_date")
        if ticker_df.empty:
            st.warning(f"No predictions found for '{ticker_input}'.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ticker_df["prediction_date"], y=ticker_df["score"],
                mode="lines+markers", name="Distress Score",
                line={"color": "#e63946"},
            ))
            fig.add_hline(y=DISTRESS_THRESHOLD, line_dash="dash", line_color="orange",
                          annotation_text=f"Alert threshold ({DISTRESS_THRESHOLD:.0%})")
            fig.update_layout(
                title=f"{ticker_input} — 90-Day Layoff Probability",
                yaxis_range=[0, 1],
                xaxis_title="Prediction Date",
                yaxis_title="Score",
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
