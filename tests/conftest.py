"""Shared fixtures for the test suite."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ── Stub out packages that live only in Docker, not in the local venv ─────────
# These are top-level imports in ingestion modules. The functions under test
# (persist_to_parquet, dedup logic) never call into these libraries, so
# lightweight stubs are safe.

def _noop_retry(*args, **kwargs):
    """Stand-in for tenacity.retry — returns the decorated function unchanged."""
    def decorator(fn):
        return fn
    return decorator

for _name, _attrs in [
    ("yfinance",      {}),
    ("feedparser",    {}),
    ("newsapi",       {}),
    ("transformers",  {}),
    ("tenacity",      {
        "retry":             _noop_retry,
        "stop_after_attempt": MagicMock(return_value=None),
        "wait_exponential":  MagicMock(return_value=None),
    }),
    ("duckdb",        {}),
    ("mlflow",        {}),
    ("mlflow.sklearn",{}),
    ("xgboost",       {}),
    ("sqlalchemy",    {}),
]:
    if _name not in sys.modules:
        mock = MagicMock()
        mock.__spec__ = None   # prevents pytest from inspecting spec attributes
        for attr, val in _attrs.items():
            setattr(mock, attr, val)
        sys.modules[_name] = mock


@pytest.fixture()
def valid_feature_df() -> pd.DataFrame:
    """Minimal valid feature matrix with one positive label."""
    n = 20
    rng = np.random.default_rng(42)
    labels = [0] * (n - 1) + [1]
    return pd.DataFrame({
        "ticker":                  ["AAPL"] * 10 + ["MSFT"] * 10,
        "date":                    pd.date_range("2025-01-01", periods=n, freq="D"),
        "close":                   rng.uniform(100, 300, n),
        "vol_7d":                  rng.uniform(0.1, 0.5, n),
        "vol_14d":                 rng.uniform(0.1, 0.5, n),
        "vol_21d":                 rng.uniform(0.1, 0.5, n),
        "debt_to_equity":          rng.uniform(0.1, 2.0, n),
        "current_ratio":           rng.uniform(1.0, 3.0, n),
        "profit_margin":           rng.uniform(0.0, 0.3, n),
        "sentiment_score":         rng.uniform(-1, 1, n),
        "sentiment_score_ma7d":    rng.uniform(-1, 1, n),
        "mention_velocity":        rng.integers(0, 50, n).astype(float),
        "unemployment_rate_total": rng.uniform(3.0, 7.0, n),
        "layoff_rate_total":       rng.uniform(0.5, 2.0, n),
        "layoff_rate_tech":        rng.uniform(0.5, 2.0, n),
        "label":                   labels,
    })


@pytest.fixture()
def valid_predictions_df() -> pd.DataFrame:
    """Minimal valid predictions DataFrame."""
    import datetime
    return pd.DataFrame({
        "ticker":          ["AAPL", "MSFT", "GOOG"],
        "prediction_date": datetime.date.today(),
        "target_date":     datetime.date.today(),
        "score":           [0.12, 0.87, 0.45],
        "model_version":   "xgboost_local",
    })
