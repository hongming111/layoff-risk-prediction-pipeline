"""Single source of truth for the pipeline's feature schema contracts.

Centralising FEATURE_COLS and PARQUET_SCHEMAS here means any column rename
breaks loudly at import time in train.py / predict.py / feature_engineer.py
rather than silently producing wrong inference scores.
"""

from __future__ import annotations

FEATURE_COLS: list[str] = [
    "close", "vol_7d", "vol_14d", "vol_21d",
    "debt_to_equity", "current_ratio", "profit_margin",
    "sentiment_score", "sentiment_score_ma7d", "mention_velocity",
    "unemployment_rate_total", "layoff_rate_total", "layoff_rate_tech",
]

LABEL_COL: str = "label"
TICKER_COL: str = "ticker"
DATE_COL: str = "date"

PARQUET_SCHEMAS: dict[str, dict] = {
    "fundamentals": {
        "ticker": "str", "date": "datetime64[ns]",
        "debt_to_equity": "float64", "current_ratio": "float64", "profit_margin": "float64",
    },
    "sentiment": {
        "ticker": "str", "date": "datetime64[ns]",
        "sentiment_score": "float64", "sentiment_score_ma7d": "float64", "mention_velocity": "float64",
    },
    "macro": {
        "date": "datetime64[ns]",
        "unemployment_rate_total": "float64", "layoff_rate_total": "float64", "layoff_rate_tech": "float64",
    },
    "labels": {
        "ticker": "str", "event_date": "datetime64[ns]", "employees_affected": "float64",
    },
}
