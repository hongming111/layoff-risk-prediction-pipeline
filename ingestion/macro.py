"""Macro headwind ingestion — sector-level labor metrics via BLS Public Data API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BLS_API_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
PROCESSED_PATH = Path("data/processed/macro.parquet")

# BLS series IDs: seasonally adjusted monthly unemployment + layoff rates by sector
# Source: CPS (unemployment) + Job Openings and Labor Turnover Survey (JOLTS)
#
# JOLTS series ID format (21 chars):
#   JT + S/U(1) + industry(6) + state(2) + area(5) + size(2) + element(2) + rate(1)
#   National = state 00, area 00000, all sizes 00, seasonally adjusted = S
BLS_SERIES = {
    "unemployment_rate_total": "LNS14000000",           # CPS: total civilian unemployment rate
    "layoff_rate_total":       "JTS000000000000000LDR", # JOLTS: total nonfarm layoffs rate (SA)
    "layoff_rate_tech":        "JTS510000000000000LDR", # JOLTS: information sector (NAICS 51)
    "layoff_rate_finance":     "JTS520000000000000LDR", # JOLTS: finance & insurance (NAICS 52)
    "layoff_rate_retail":      "JTS440000000000000LDR", # JOLTS: retail trade (NAICS 44)
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=60))
def _fetch_bls_series(series_ids: list[str], start_year: str, end_year: str, api_key: str) -> dict:
    payload = {
        "seriesid": series_ids,
        "startyear": start_year,
        "endyear": end_year,
        "registrationkey": api_key,
    }
    resp = requests.post(BLS_API_V2, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_bls_series(
    start_year: str = "2019",
    end_year: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch monthly BLS labor metrics and return as a tidy DataFrame.

    Columns: date (monthly, last day), unemployment_rate_total, layoff_rate_total, ...
    """
    import datetime

    key = api_key or os.getenv("BLS_API_KEY", "")
    end = end_year or str(datetime.date.today().year)

    data = _fetch_bls_series(list(BLS_SERIES.values()), start_year, end, key)

    if data.get("status") != "REQUEST_SUCCEEDED":
        logger.error(f"BLS API error: {data.get('message', data.get('status'))}")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    series_id_to_name = {v: k for k, v in BLS_SERIES.items()}

    for series in data.get("Results", {}).get("series", []):
        series_name = series_id_to_name.get(series["seriesID"], series["seriesID"])
        rows = []
        for d in series.get("data", []):
            if not d.get("period", "").startswith("M"):
                continue
            try:
                val = float(d["value"])
            except (ValueError, TypeError):
                continue  # BLS uses "-" for suppressed/missing values
            rows.append({
                "date": pd.Period(f"{d['year']}-{d['period'].replace('M', '')}", freq="M").to_timestamp("M"),
                series_name: val,
            })
        frames.append(pd.DataFrame(rows))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    from functools import reduce

    combined = reduce(lambda a, b: a.merge(b, on="date", how="outer"), frames)
    combined = combined.sort_values("date").reset_index(drop=True)
    logger.info(f"BLS data: {combined.shape[0]} months, {combined.shape[1] - 1} series")
    return combined


def persist_to_parquet(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    before = len(df)
    df = df.drop_duplicates(subset=["date"], keep="last")
    if len(df) < before:
        logger.info(f"Macro dedup: dropped {before - len(df)} duplicate date rows")
    df.to_parquet(path, index=False, compression="snappy")
    logger.info(f"Macro → {path} ({df.shape})")
