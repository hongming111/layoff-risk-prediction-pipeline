"""Market & price ingestion — daily OHLCV + rolling volatility via yfinance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed/market.parquet")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=120))
def fetch_ohlcv(
    tickers: list[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    timeout: int = 60,
) -> pd.DataFrame:
    """Download adjusted OHLCV for the given tickers and date range."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        timeout=timeout,
    )
    return raw


def compute_rolling_volatility(close: pd.Series, windows: tuple[int, ...] = (7, 14, 21)) -> pd.DataFrame:
    """Annualised realised volatility from log returns over multiple windows."""
    log_ret = np.log(close / close.shift(1))
    vol = pd.DataFrame(index=close.index)
    for w in windows:
        vol[f"vol_{w}d"] = log_ret.rolling(w, min_periods=max(w // 2, 2)).std() * np.sqrt(252)
    return vol


def persist_to_parquet(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reset DatetimeIndex (named 'Date' by yfinance) to a regular lowercase column
    # so DuckDB and pandas both see it as a joinable column, not metadata.
    if df.index.name and df.index.name.lower() == "date":
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    if len(df) < before:
        logger.info(f"Market dedup: dropped {before - len(df)} duplicate (ticker, date) rows")
    df.to_parquet(path, index=False, compression="snappy")
    logger.info(f"Market data -> {path} ({df.shape})")
