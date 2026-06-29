"""Fundamental metrics — quarterly financials via Alpha Vantage (FMP as fallback).

Rate limits
-----------
Alpha Vantage free tier: 25 req/day, 5 req/min.
Each ticker consumes 2 calls (balance_sheet + income_statement).
10 tickers → 20 calls/day — within budget with 5 spare.

Staleness guard
---------------
Fundamental data changes quarterly. The parquet cache is skipped only when it
is older than STALE_AFTER_DAYS (default 7). This prevents burning the 25-call
daily allowance on days when nothing has changed.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint
from ingestion.rate_limits import remaining, throttle_and_check

logger = logging.getLogger(__name__)

AV_BASE = "https://www.alphavantage.co/query"
FMP_BASE = "https://financialmodelingprep.com/api/v3"
PROCESSED_PATH = Path("data/processed/fundamentals.parquet")
STALE_AFTER_DAYS = 7       # re-fetch at most once per week
CALLS_PER_TICKER = 2       # balance_sheet + income_statement


# ── HTTP helpers (pure; throttle_and_check is called by the caller) ───────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=120))
def _av_balance_sheet(ticker: str, api_key: str) -> dict:
    resp = requests.get(
        AV_BASE,
        params={"function": "BALANCE_SHEET", "symbol": ticker, "apikey": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=120))
def _av_income_statement(ticker: str, api_key: str) -> dict:
    resp = requests.get(
        AV_BASE,
        params={"function": "INCOME_STATEMENT", "symbol": ticker, "apikey": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=60))
def _fmp_ratios(ticker: str, api_key: str) -> list[dict]:
    resp = requests.get(
        f"{FMP_BASE}/ratios/{ticker}",
        params={"limit": 20, "apikey": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── Staleness check ───────────────────────────────────────────────────────────

def _cache_age_days(path: Path) -> float | None:
    """Return age in days of the parquet cache, or None if it doesn't exist."""
    if not path.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age.total_seconds() / 86_400


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_fundamentals(
    tickers: list[str],
    api_key: str | None = None,
    provider: str = "alpha_vantage",
    output_path: Path = PROCESSED_PATH,
    stale_after_days: int = STALE_AFTER_DAYS,
) -> pd.DataFrame:
    """Fetch quarterly balance sheet ratios, with budget and staleness guards.

    Staleness: if the output parquet exists and is younger than
    `stale_after_days`, the cached file is returned immediately without
    consuming any API quota — fundamentals only change quarterly.

    Budget: tickers are capped so the total call count never exceeds the
    remaining daily Alpha Vantage allowance. A warning is logged when
    the ticker list is trimmed.

    Returns DataFrame: ticker, date, debt_to_equity, current_ratio, profit_margin
    """
    # ── Staleness guard ───────────────────────────────────────────────────────
    age = _cache_age_days(output_path)
    if age is not None and age < stale_after_days:
        logger.info(
            f"Fundamentals cache is {age:.1f}d old (threshold {stale_after_days}d) "
            "— returning cached data without consuming API quota."
        )
        return pd.read_parquet(output_path)

    # ── Budget guard: trim ticker list if allowance is insufficient ───────────
    budget = remaining("alpha_vantage")
    max_tickers = budget // CALLS_PER_TICKER
    if max_tickers == 0:
        raise RuntimeError(
            f"[fundamentals] Alpha Vantage daily budget exhausted "
            f"({budget} calls left, need {CALLS_PER_TICKER}/ticker). Skipping."
        )
    if max_tickers < len(tickers):
        logger.warning(
            f"[fundamentals] Budget allows {max_tickers} tickers "
            f"(of {len(tickers)} requested, {CALLS_PER_TICKER} calls each). "
            "Remaining tickers will be covered on subsequent days."
        )
        tickers = tickers[:max_tickers]

    key = api_key or os.getenv("ALPHA_VANTAGE_KEY", "")
    fmp_key = os.getenv("FMP_KEY", "")
    records: list[pd.DataFrame] = []

    done = load_checkpoint("fundamentals")
    for ticker in tickers:
        if ticker in done:
            logger.info(f"Fundamentals [{ticker}]: skipping (checkpoint)")
            continue
        try:
            if provider == "alpha_vantage" and key:
                # ── Call 1: balance sheet ─────────────────────────────────────
                throttle_and_check("alpha_vantage")
                bs = _av_balance_sheet(ticker, key).get("quarterlyReports", [])

                # ── Call 2: income statement ──────────────────────────────────
                throttle_and_check("alpha_vantage")
                is_ = _av_income_statement(ticker, key).get("quarterlyReports", [])

                bs_df = pd.DataFrame(bs)[[
                    "fiscalDateEnding", "totalCurrentAssets",
                    "totalCurrentLiabilities", "longTermDebt", "totalShareholderEquity",
                ]]
                is_df = pd.DataFrame(is_)[["fiscalDateEnding", "netIncome", "totalRevenue"]]

                merged = bs_df.merge(is_df, on="fiscalDateEnding", how="left")
                fiscal_dates = pd.to_datetime(merged["fiscalDateEnding"], errors="coerce")
                merged = merged.drop(columns=["fiscalDateEnding"]).apply(pd.to_numeric, errors="coerce")
                merged["date"] = fiscal_dates   # re-attach after numeric conversion to keep datetime dtype
                merged["ticker"] = ticker
                merged["debt_to_equity"] = (
                    merged["longTermDebt"]
                    / merged["totalShareholderEquity"].replace(0, float("nan"))
                )
                merged["current_ratio"] = (
                    merged["totalCurrentAssets"]
                    / merged["totalCurrentLiabilities"].replace(0, float("nan"))
                )
                merged["profit_margin"] = (
                    merged["netIncome"]
                    / merged["totalRevenue"].replace(0, float("nan"))
                )
                records.append(merged[["ticker", "date", "debt_to_equity", "current_ratio", "profit_margin"]])
                done.add(ticker)
                save_checkpoint("fundamentals", done)
                logger.info(f"Fundamentals AV [{ticker}]: {len(merged)} quarters")

            elif fmp_key:
                # ── FMP fallback: 1 call covers all ratios ────────────────────
                throttle_and_check("fmp")
                ratios = _fmp_ratios(ticker, fmp_key)
                if ratios:
                    df = pd.DataFrame(ratios)[["date", "debtEquityRatio", "currentRatio", "netProfitMargin"]]
                    df = df.rename(columns={
                        "debtEquityRatio": "debt_to_equity",
                        "currentRatio": "current_ratio",
                        "netProfitMargin": "profit_margin",
                    })
                    df["ticker"] = ticker
                    df["date"] = pd.to_datetime(df["date"])
                    records.append(df[["ticker", "date", "debt_to_equity", "current_ratio", "profit_margin"]])
                    done.add(ticker)
                    save_checkpoint("fundamentals", done)
                    logger.info(f"Fundamentals FMP [{ticker}]: {len(df)} quarters")

            else:
                logger.warning(f"Fundamentals [{ticker}]: no API key available — skipping")

        except RuntimeError:
            raise  # propagate budget-exhaustion to stop the loop
        except Exception as exc:
            logger.warning(f"Fundamentals [{ticker}]: failed — {exc}")

    clear_checkpoint("fundamentals")
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def persist_to_parquet(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    if len(df) < before:
        logger.info(f"Fundamentals dedup: dropped {before - len(df)} duplicate (ticker, date) rows")
    df.to_parquet(path, index=False, compression="snappy")
    logger.info(f"Fundamentals → {path} ({df.shape})")
