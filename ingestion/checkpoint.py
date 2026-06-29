"""Lightweight checkpoint/resume for stateful ingestion loops.

A checkpoint records which items (states, tickers, …) have already been
successfully processed during a given calendar-day run.  On retry the loop
skips those items, preventing redundant API calls and preserving daily budget.

The checkpoint is keyed by *run_date*: if today's date does not match the
stored date, the file is ignored and the run starts fresh.  This ensures that
a partial run from yesterday doesn't permanently suppress items.

Usage
-----
    from ingestion.checkpoint import load_checkpoint, save_checkpoint

    done = load_checkpoint("fundamentals")
    for ticker in tickers:
        if ticker in done:
            logger.info(f"[checkpoint] skipping {ticker} (already processed)")
            continue
        # … do the work …
        done.add(ticker)
        save_checkpoint("fundamentals", done)
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path("data/processed")


def _path(name: str) -> Path:
    return _CHECKPOINT_DIR / f".checkpoint_{name}.json"


def load_checkpoint(name: str) -> set[str]:
    """Return the set of items already processed today.

    Returns an empty set when no checkpoint exists, the file is unreadable,
    or the stored run_date does not match today — so the caller always receives
    a clean slate at the start of a new calendar day.
    """
    p = _path(name)
    if not p.exists():
        return set()
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
        if state.get("run_date") != str(date.today()):
            logger.info(f"[checkpoint:{name}] stale (run_date={state.get('run_date')}) — starting fresh")
            return set()
        completed = set(state.get("completed", []))
        if completed:
            logger.info(f"[checkpoint:{name}] resuming — {len(completed)} items already done: {sorted(completed)}")
        return completed
    except Exception as exc:
        logger.warning(f"[checkpoint:{name}] could not read checkpoint, starting fresh: {exc}")
        return set()


def save_checkpoint(name: str, completed: set[str]) -> None:
    """Persist the set of completed items to disk.

    Called after each item succeeds so a mid-loop crash loses at most one
    item of work rather than the entire run.
    """
    p = _path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"run_date": str(date.today()), "completed": sorted(completed)}, indent=2),
        encoding="utf-8",
    )


def clear_checkpoint(name: str) -> None:
    """Remove a checkpoint file (e.g. after a successful full run)."""
    p = _path(name)
    if p.exists():
        p.unlink()
        logger.info(f"[checkpoint:{name}] cleared")


def read_watermark(
    parquet_path: Union[str, Path],
    date_col: str = "date",
    fallback_days: int = 730,
) -> date:
    """Return the high-water mark date from an existing parquet file.

    Reads the maximum value in `date_col` and returns it as a date so the
    caller can request only the delta (new data since the last successful run)
    instead of re-fetching the full history every day.

    Falls back to `today - fallback_days` when the parquet does not exist,
    is empty, or the date column cannot be parsed — this guarantees an initial
    full backfill on first run.

    Usage
    -----
        from ingestion.checkpoint import read_watermark

        start = read_watermark("data/processed/market.parquet", fallback_days=730)
        # fetch only from `start` onward
    """
    p = Path(parquet_path)
    fallback = date.today() - timedelta(days=fallback_days)
    if not p.exists():
        logger.info(f"[watermark] {p.name} not found — using fallback start {fallback}")
        return fallback
    try:
        import pandas as pd
        df = pd.read_parquet(p, columns=[date_col])
        if df.empty or date_col not in df.columns:
            logger.info(f"[watermark] {p.name} is empty — using fallback start {fallback}")
            return fallback
        max_ts = pd.to_datetime(df[date_col]).max()
        watermark = max_ts.date()
        logger.info(f"[watermark] {p.name} high-water mark: {watermark}")
        return watermark
    except Exception as exc:
        logger.warning(f"[watermark] could not read {p.name} — using fallback start {fallback}: {exc}")
        return fallback
