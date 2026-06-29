"""Per-provider daily budget tracking and per-minute rate gating.

Budget state is persisted to data/processed/.budget_state.json so it
survives across Airflow task subprocess boundaries within the same day.

Usage:
    from ingestion.rate_limits import throttle_and_check, remaining

    # Before every API call:
    throttle_and_check("alpha_vantage")     # sleeps if needed, raises if daily cap hit
    resp = _av_balance_sheet(ticker, key)

Design notes:
- Conservative caps are set ~10 % below the documented hard limits so
  retries and occasional re-runs stay safely within the free tier.
- Alpha Vantage enforces 5 req/min (hard). MIN_INTERVAL is set to 13 s
  (one second above the 12 s minimum) to absorb clock drift and latency.
- Retry decorators on the HTTP helpers consume one budget slot per logical
  call attempt, not per retry, because throttle_and_check is called by the
  caller before invoking the helper.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

BUDGET_FILE = Path("data/processed/.budget_state.json")

# ── Conservative daily caps (actual limit × 0.88) ────────────────────────────
DAILY_CAPS: dict[str, int] = {
    "alpha_vantage": 22,   # hard limit: 25   — 2 calls/ticker × 10 tickers = 20, leaves 2 spare
    "fmp":          220,   # hard limit: 250
    "news_api":      88,   # hard limit: 100  — 1-3 pages × 10 tickers = 10-30, very comfortable
}

# ── Minimum seconds between consecutive calls to the same provider ────────────
MIN_INTERVAL_SECONDS: dict[str, float] = {
    "alpha_vantage": 13.0,  # 5 req/min  → 12 s min; +1 s buffer
    "fmp":            0.5,  # no documented per-minute cap; be polite
    "news_api":       1.0,  # no hard per-minute cap; avoid hammering
}

# ── In-process lock (covers threading within a single Airflow worker) ─────────
_lock = threading.Lock()

# Per-provider last-call timestamp (monotonic; in-process only)
_last_call_ts: dict[str, float] = {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load() -> dict:
    """Load today's state from disk; return a fresh dict if stale or missing."""
    if BUDGET_FILE.exists():
        try:
            state = json.loads(BUDGET_FILE.read_text())
            if state.get("date") == str(date.today()):
                return state
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return {"date": str(date.today()), "counts": {}}


def _save(state: dict) -> None:
    """Atomically write state (rename prevents partial reads)."""
    BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = BUDGET_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(BUDGET_FILE)


# ── Public API ────────────────────────────────────────────────────────────────

def throttle_and_check(provider: str, n: int = 1) -> None:
    """Gate a call to `provider` for `n` request slots.

    1. Checks the daily budget — raises RuntimeError if exhausted.
    2. Sleeps the remainder of MIN_INTERVAL if the last call was too recent.
    3. Debits `n` slots from today's count and persists to disk.

    Call this immediately before each API request.
    """
    cap = DAILY_CAPS.get(provider, 9_999)
    gap = MIN_INTERVAL_SECONDS.get(provider, 0.0)

    with _lock:
        state = _load()
        used = state["counts"].get(provider, 0)

        if used + n > cap:
            raise RuntimeError(
                f"[rate_limits] {provider}: daily budget exhausted "
                f"({used}/{cap} calls used). Skipping to stay within the free tier."
            )

        # Per-minute throttle — sleep only the remaining fraction of the gap
        elapsed = time.monotonic() - _last_call_ts.get(provider, 0.0)
        wait = gap - elapsed
        if wait > 0:
            logger.debug(f"[rate_limits] {provider}: sleeping {wait:.2f}s (per-minute gate)")
            time.sleep(wait)

        state["counts"][provider] = used + n
        _last_call_ts[provider] = time.monotonic()
        _save(state)

    logger.debug(f"[rate_limits] {provider}: {used + n}/{cap} daily calls used")


def remaining(provider: str) -> int:
    """Return how many calls are still available today for `provider`."""
    used = _load()["counts"].get(provider, 0)
    return max(0, DAILY_CAPS.get(provider, 9_999) - used)


def log_status() -> None:
    """Emit an INFO line summarising today's usage for all tracked providers."""
    state = _load()
    for provider, cap in DAILY_CAPS.items():
        used = state["counts"].get(provider, 0)
        logger.info(f"[rate_limits] {provider}: {used}/{cap} calls used today")
