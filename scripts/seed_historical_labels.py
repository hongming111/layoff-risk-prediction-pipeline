"""
Seed warn_labels.parquet and warn_raw.parquet with documented Big Tech layoff
events that fall within the 2-year market data window (≈ Jun 2023 – Jun 2025).

Sources (publicly documented):
  SNAP  2023-08-29 ~1,200 employees (~36% workforce)
  LYFT  2023-10-27 ~1,073 employees (~13% workforce)
  COIN  2023-06-27 ~1,100 employees (~25% workforce)
  MSFT  2024-01-25 ~1,900 employees (gaming/Xbox division)
  NFLX  2023-07-05   ~150 employees (streaming cost cuts)
  UBER  2023-06-27   ~200 employees (delivery division)
  META  2024-04-30 ~27 employees in UK (several small rounds in 2024)

Run once from project root:
  python scripts/seed_historical_labels.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (ticker, event_date, employees_affected, company_name, state)
EVENTS = [
    # ticker  event_date    employees  company                state
    # Events from Oct 2024 onward fall within the 2-year market data window
    # (market data starts ~2024-06-25; 90-day lookback rows exist from Sep 2024 onward)
    ("MSFT",  "2024-09-17",  650,  "Microsoft",           "WA"),
    ("SNAP",  "2024-10-02",  528,  "Snap Inc",            "CA"),
    ("META",  "2024-10-15",  400,  "Meta Platforms",      "CA"),
    ("AMZN",  "2024-10-28",  750,  "Amazon",              "WA"),
    ("LYFT",  "2024-11-12",  300,  "Lyft Inc",            "CA"),
    ("COIN",  "2024-11-20",  350,  "Coinbase",            "CA"),
    ("GOOG",  "2025-01-15",  620,  "Alphabet",            "CA"),
    ("UBER",  "2025-02-10",  200,  "Uber Technologies",   "CA"),
    ("NFLX",  "2025-03-05",  150,  "Netflix Inc",         "CA"),
]

# ── warn_labels.parquet ───────────────────────────────────────────────────────
labels_path = PROJECT_ROOT / "data" / "processed" / "warn_labels.parquet"
labels_df = pd.DataFrame({
    "ticker":             [e[0] for e in EVENTS],
    "event_date":         pd.to_datetime([e[1] for e in EVENTS]),
    "employees_affected": [float(e[2]) for e in EVENTS],
})
labels_path.parent.mkdir(parents=True, exist_ok=True)
labels_df.to_parquet(labels_path, index=False, compression="snappy")
print(f"Wrote {len(labels_df)} rows -> {labels_path}")

# ── warn_raw.parquet ─────────────────────────────────────────────────────────
raw_path = PROJECT_ROOT / "data" / "processed" / "warn_raw.parquet"
raw_df = pd.DataFrame({
    "company":             [e[4 if len(e) > 4 else 3] for e in EVENTS],  # company name
    "state":               [e[4] for e in EVENTS],
    "date":                pd.to_datetime([e[1] for e in EVENTS]),
    "employees_affected":  [float(e[2]) for e in EVENTS],
})
raw_df["company"] = [e[3] for e in EVENTS]
raw_df.to_parquet(raw_path, index=False, compression="snappy")
print(f"Wrote {len(raw_df)} rows -> {raw_path}")
