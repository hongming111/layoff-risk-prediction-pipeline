"""End-to-end pipeline health check with scoring.

Covers every stage from raw data files through to what the dashboard serves.
Run from the project root:

    python scripts/pipeline_health_check.py

Postgres checks are skipped gracefully when the database is not reachable.
No Airflow or Docker daemon connection is required --checks run against the
files and database that the live pipeline has already produced.

Scoring
-------
Each check is worth 0–5 points.  Totals are shown per stage and overall.
  PASS  = full marks   (green)
  WARN  = partial      (yellow)
  FAIL  = zero marks   (red)
  SKIP  = not counted  (grey)
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "close", "vol_7d", "vol_14d", "vol_21d",
    "debt_to_equity", "current_ratio", "profit_margin",
    "sentiment_score", "sentiment_score_ma7d", "mention_velocity",
    "unemployment_rate_total", "layoff_rate_total", "layoff_rate_tech",
]
EXPECTED_TICKERS = ["AMZN", "GOOG", "META", "MSFT", "AAPL",
                    "NFLX", "SNAP", "LYFT", "UBER", "COIN"]

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://retrench:retrench_pass@localhost:5432/retrenchment_db",
)

# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    label: str
    status: str        # PASS | WARN | FAIL | SKIP
    detail: str
    score: int
    max_score: int

    @property
    def pct(self) -> float:
        return (self.score / self.max_score * 100) if self.max_score else 0.0


_results: list[CheckResult] = []


def _record(label: str, status: str, detail: str, score: int, max_score: int = 5) -> CheckResult:
    r = CheckResult(label, status, detail, score, max_score)
    _results.append(r)
    return r


# ── Colour helpers ────────────────────────────────────────────────────────────

_USE_COLOUR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def _green(t):  return _c(t, "32")
def _yellow(t): return _c(t, "33")
def _red(t):    return _c(t, "31")
def _grey(t):   return _c(t, "90")
def _bold(t):   return _c(t, "1")

def _status_str(status: str) -> str:
    if status == "PASS": return _green("PASS")
    if status == "WARN": return _yellow("WARN")
    if status == "FAIL": return _red("FAIL")
    return _grey("SKIP")


# ── Stage helpers ─────────────────────────────────────────────────────────────

def _print_stage(title: str) -> None:
    print(f"\n{_bold(title)}")
    print("-" * 70)


def _print_result(r: CheckResult) -> None:
    score_str = f"{r.score}/{r.max_score}" if r.status != "SKIP" else " --"
    print(f"  [{_status_str(r.status)}]  {r.label:<30}  {r.detail:<30}  {score_str}")


# ── Stage 1: Data Layer ───────────────────────────────────────────────────────

def _check_parquet(
    label: str,
    path: Path,
    required_cols: list[str],
    min_rows: int = 1,
) -> CheckResult:
    if not path.exists():
        return _record(label, "FAIL", "file not found", 0)

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return _record(label, "FAIL", f"unreadable: {exc}", 0)

    missing_cols = [c for c in required_cols if c not in df.columns]
    rows = len(df)
    age_days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days

    if missing_cols:
        detail = f"{rows} rows, missing cols: {missing_cols}"
        return _record(label, "FAIL", detail, 1)

    if rows < min_rows:
        detail = f"only {rows} rows (need >= {min_rows})"
        return _record(label, "FAIL", detail, 2)

    # Date range info if there's a date column
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col:
        try:
            dmin = pd.to_datetime(df[date_col]).min().date()
            dmax = pd.to_datetime(df[date_col]).max().date()
            detail = f"{rows:,} rows  {dmin} -> {dmax}"
        except Exception:
            detail = f"{rows:,} rows"
    else:
        detail = f"{rows:,} rows"

    score = 5 if age_days <= 7 else 4  # slight deduction if data is stale
    status = "PASS" if score == 5 else "WARN"
    if score < 5:
        detail += f"  (stale: {age_days}d old)"
    return _record(label, status, detail, score)


def stage_data_layer() -> None:
    _print_stage("STAGE 1 - DATA LAYER")

    checks = [
        ("Market (OHLCV)",   ROOT / "data/processed/market.parquet",
         ["ticker", "date", "close", "vol_7d"],          365),
        ("Fundamentals",     ROOT / "data/processed/fundamentals.parquet",
         ["ticker", "date", "debt_to_equity"],             1),
        ("Sentiment",        ROOT / "data/processed/sentiment.parquet",
         ["ticker", "date", "sentiment_score"],            1),
        ("Macro (BLS)",      ROOT / "data/processed/macro.parquet",
         ["date", "unemployment_rate_total"],              12),
        ("WARN Labels",      ROOT / "data/processed/warn_labels.parquet",
         ["ticker", "event_date"],                         1),
    ]
    for label, path, cols, min_rows in checks:
        _print_result(_check_parquet(label, path, cols, min_rows))


# ── Stage 2: ETL / Feature Matrix ────────────────────────────────────────────

def stage_etl() -> None:
    _print_stage("STAGE 2 - ETL / FEATURE MATRIX")

    fm_path = ROOT / "data/features/feature_matrix.parquet"
    if not fm_path.exists():
        _print_result(_record("Feature matrix", "FAIL", "file not found", 0, 5))
        _print_result(_record("Positive labels", "SKIP", "matrix missing", 0, 5))
        _print_result(_record("Market null rate", "SKIP", "matrix missing", 0, 5))
        _print_result(_record("Ticker coverage", "SKIP", "matrix missing", 0, 5))
        return

    try:
        fm = pd.read_parquet(fm_path)
    except Exception as exc:
        _print_result(_record("Feature matrix", "FAIL", str(exc), 0, 5))
        return

    # Check 1: shape and columns
    missing = [c for c in FEATURE_COLS + ["ticker", "date", "label"] if c not in fm.columns]
    rows, cols = fm.shape
    if missing:
        _print_result(_record("Feature matrix", "WARN",
                               f"{rows:,} rows, missing: {missing}", 3, 5))
    else:
        _print_result(_record("Feature matrix", "PASS",
                               f"{rows:,} rows × {cols} cols, {fm['ticker'].nunique()} tickers", 5, 5))

    # Check 2: positive labels
    if "label" in fm.columns:
        pos = int(fm["label"].sum())
        pos_pct = pos / rows * 100 if rows else 0
        if pos == 0:
            _print_result(_record("Positive labels", "FAIL",
                                   "0 positive labels --run seed_historical_labels.py", 0, 5))
        elif pos < 5:
            _print_result(_record("Positive labels", "WARN",
                                   f"{pos} labels ({pos_pct:.2f}% of rows)", 3, 5))
        else:
            _print_result(_record("Positive labels", "PASS",
                                   f"{pos} labels ({pos_pct:.2f}% of rows)", 5, 5))
    else:
        _print_result(_record("Positive labels", "FAIL", "'label' column missing", 0, 5))

    # Check 3: null rate on critical market columns
    critical = [c for c in ["close", "vol_7d", "vol_14d"] if c in fm.columns]
    if critical:
        null_rates = {c: fm[c].isna().mean() * 100 for c in critical}
        max_null = max(null_rates.values())
        detail = "  ".join(f"{c}: {v:.1f}%" for c, v in null_rates.items())
        if max_null > 20:
            _print_result(_record("Market null rate", "FAIL", detail, 0, 5))
        elif max_null > 5:
            _print_result(_record("Market null rate", "WARN", detail, 3, 5))
        else:
            _print_result(_record("Market null rate", "PASS", detail, 5, 5))
    else:
        _print_result(_record("Market null rate", "SKIP", "close/vol cols missing", 0, 5))

    # Check 4: ticker coverage
    if "ticker" in fm.columns:
        found = set(fm["ticker"].dropna().unique())
        expected = set(EXPECTED_TICKERS)
        missing_tickers = expected - found
        pct = len(found & expected) / len(expected) * 100
        detail = f"{len(found & expected)}/{len(expected)} tickers"
        if missing_tickers:
            detail += f"  missing: {sorted(missing_tickers)}"
        status = "PASS" if not missing_tickers else ("WARN" if pct >= 70 else "FAIL")
        score = 5 if not missing_tickers else (3 if pct >= 70 else 1)
        _print_result(_record("Ticker coverage", status, detail, score, 5))
    else:
        _print_result(_record("Ticker coverage", "FAIL", "'ticker' column missing", 0, 5))

    # Check 5: feature registry
    registry_path = ROOT / "data/features/feature_registry.json"
    if not registry_path.exists():
        _print_result(_record("Feature registry", "FAIL",
                               "feature_registry.json not found", 0, 5))
    else:
        try:
            import json as _json
            with open(registry_path) as _f:
                reg = _json.load(_f)
            n_features = len(reg.get("features", {}))
            expected_n = len(FEATURE_COLS)
            if n_features >= expected_n:
                _print_result(_record("Feature registry", "PASS",
                                       f"{n_features} features documented", 5, 5))
            else:
                _print_result(_record("Feature registry", "WARN",
                                       f"{n_features}/{expected_n} features in registry", 3, 5))
        except Exception as exc:
            _print_result(_record("Feature registry", "FAIL", str(exc), 0, 5))

    # Check 6: versioned snapshots
    features_dir = ROOT / "data/features"
    snapshots = sorted(features_dir.glob("feature_matrix_????????_????????.parquet"))
    if not snapshots:
        _print_result(_record("Versioned snapshots", "WARN",
                               "no versioned snapshots yet — run ETL to create first", 2, 5))
    else:
        latest_snap = snapshots[-1]
        n = len(snapshots)
        _print_result(_record("Versioned snapshots", "PASS",
                               f"{n} snapshot(s), latest: {latest_snap.name}", 5, 5))


# ── Stage 3: ML Model ─────────────────────────────────────────────────────────

def stage_ml() -> None:
    _print_stage("STAGE 3 - ML MODEL")

    fm_path = ROOT / "data/features/feature_matrix.parquet"

    # Check 1: model artifact exists
    model_path = None
    for pkl in ["xgboost.pkl", "random_forest.pkl"]:
        p = ROOT / "data/models" / pkl
        if p.exists():
            size_kb = p.stat().st_size / 1024
            _print_result(_record(
                f"Model artifact ({pkl})", "PASS",
                f"{size_kb:.0f} KB", 5, 5,
            ))
            model_path = p
            break
    if model_path is None:
        _print_result(_record("Model artifact", "FAIL",
                               "no .pkl found in data/models/", 0, 5))

    # Check 2: model can load and score
    if model_path and fm_path.exists():
        try:
            with open(model_path, "rb") as fh:
                model = pickle.load(fh)
            fm = pd.read_parquet(fm_path)
            present_cols = [c for c in FEATURE_COLS if c in fm.columns]
            missing_cols = [c for c in FEATURE_COLS if c not in fm.columns]
            X = fm[present_cols].fillna(0).values
            scores = model.predict_proba(X)[:, 1]
            detail = f"{len(scores)} predictions generated"
            if missing_cols:
                detail += f"  ({len(missing_cols)} feature cols missing -> filled 0)"
            _print_result(_record("Model scoring", "PASS", detail, 5, 5))
        except Exception as exc:
            _print_result(_record("Model scoring", "FAIL", str(exc), 0, 5))
            scores = None
    else:
        _print_result(_record("Model scoring", "SKIP",
                               "model or feature matrix not found", 0, 5))
        scores = None

    # Check 3: all scores in [0, 1]
    if scores is not None:
        out_of_range = int(((scores < 0) | (scores > 1)).sum())
        if out_of_range:
            _print_result(_record("Score range [0,1]", "FAIL",
                                   f"{out_of_range} scores outside [0,1]", 0, 5))
        else:
            lo, hi = float(scores.min()), float(scores.max())
            _print_result(_record("Score range [0,1]", "PASS",
                                   f"min={lo:.3f}  max={hi:.3f}", 5, 5))
    else:
        _print_result(_record("Score range [0,1]", "SKIP", "no scores produced", 0, 5))

    # Check 4: score distribution is non-degenerate
    if scores is not None and len(scores) > 1:
        import numpy as np
        std = float(np.std(scores))
        mean = float(np.mean(scores))
        if std < 0.01:
            _print_result(_record("Score variance", "FAIL",
                                   f"std={std:.4f} --all scores near identical", 0, 5))
        elif std < 0.05:
            _print_result(_record("Score variance", "WARN",
                                   f"mean={mean:.3f}  std={std:.3f} (low spread)", 3, 5))
        else:
            _print_result(_record("Score variance", "PASS",
                                   f"mean={mean:.3f}  std={std:.3f}", 5, 5))
    else:
        _print_result(_record("Score variance", "SKIP", "no scores produced", 0, 5))


# ── Stage 4: Postgres ─────────────────────────────────────────────────────────

def stage_postgres() -> None:
    _print_stage("STAGE 4 - POSTGRES / PREDICTIONS LOG")

    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        msg = f"DB not reachable --{exc}"
        for label in ["Row count", "No duplicates", "Data freshness", "Score validity"]:
            _print_result(_record(label, "SKIP", msg, 0, 5))
        return

    with engine.connect() as conn:
        # Check 1: row count
        try:
            row = conn.execute(text(
                "SELECT COUNT(*) AS n, MAX(prediction_date) AS latest FROM predictions_log"
            )).fetchone()
            n, latest = row.n, row.latest
            if n == 0:
                _print_result(_record("Row count", "FAIL", "0 rows in predictions_log", 0, 5))
            else:
                _print_result(_record("Row count", "PASS",
                                       f"{n} rows, latest: {latest}", 5, 5))
        except Exception as exc:
            _print_result(_record("Row count", "FAIL", str(exc), 0, 5))

        # Check 2: no duplicates
        try:
            dup_row = conn.execute(text("""
                SELECT COUNT(*) AS dupes FROM (
                    SELECT ticker, prediction_date, COUNT(*) AS c
                    FROM predictions_log
                    GROUP BY ticker, prediction_date
                    HAVING COUNT(*) > 1
                ) sub
            """)).fetchone()
            dupes = dup_row.dupes
            if dupes:
                _print_result(_record("No duplicates", "FAIL",
                                       f"{dupes} duplicate (ticker, date) pairs", 0, 5))
            else:
                _print_result(_record("No duplicates", "PASS",
                                       "0 duplicate rows", 5, 5))
        except Exception as exc:
            _print_result(_record("No duplicates", "FAIL", str(exc), 0, 5))

        # Check 3: data freshness
        try:
            latest_row = conn.execute(text(
                "SELECT MAX(prediction_date) AS latest FROM predictions_log"
            )).fetchone()
            latest_date = latest_row.latest
            if latest_date:
                age = (date.today() - latest_date).days
                if age <= 1:
                    _print_result(_record("Data freshness", "PASS",
                                           f"latest prediction: {latest_date} ({age}d ago)", 5, 5))
                elif age <= 3:
                    _print_result(_record("Data freshness", "WARN",
                                           f"latest prediction: {latest_date} ({age}d ago)", 3, 5))
                else:
                    _print_result(_record("Data freshness", "FAIL",
                                           f"stale: latest prediction {age}d ago", 0, 5))
            else:
                _print_result(_record("Data freshness", "FAIL", "no predictions found", 0, 5))
        except Exception as exc:
            _print_result(_record("Data freshness", "FAIL", str(exc), 0, 5))

        # Check 4: scores are valid
        try:
            score_row = conn.execute(text("""
                SELECT
                    MIN(score)                                    AS min_score,
                    MAX(score)                                    AS max_score,
                    AVG(score)                                    AS avg_score,
                    SUM(CASE WHEN score < 0 OR score > 1 THEN 1 ELSE 0 END) AS out_of_range
                FROM predictions_log
            """)).fetchone()
            out = score_row.out_of_range or 0
            if out:
                _print_result(_record("Score validity", "FAIL",
                                       f"{out} scores outside [0,1]", 0, 5))
            else:
                _print_result(_record("Score validity", "PASS",
                                       f"min={score_row.min_score:.3f}  "
                                       f"max={score_row.max_score:.3f}  "
                                       f"avg={score_row.avg_score:.3f}", 5, 5))
        except Exception as exc:
            _print_result(_record("Score validity", "FAIL", str(exc), 0, 5))


# ── Stage 5: Dashboard Data Serving ──────────────────────────────────────────

def stage_dashboard() -> None:
    _print_stage("STAGE 5 - DASHBOARD DATA SERVING")

    # Check 1: Postgres query (same path as load_predictions in app.py)
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM predictions_log ORDER BY prediction_date DESC LIMIT 2000")
            )
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        if df.empty:
            _print_result(_record("Dashboard query", "FAIL",
                                   "predictions_log returned 0 rows", 0, 5))
        else:
            tickers = df["ticker"].nunique()
            latest = pd.to_datetime(df["prediction_date"]).max().date()
            _print_result(_record("Dashboard query", "PASS",
                                   f"{len(df)} rows  {tickers} tickers  latest: {latest}", 5, 5))
    except Exception as exc:
        _print_result(_record("Dashboard query", "SKIP",
                               f"DB not reachable --{exc}", 0, 5))

    # Check 2: source file freshness (same as load_data_freshness in app.py)
    sources = {
        "market":      ROOT / "data/processed/market.parquet",
        "fundamentals":ROOT / "data/processed/fundamentals.parquet",
        "sentiment":   ROOT / "data/processed/sentiment.parquet",
        "macro":       ROOT / "data/processed/macro.parquet",
        "warn_labels": ROOT / "data/processed/warn_raw.parquet",
        "features":    ROOT / "data/features/feature_matrix.parquet",
    }
    found = {k: p for k, p in sources.items() if p.exists()}
    missing = [k for k, p in sources.items() if not p.exists()]
    if missing:
        _print_result(_record("Source freshness", "WARN",
                               f"{len(found)}/6 files found  missing: {missing}", 3, 5))
    else:
        oldest = max(
            (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days
            for p in found.values()
        )
        _print_result(_record("Source freshness", "PASS" if oldest <= 2 else "WARN",
                               f"all 6 files found  oldest: {oldest}d ago",
                               5 if oldest <= 2 else 3, 5))

    # Check 3: local predictions parquet (fallback when DB is down)
    pred_parquet = ROOT / "data/processed/predictions_log.parquet"
    if pred_parquet.exists():
        try:
            df_local = pd.read_parquet(pred_parquet)
            tickers = df_local["ticker"].nunique() if "ticker" in df_local.columns else 0
            _print_result(_record("Local predictions log", "PASS",
                                   f"{len(df_local)} rows  {tickers} tickers", 5, 5))
        except Exception as exc:
            _print_result(_record("Local predictions log", "FAIL", str(exc), 0, 5))
    else:
        _print_result(_record("Local predictions log", "FAIL", "file not found", 0, 5))


# ── Final score report ────────────────────────────────────────────────────────

def _print_summary() -> None:
    counted = [r for r in _results if r.status != "SKIP"]
    total_score = sum(r.score for r in counted)
    total_max   = sum(r.max_score for r in counted)
    skipped     = len(_results) - len(counted)
    pct = total_score / total_max * 100 if total_max else 0

    bar_len = 40
    filled  = int(bar_len * pct / 100)
    bar     = "#" * filled + "." * (bar_len - filled)

    if pct >= 90:
        grade, colour = "EXCELLENT", _green
    elif pct >= 75:
        grade, colour = "GOOD", _green
    elif pct >= 55:
        grade, colour = "FAIR", _yellow
    else:
        grade, colour = "NEEDS ATTENTION", _red

    print("\n" + "=" * 70)
    print(_bold("PIPELINE HEALTH SUMMARY"))
    print("-" * 70)

    # Per-stage breakdown
    stages = {
        "DATA LAYER":       [r for r in _results if r.label in
                             ("Market (OHLCV)", "Fundamentals", "Sentiment", "Macro (BLS)", "WARN Labels")],
        "ETL":              [r for r in _results if r.label in
                             ("Feature matrix", "Positive labels", "Market null rate", "Ticker coverage",
                              "Feature registry", "Versioned snapshots")],
        "ML MODEL":         [r for r in _results if "Model" in r.label or "Score" in r.label
                             or "score" in r.label.lower() or "variance" in r.label.lower()],
        "POSTGRES":         [r for r in _results if r.label in
                             ("Row count", "No duplicates", "Data freshness", "Score validity")],
        "DASHBOARD":        [r for r in _results if r.label in
                             ("Dashboard query", "Source freshness", "Local predictions log")],
    }
    for stage_name, stage_results in stages.items():
        if not stage_results:
            continue
        s_score = sum(r.score for r in stage_results if r.status != "SKIP")
        s_max   = sum(r.max_score for r in stage_results if r.status != "SKIP")
        s_skip  = sum(1 for r in stage_results if r.status == "SKIP")
        skip_note = f"  ({s_skip} skipped)" if s_skip else ""
        print(f"  {stage_name:<18} {s_score:>2}/{s_max:<2}{skip_note}")

    print()
    print(f"  {colour(bar)}  {colour(f'{total_score}/{total_max}')}")
    print(f"  {colour(_bold(f'FINAL SCORE: {pct:.0f}% --{grade}'))}")
    if skipped:
        print(f"  {_grey(f'({skipped} checks skipped --run with a live DB for full score)')}")
    print("=" * 70)

    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"  Checked at {now}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(_bold("\nPIPELINE HEALTH CHECK"))
    print("=" * 70)

    stage_data_layer()
    stage_etl()
    stage_ml()
    stage_postgres()
    stage_dashboard()
    _print_summary()

    # Exit code reflects overall health (useful for CI)
    counted = [r for r in _results if r.status != "SKIP"]
    pct = sum(r.score for r in counted) / sum(r.max_score for r in counted) * 100 if counted else 0
    sys.exit(0 if pct >= 75 else 1)
