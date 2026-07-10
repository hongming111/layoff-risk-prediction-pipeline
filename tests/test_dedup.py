"""Tests for deduplication in all persist_to_parquet functions."""

from __future__ import annotations


import pandas as pd
import pytest


# ── Market ────────────────────────────────────────────────────────────────────

class TestMarketDedup:
    def test_drops_duplicate_ticker_date(self, tmp_path):
        from ingestion.market import persist_to_parquet

        df = pd.DataFrame({
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "date":   pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01"]),
            "close":  [150.0, 155.0, 300.0],
            "vol_7d": [0.2, 0.21, 0.15],
            "vol_14d":[0.2, 0.21, 0.15],
            "vol_21d":[0.2, 0.21, 0.15],
        })
        out = tmp_path / "market.parquet"
        persist_to_parquet(df, path=out)
        result = pd.read_parquet(out)
        assert len(result) == 2  # duplicate AAPL 2025-01-01 removed

    def test_keep_last_on_duplicate(self, tmp_path):
        from ingestion.market import persist_to_parquet

        df = pd.DataFrame({
            "ticker": ["AAPL", "AAPL"],
            "date":   pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "close":  [150.0, 999.0],   # second row should win
            "vol_7d": [0.2, 0.9],
            "vol_14d":[0.2, 0.9],
            "vol_21d":[0.2, 0.9],
        })
        out = tmp_path / "market.parquet"
        persist_to_parquet(df, path=out)
        result = pd.read_parquet(out)
        assert result["close"].iloc[0] == pytest.approx(999.0)

    def test_no_duplicates_unchanged(self, tmp_path):
        from ingestion.market import persist_to_parquet

        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "date":   pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "close":  [150.0, 300.0],
            "vol_7d": [0.2, 0.15],
            "vol_14d":[0.2, 0.15],
            "vol_21d":[0.2, 0.15],
        })
        out = tmp_path / "market.parquet"
        persist_to_parquet(df, path=out)
        assert len(pd.read_parquet(out)) == 2


# ── Fundamentals ──────────────────────────────────────────────────────────────

class TestFundamentalsDedup:
    def _make_df(self, extra_row=False):
        rows = [
            {"ticker": "AAPL", "date": pd.Timestamp("2025-01-01"),
             "debt_to_equity": 1.2, "current_ratio": 1.5, "profit_margin": 0.25},
            {"ticker": "AAPL", "date": pd.Timestamp("2025-01-01"),  # duplicate
             "debt_to_equity": 9.9, "current_ratio": 9.9, "profit_margin": 9.9},
        ]
        if extra_row:
            rows.append({"ticker": "MSFT", "date": pd.Timestamp("2025-01-01"),
                         "debt_to_equity": 0.8, "current_ratio": 2.0, "profit_margin": 0.35})
        return pd.DataFrame(rows)

    def test_drops_duplicate(self, tmp_path):
        from ingestion.fundamentals import persist_to_parquet
        out = tmp_path / "fundamentals.parquet"
        persist_to_parquet(self._make_df(), path=out)
        assert len(pd.read_parquet(out)) == 1

    def test_keeps_unique_rows(self, tmp_path):
        from ingestion.fundamentals import persist_to_parquet
        out = tmp_path / "fundamentals.parquet"
        persist_to_parquet(self._make_df(extra_row=True), path=out)
        assert len(pd.read_parquet(out)) == 2


# ── Sentiment ─────────────────────────────────────────────────────────────────

class TestSentimentDedup:
    def test_drops_duplicate_ticker_date(self, tmp_path):
        from ingestion.sentiment import persist_to_parquet

        df = pd.DataFrame({
            "ticker":               ["AAPL", "AAPL", "MSFT"],
            "date":                 pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01"]),
            "sentiment_score":      [0.5, 0.9, 0.1],
            "sentiment_score_ma7d": [0.4, 0.8, 0.1],
            "mention_velocity":     [10.0, 20.0, 5.0],
        })
        out = tmp_path / "sentiment.parquet"
        persist_to_parquet(df, path=out)
        assert len(pd.read_parquet(out)) == 2


# ── Macro ─────────────────────────────────────────────────────────────────────

class TestMacroDedup:
    def test_drops_duplicate_date(self, tmp_path):
        from ingestion.macro import persist_to_parquet

        df = pd.DataFrame({
            "date":                     pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28"]),
            "unemployment_rate_total":  [4.1, 4.2, 4.0],
            "layoff_rate_total":        [1.2, 1.3, 1.1],
            "layoff_rate_tech":         [0.8, 0.9, 0.7],
        })
        out = tmp_path / "macro.parquet"
        persist_to_parquet(df, path=out)
        assert len(pd.read_parquet(out)) == 2

    def test_keep_last_on_duplicate(self, tmp_path):
        from ingestion.macro import persist_to_parquet

        df = pd.DataFrame({
            "date":                     pd.to_datetime(["2025-01-31", "2025-01-31"]),
            "unemployment_rate_total":  [4.1, 9.9],   # second should win
            "layoff_rate_total":        [1.2, 9.9],
            "layoff_rate_tech":         [0.8, 9.9],
        })
        out = tmp_path / "macro.parquet"
        persist_to_parquet(df, path=out)
        result = pd.read_parquet(out)
        assert result["unemployment_rate_total"].iloc[0] == pytest.approx(9.9)


# ── Predictions log append ────────────────────────────────────────────────────

class TestPredictionsDedup:
    def test_drops_duplicate_ticker_prediction_date(self, tmp_path, monkeypatch):
        import datetime
        from ml_engine.predict import _append_to_parquet

        monkeypatch.setattr("ml_engine.predict.PREDICTIONS_PARQUET", tmp_path / "predictions_log.parquet")

        today = datetime.date.today()
        df1 = pd.DataFrame({
            "ticker": ["AAPL"], "prediction_date": [today],
            "target_date": [today], "score": [0.3], "model_version": ["v1"],
        })
        df2 = pd.DataFrame({
            "ticker": ["AAPL"], "prediction_date": [today],
            "target_date": [today], "score": [0.9], "model_version": ["v2"],  # retry → higher score
        })
        _append_to_parquet(df1)
        _append_to_parquet(df2)

        result = pd.read_parquet(tmp_path / "predictions_log.parquet")
        assert len(result) == 1
        assert result["score"].iloc[0] == pytest.approx(0.9)  # last write wins
