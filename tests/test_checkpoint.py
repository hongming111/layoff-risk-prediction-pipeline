"""Tests for ingestion/checkpoint.py — checkpoint/resume and watermark logic."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from ingestion.checkpoint import (
    clear_checkpoint,
    load_checkpoint,
    read_watermark,
    save_checkpoint,
)


# ── Checkpoint / resume ───────────────────────────────────────────────────────

class TestLoadCheckpoint:
    def test_returns_empty_set_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.checkpoint._CHECKPOINT_DIR", tmp_path)
        assert load_checkpoint("test") == set()

    def test_returns_empty_set_when_stale_date(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.checkpoint._CHECKPOINT_DIR", tmp_path)
        yesterday = str(date.today() - timedelta(days=1))
        (tmp_path / ".checkpoint_test.json").write_text(
            json.dumps({"run_date": yesterday, "completed": ["AAPL", "MSFT"]}),
            encoding="utf-8",
        )
        assert load_checkpoint("test") == set()

    def test_returns_completed_set_for_today(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.checkpoint._CHECKPOINT_DIR", tmp_path)
        (tmp_path / ".checkpoint_test.json").write_text(
            json.dumps({"run_date": str(date.today()), "completed": ["AAPL", "MSFT"]}),
            encoding="utf-8",
        )
        assert load_checkpoint("test") == {"AAPL", "MSFT"}

    def test_returns_empty_set_on_corrupt_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.checkpoint._CHECKPOINT_DIR", tmp_path)
        (tmp_path / ".checkpoint_test.json").write_text("not-json", encoding="utf-8")
        assert load_checkpoint("test") == set()


class TestSaveAndClearCheckpoint:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.checkpoint._CHECKPOINT_DIR", tmp_path)
        save_checkpoint("test", {"AAPL", "MSFT", "GOOG"})
        assert load_checkpoint("test") == {"AAPL", "MSFT", "GOOG"}

    def test_incremental_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.checkpoint._CHECKPOINT_DIR", tmp_path)
        done: set[str] = set()
        for ticker in ["AAPL", "MSFT", "GOOG"]:
            done.add(ticker)
            save_checkpoint("test", done)
        assert load_checkpoint("test") == {"AAPL", "MSFT", "GOOG"}

    def test_clear_removes_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.checkpoint._CHECKPOINT_DIR", tmp_path)
        save_checkpoint("test", {"AAPL"})
        clear_checkpoint("test")
        assert not (tmp_path / ".checkpoint_test.json").exists()

    def test_clear_is_safe_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ingestion.checkpoint._CHECKPOINT_DIR", tmp_path)
        clear_checkpoint("nonexistent")  # should not raise


# ── Watermark ─────────────────────────────────────────────────────────────────

class TestReadWatermark:
    def test_fallback_when_file_missing(self, tmp_path):
        result = read_watermark(tmp_path / "nonexistent.parquet", fallback_days=30)
        assert result == date.today() - timedelta(days=30)

    def test_returns_max_date_from_parquet(self, tmp_path):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-03-15", "2025-02-10"]),
            "value": [1.0, 2.0, 3.0],
        })
        p = tmp_path / "test.parquet"
        df.to_parquet(p, index=False)
        assert read_watermark(p) == date(2025, 3, 15)

    def test_fallback_when_file_is_empty(self, tmp_path):
        df = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})
        p = tmp_path / "empty.parquet"
        df.to_parquet(p, index=False)
        result = read_watermark(p, fallback_days=7)
        assert result == date.today() - timedelta(days=7)

    def test_fallback_when_date_col_missing(self, tmp_path):
        df = pd.DataFrame({"value": [1.0, 2.0]})
        p = tmp_path / "no_date.parquet"
        df.to_parquet(p, index=False)
        result = read_watermark(p, date_col="date", fallback_days=14)
        assert result == date.today() - timedelta(days=14)

    def test_watermark_is_older_than_today(self, tmp_path):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-01", "2024-12-31"]),
            "value": [1.0, 2.0],
        })
        p = tmp_path / "old.parquet"
        df.to_parquet(p, index=False)
        result = read_watermark(p)
        assert result < date.today()
