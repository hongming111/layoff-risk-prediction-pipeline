"""Tests for core_etl/validator.py — Pydantic data validation gates."""

from __future__ import annotations


import numpy as np
import pytest

from core_etl.validator import validate_feature_matrix, validate_predictions


# ── validate_feature_matrix ───────────────────────────────────────────────────

class TestValidateFeatureMatrix:
    def test_passes_on_valid_data(self, valid_feature_df):
        validate_feature_matrix(valid_feature_df)  # should not raise

    def test_passes_with_positive_labels_required(self, valid_feature_df):
        validate_feature_matrix(valid_feature_df, require_positive_labels=True)

    def test_raises_on_missing_column(self, valid_feature_df):
        df = valid_feature_df.drop(columns=["close"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_feature_matrix(df)

    def test_raises_on_empty_dataframe(self, valid_feature_df):
        df = valid_feature_df.iloc[0:0]
        with pytest.raises(ValueError, match="empty"):
            validate_feature_matrix(df)

    def test_raises_on_null_ticker(self, valid_feature_df):
        df = valid_feature_df.copy()
        df.loc[0, "ticker"] = None
        with pytest.raises(ValueError, match="ticker.*null"):
            validate_feature_matrix(df)

    def test_raises_when_critical_column_all_null(self, valid_feature_df):
        df = valid_feature_df.copy()
        df["close"] = np.nan
        with pytest.raises(ValueError, match="'close'.*null for every row"):
            validate_feature_matrix(df)

    def test_raises_when_no_positive_labels_and_required(self, valid_feature_df):
        df = valid_feature_df.copy()
        df["label"] = 0
        with pytest.raises(ValueError, match="zero positive labels"):
            validate_feature_matrix(df, require_positive_labels=True)

    def test_passes_when_no_positive_labels_not_required(self, valid_feature_df):
        df = valid_feature_df.copy()
        df["label"] = 0
        validate_feature_matrix(df, require_positive_labels=False)

    def test_raises_on_blank_ticker(self, valid_feature_df):
        df = valid_feature_df.copy()
        df.loc[0, "ticker"] = "   "
        with pytest.raises(ValueError):
            validate_feature_matrix(df)

    def test_tolerates_partial_nulls_in_non_critical_columns(self, valid_feature_df):
        df = valid_feature_df.copy()
        df["sentiment_score"] = np.nan   # sentiment can be missing — not critical
        validate_feature_matrix(df)  # should not raise


# ── validate_predictions ──────────────────────────────────────────────────────

class TestValidatePredictions:
    def test_passes_on_valid_data(self, valid_predictions_df):
        validate_predictions(valid_predictions_df)

    def test_raises_on_missing_column(self, valid_predictions_df):
        df = valid_predictions_df.drop(columns=["score"])
        with pytest.raises(ValueError, match="missing columns"):
            validate_predictions(df)

    def test_raises_on_empty_dataframe(self, valid_predictions_df):
        df = valid_predictions_df.iloc[0:0]
        with pytest.raises(ValueError, match="empty"):
            validate_predictions(df)

    def test_raises_on_score_above_one(self, valid_predictions_df):
        df = valid_predictions_df.copy()
        df.loc[0, "score"] = 1.5
        with pytest.raises(ValueError, match="score"):
            validate_predictions(df)

    def test_raises_on_score_below_zero(self, valid_predictions_df):
        df = valid_predictions_df.copy()
        df.loc[0, "score"] = -0.1
        with pytest.raises(ValueError, match="score"):
            validate_predictions(df)

    def test_passes_on_boundary_scores(self, valid_predictions_df):
        df = valid_predictions_df.copy()
        df["score"] = [0.0, 1.0, 0.5]
        validate_predictions(df)

    def test_raises_on_blank_ticker(self, valid_predictions_df):
        df = valid_predictions_df.copy()
        df.loc[0, "ticker"] = ""
        with pytest.raises(ValueError):
            validate_predictions(df)
