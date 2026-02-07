from datetime import date, timedelta
from typing import cast

import numpy as np
import polars as pl
import pytest

from quant_trade.model.process import (
    GaussianLabelBuilder,
    PurgedKFold,
    PurgedTimeSplit,
    WalkForwardValidation,
)


def default_start_date() -> date:
    return date(2015, 1, 12)


def mock_stocks(
    n_stocks: int = 10, n_days: int = 1000, start_date: date | None = None
) -> pl.DataFrame:
    """Create synthetic financial data for testing."""
    # Create dates as a list of strings first, then convert
    if start_date is None:
        start_date = default_start_date()
    start_dt = start_date
    dates = [start_dt + timedelta(days=i) for i in range(n_days)]

    data = []
    for i in range(n_stocks):
        stock_data = {
            "date": dates,
            "ts_code": [f"stock_{i:03d}"] * n_days,
            "close": 100.0 + np.random.randn(n_days).cumsum() * 2,
            "ret_fwd_21d": np.random.randn(n_days) * 0.02,
            "ret_fwd_63d": np.random.randn(n_days) * 0.03,
            "feature_1_z": np.random.randn(n_days),
            "feature_2_z": np.random.randn(n_days),
            "sw_l1_code": [f"industry_{i % 3}"] * n_days,
        }
        data.append(pl.DataFrame(stock_data))

    return pl.concat(data).sort(["date", "ts_code"])


def mock_label_end_dates(df: pl.DataFrame, horizon: int = 21) -> pl.DataFrame:
    """Add label end dates for purging tests."""
    return df.with_columns(
        [(pl.col("date") + pl.duration(days=horizon)).alias("label_end_date")]
    )


# ============================================================================
# PurgedTimeSplit Tests
# ============================================================================


def test_minimal_purged_split():
    """Minimal test that doesn't require complex setup."""
    # Create minimal test data
    df = mock_stocks(10, 100)
    split_date = default_start_date() + timedelta(days=50)
    splitter = PurgedTimeSplit(horizon_days=5, embargo_days=2)
    train, test = splitter.split(df, split_date=split_date, date_col="date")

    assert len(train) > 0
    assert len(test) > 0
    assert cast(date, train["date"].max()) < split_date - timedelta(
        days=5
    )  # purge cutoff
    assert cast(date, test["date"].min()) >= split_date + timedelta(
        days=2
    )  # embargo start


def test_purged_time_split_basic():
    """Test basic functionality of PurgedTimeSplit."""
    df = mock_stocks(n_days=100)
    splitter = PurgedTimeSplit(horizon_days=21, embargo_days=5)

    split_date = default_start_date() + timedelta(days=50)
    train, test = splitter.split(df, split_date=split_date, date_col="date")

    assert isinstance(train, pl.DataFrame)
    assert isinstance(test, pl.DataFrame)

    assert len(train) > 0
    assert len(test) > 0

    # Check no overlap using Polars expressions
    purge_cutoff = split_date - timedelta(days=21)
    embargo_start = split_date + timedelta(days=5)

    # All training dates should be before purge cutoff
    train_after_cutoff = train.filter(pl.col("date") >= purge_cutoff)
    assert len(train_after_cutoff) == 0, (
        f"Found {len(train_after_cutoff)} training samples after purge cutoff"
    )

    # All test dates should be after embargo start
    test_before_embargo = test.filter(pl.col("date") < embargo_start)
    assert len(test_before_embargo) == 0, (
        f"Found {len(test_before_embargo)} test samples before embargo"
    )


def test_purged_time_split_with_label_end():
    """Test purging with label end dates."""
    df = mock_stocks(n_days=100)
    df = mock_label_end_dates(df, horizon=21)

    splitter = PurgedTimeSplit(horizon_days=21, embargo_days=5)
    split_date = default_start_date() + timedelta(days=50)

    train, test = splitter.split(
        df, split_date=split_date, date_col="date", label_end_col="label_end_date"
    )

    # Verify no label leakage
    purge_cutoff = split_date - timedelta(days=21)

    # All training labels should end before purge cutoff
    if "label_end_date" in train.columns:
        train_label_leak = train.filter(pl.col("label_end_date") >= purge_cutoff)
        assert len(train_label_leak) == 0, (
            f"Found {len(train_label_leak)} training labels that leak into test period"
        )

    # Test data should be after embargo
    embargo_start = split_date + timedelta(days=5)
    test_before_embargo = test.filter(pl.col("date") < embargo_start)
    assert len(test_before_embargo) == 0, (
        f"Found {len(test_before_embargo)} test samples before embargo"
    )


def test_purged_time_split_edge_cases():
    """Test edge cases and error handling."""
    # Create very small dataset
    df = mock_stocks(n_days=100, n_stocks=2)

    splitter = PurgedTimeSplit(horizon_days=21, embargo_days=5)

    split_date = default_start_date() + timedelta(days=50)
    train, test = splitter.split(df, split_date=split_date, date_col="date")
    assert len(train) > 0
    assert len(test) > 0

    # Should fail with very late split date (no test data)
    with pytest.raises(ValueError):
        splitter.split(df, split_date=date(2025, 1, 1), date_col="date")

    # Should fail with very early split date (no training data)
    with pytest.raises(ValueError):
        splitter.split(df, split_date=date(2019, 2, 1), date_col="date")


# ============================================================================
# PurgedKFold Tests
# ============================================================================


def test_purged_kfold_basic():
    """Test basic K-fold splitting."""
    df = mock_stocks(n_days=500)
    kfold = PurgedKFold(n_splits=5, horizon_days=21, embargo_days=5)

    folds = list(kfold.split(df, date_col="date"))
    assert len(folds) == 5

    for i, (train, test) in enumerate(folds):
        assert len(train) > 0, f"Fold {i}: Empty training set"
        assert len(test) > 0, f"Fold {i}: Empty test set"
        assert cast(date, train["date"].max()) < cast(date, test["date"].min()), (
            f"Fold {i}: Training dates overlap with test dates"
        )


def test_purged_kfold_with_label_end():
    """Test K-fold with label end dates."""
    df = mock_stocks(n_days=500)
    df = mock_label_end_dates(df, horizon=21)

    kfold = PurgedKFold(n_splits=3, horizon_days=21, embargo_days=5, min_train_size=10)

    for train, test in kfold.split(df, date_col="date", label_end_col="label_end_date"):
        # Verify no label leakage
        test_start = cast(date, test["date"].min())
        purge_cutoff = test_start - timedelta(days=21 + 5)

        if "label_end_date" in train.columns:
            assert cast(date, train["label_end_date"].max()) < purge_cutoff

        # Check minimum sizes
        assert len(train) >= 10
        assert len(test) > 0


def test_purged_kfold_insufficient_data():
    """Test K-fold with insufficient data."""
    df = mock_stocks(n_days=5)  # Very small dataset

    kfold = PurgedKFold(n_splits=3, horizon_days=21, embargo_days=5)

    with pytest.raises(ValueError, match="Not enough unique dates"):
        list(kfold.split(df, date_col="date"))


def test_walkforward_expanding():
    """Test expanding window walk-forward."""
    df = mock_stocks(n_days=500)
    wfv = WalkForwardValidation(
        n_windows=3,
        window_size=100,
        step_size=50,
        horizon_days=21,
        embargo_days=5,
        expanding=True,
    )

    windows = list(wfv.split(df, date_col="date"))
    assert len(windows) > 0

    # Check expanding nature
    train_sizes = [len(train) for train, _ in windows]
    assert all(
        train_sizes[i] <= train_sizes[i + 1] for i in range(len(train_sizes) - 1)
    )


def test_walkforward_rolling():
    """Test rolling window walk-forward."""
    df = mock_stocks(n_days=500)
    wfv = WalkForwardValidation(
        n_windows=3,
        window_size=100,
        step_size=50,
        horizon_days=21,
        embargo_days=5,
        expanding=False,
    )

    windows = list(wfv.split(df, date_col="date"))

    if len(windows) >= 2:
        train_sizes = [len(train) for train, _ in windows]
        size_diff = max(train_sizes) - min(train_sizes)
        assert size_diff <= 50


# # ============================================================================
# # GaussianLabelBuilder Tests
# # ============================================================================


def test_gaussian_label_builder_basic():
    """Test basic Gaussian label transformation."""
    # Create simple test data
    builder = GaussianLabelBuilder(
        factor="ret_fwd_21d",
        rank_by="date",
        by=["sw_l1_code"],
        winsor_limits=(0.05, 0.95),
    )

    labeled_df = builder.label(mock_stocks(3, 100))

    # Check output columns
    label_col = builder.label_name
    assert label_col in labeled_df.columns
    assert label_col == "label_ret_fwd_21d"

    # Check no null labels
    assert labeled_df[label_col].null_count() == 0

    # Check properties
    assert builder.label_name == "label_ret_fwd_21d"
    assert builder.rank_by_name == "date"


def test_gaussian_label_transformation_properties():
    """Test mathematical properties of Gaussian transformation."""
    test_data = mock_stocks(n_stocks=10, n_days=100)
    builder = GaussianLabelBuilder(factor="ret_fwd_21d", rank_by="date", by=None)

    labeled_df = builder.label(test_data)
    for date_ in test_data["date"]:
        date_data = labeled_df.filter(pl.col("date") == date_)
        returns = date_data["ret_fwd_21d"].to_numpy()
        labels = date_data[builder.label_name].to_numpy()

        sorted_indices = np.argsort(returns)
        sorted_labels = labels[sorted_indices]

        # Labels should be increasing (higher returns → higher labels)
        diffs = np.diff(sorted_labels)
        assert np.all(np.argsort(returns) == np.argsort(labels))
        assert np.all(diffs > -1e-12), f"Non-monotonic transformation found: {diffs}"


def test_gaussian_label_builder_with_missing():
    """Test handling of missing values."""
    test_data = mock_stocks(10, 100)
    builder = GaussianLabelBuilder(factor="ret_fwd_21d", rank_by="date")
    labeled_df = builder.label(test_data)

    # Should filter out null returns
    assert labeled_df["label_ret_fwd_21d"].null_count() == 0

    # Check we have fewer rows than original (nulls removed)
    original_with_nulls = test_data.filter(pl.col("ret_fwd_21d").is_not_null())
    assert len(labeled_df) == len(original_with_nulls)
