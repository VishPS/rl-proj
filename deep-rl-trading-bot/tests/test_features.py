"""Tests for feature engineering and dataset splitting."""

import numpy as np
import pandas as pd
import pytest

from src.features import add_features, split_data


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate synthetic OHLCV data spanning multiple years."""
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="B")
    n = len(dates)
    np.random.seed(42)

    # Random walk for close prices
    returns = np.random.normal(loc=0.0005, scale=0.015, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.abs(np.random.normal(0, 0.005, size=n)))
    low = close * (1.0 - np.abs(np.random.normal(0, 0.005, size=n)))
    open_p = low + (high - low) * np.random.uniform(0.1, 0.9, size=n)
    volume = np.random.randint(1_000_000, 10_000_000, size=n)

    df = pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_add_features_no_nans(sample_ohlcv_data):
    """Test that add_features produces a DataFrame with absolutely no NaN values."""
    feat_df = add_features(sample_ohlcv_data)

    # Check dataframe is non-empty
    assert not feat_df.empty, "DataFrame should not be empty after adding features"

    # Check that no NaN or null values exist anywhere in the DataFrame
    assert not feat_df.isna().any().any(), "DataFrame contains NaN values after add_features"
    assert not feat_df.isnull().any().any(), "DataFrame contains null values after add_features"

    # Verify expected columns exist
    expected_cols = {"Open", "High", "Low", "Close", "Volume", "ret", "sma10", "sma30", "sma_ratio", "rsi", "macd_delta"}
    assert expected_cols.issubset(feat_df.columns), f"Missing columns: {expected_cols - set(feat_df.columns)}"


def test_rsi_bounds(sample_ohlcv_data):
    """Test that RSI values are strictly bounded between 0.0 and 1.0."""
    feat_df = add_features(sample_ohlcv_data)

    assert (feat_df["rsi"] >= 0.0).all(), "RSI contains values strictly less than 0.0"
    assert (feat_df["rsi"] <= 1.0).all(), "RSI contains values strictly greater than 1.0"


def test_rsi_extremes():
    """Test RSI behavior under extreme price scenarios (pure gain, pure loss, constant)."""
    dates = pd.date_range("2021-01-01", periods=100, freq="D")

    # 1. Monotonically increasing price series
    up_prices = np.linspace(10, 200, 100)
    df_up = pd.DataFrame({"Close": up_prices}, index=dates)
    feat_up = add_features(df_up)
    assert (feat_up["rsi"] >= 0.0).all() and (feat_up["rsi"] <= 1.0).all()
    # Continuous gains should push RSI close to 1.0
    assert (feat_up["rsi"].iloc[-10:] > 0.9).all()

    # 2. Monotonically decreasing price series
    down_prices = np.linspace(200, 10, 100)
    df_down = pd.DataFrame({"Close": down_prices}, index=dates)
    feat_down = add_features(df_down)
    assert (feat_down["rsi"] >= 0.0).all() and (feat_down["rsi"] <= 1.0).all()
    # Continuous losses should push RSI close to 0.0
    assert (feat_down["rsi"].iloc[-10:] < 0.1).all()

    # 3. Constant prices
    flat_prices = np.full(100, 50.0)
    df_flat = pd.DataFrame({"Close": flat_prices}, index=dates)
    feat_flat = add_features(df_flat)
    assert not feat_flat.isna().any().any()
    assert (feat_flat["rsi"] >= 0.0).all() and (feat_flat["rsi"] <= 1.0).all()


def test_split_data_partitions(sample_ohlcv_data):
    """Test that split_data partitions data into train (<=2021), val (2022), and test (>=2023)."""
    feat_df = add_features(sample_ohlcv_data)
    train_df, val_df, test_df = split_data(feat_df)

    # Verify no partitions are empty given the 2020-2023 dataset
    assert len(train_df) > 0, "Train split should not be empty"
    assert len(val_df) > 0, "Validation split should not be empty"
    assert len(test_df) > 0, "Test split should not be empty"

    # Verify year ranges
    assert (train_df.index.year <= 2021).all(), "Train set contains records past 2021"
    assert (val_df.index.year == 2022).all(), "Validation set contains records not in 2022"
    assert (test_df.index.year >= 2023).all(), "Test set contains records prior to 2023"

    # Verify no overlap in indices
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    assert train_idx.isdisjoint(val_idx), "Train and Validation sets overlap"
    assert train_idx.isdisjoint(test_idx), "Train and Test sets overlap"
    assert val_idx.isdisjoint(test_idx), "Validation and Test sets overlap"

    # Verify total rows match
    assert len(train_df) + len(val_df) + len(test_df) == len(feat_df)
