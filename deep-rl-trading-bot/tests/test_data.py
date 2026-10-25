"""Tests for market data downloading, column flattening, and caching."""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.data import download_data


@pytest.fixture
def mock_yfinance_raw():
    """Create a mock yfinance DataFrame with MultiIndex columns and some NaNs."""
    dates = pd.date_range("2021-01-01", periods=10, freq="B")
    cols = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Volume", "AAPL"),
        ],
        names=["Price", "Ticker"],
    )
    data = np.arange(50, dtype=float).reshape((10, 5))
    # Insert a NaN row to test dropna
    data[3, 1] = np.nan

    df = pd.DataFrame(data, index=dates, columns=cols)
    return df


def test_download_data_flatten_and_cache(tmp_path, mock_yfinance_raw):
    """Test that download_data flattens MultiIndex columns, drops NaNs, and caches CSV."""
    with patch("yfinance.download", return_value=mock_yfinance_raw) as mock_dl:
        cache_dir = tmp_path / "data"
        df = download_data("AAPL", start="2021-01-01", end="2021-01-15", cache_dir=cache_dir, use_cache=False)

        mock_dl.assert_called_once_with("AAPL", start="2021-01-01", end="2021-01-15", auto_adjust=True, progress=False)

        # MultiIndex should be flattened
        assert not isinstance(df.columns, pd.MultiIndex)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

        # NaN row at index 3 should be dropped
        assert not df.isna().any().any()
        assert len(df) == 9

        # CSV file should be created in cache_dir
        cached_file = cache_dir / "AAPL.csv"
        assert cached_file.exists()


def test_download_data_load_from_cache(tmp_path, mock_yfinance_raw):
    """Test that subsequent calls load from cache without re-downloading."""
    with patch("yfinance.download", return_value=mock_yfinance_raw) as mock_dl:
        cache_dir = tmp_path / "data"

        # First call downloads and caches
        df1 = download_data("AAPL", start="2021-01-01", end="2021-01-15", cache_dir=cache_dir, use_cache=True)
        assert mock_dl.call_count == 1

        # Second call should load from cache, not calling yf.download again
        df2 = download_data("AAPL", start="2021-01-01", end="2021-01-15", cache_dir=cache_dir, use_cache=True)
        assert mock_dl.call_count == 1

        pd.testing.assert_frame_equal(df1, df2)
