"""Feature engineering and dataset splitting module.

This module is responsible for:
- Computing financial technical indicators (SMA ratio, RSI, MACD delta).
- Scaling and formatting observation features for reinforcement learning.
- Splitting processed time-series data chronologically into train, validation, and test sets.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators and append feature columns to DataFrame.

    Indicators added:
    - `ret`: Daily percentage return.
    - `sma10`: 10-period Simple Moving Average of Close price.
    - `sma30`: 30-period Simple Moving Average of Close price.
    - `sma_ratio`: (SMA10 / SMA30) - 1 (trend indicator).
    - `rsi`: 14-period Exponentially Weighted Moving Average RSI, scaled to [0, 1].
    - `macd_delta`: (MACD - signal) / Close (scale-free momentum).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame containing at least a 'Close' column.

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with computed indicators, with initial NaN rows dropped.
    """
    df = df.copy()
    c = df["Close"]

    # Daily return
    df["ret"] = c.pct_change()

    # SMA Ratio
    df["sma10"] = c.rolling(window=10).mean()
    df["sma30"] = c.rolling(window=30).mean()
    df["sma_ratio"] = (df["sma10"] / df["sma30"]) - 1.0

    # RSI (14-period EWM, scaled 0-1)
    delta = c.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    rsi_raw = (100.0 - (100.0 / (1.0 + rs))) / 100.0
    df["rsi"] = rsi_raw.clip(lower=0.0, upper=1.0)

    # MACD Delta
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_delta"] = (macd - signal) / c

    # Drop warm-up periods containing NaNs
    return df.dropna()


def split_data(
    df: pd.DataFrame,
    train_end_year: int = 2021,
    val_year: int = 2022,
    test_start_year: int = 2023,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train, validation, and test subsets based on year.

    Splits:
    - Train: up to 2021 (inclusive)
    - Validation: 2022
    - Test: 2023 onward (inclusive)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex or parseable date index.
    train_end_year : int, default 2021
        Latest year for training data.
    val_year : int, default 2022
        Calendar year for validation data.
    test_start_year : int, default 2023
        Starting year for testing data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    train_df = df[df.index.year <= train_end_year]
    val_df = df[df.index.year == val_year]
    test_df = df[df.index.year >= test_start_year]

    return train_df, val_df, test_df
