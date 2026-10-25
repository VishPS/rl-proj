"""Market data ingestion and preprocessing module.

This module is responsible for:
- Fetching historical OHLCV market data using yfinance with auto_adjust=True.
- Cleaning, validating, and flattening column structures.
- Keeping core price/volume series (Open, High, Low, Close, Volume).
- Caching downloaded datasets to CSV for offline access and reproducibility.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
import yfinance as yf


def download_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache_dir: Union[str, Path] = "data",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download historical stock data from Yahoo Finance and cache to disk.

    Parameters
    ----------
    ticker : str
        Ticker symbol to download (e.g., 'AAPL', 'MSFT').
    start : str, optional
        Start date in 'YYYY-MM-DD' format.
    end : str, optional
        End date in 'YYYY-MM-DD' format.
    cache_dir : str or Path, default "data"
        Directory path where cached CSV files are stored.
    use_cache : bool, default True
        If True and cache file exists covering the date range, loads from disk.

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV DataFrame with DatetimeIndex and columns:
        ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    cache_dir_path = Path(cache_dir)
    cache_file = cache_dir_path / f"{ticker}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Check whether cache covers requested date range
        covers_start = True if start is None else (df.index.min() <= pd.to_datetime(start) + pd.Timedelta(days=5))
        covers_end = True if end is None else (df.index.max() >= pd.to_datetime(end) - pd.Timedelta(days=5))

        if covers_start and covers_end:
            if start is not None:
                df = df.loc[df.index >= pd.to_datetime(start)]
            if end is not None:
                df = df.loc[df.index <= pd.to_datetime(end)]
            if not df.empty:
                return df

    # Fetch data using yfinance with auto_adjust=True
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Flatten multi-index columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns.name = None

    # Keep core OHLCV columns and drop NaNs
    target_columns = ["Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in target_columns if c in raw.columns]
    df = raw[available_cols].dropna()

    # Normalize DatetimeIndex and timezone
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    # Cache result to data/{ticker}.csv
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file)

    return df
