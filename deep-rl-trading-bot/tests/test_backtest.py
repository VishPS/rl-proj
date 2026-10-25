"""Tests for backtesting, performance metrics, and visualization."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.backtest import calculate_metrics, plot_backtest


def test_calculate_metrics_flat_curve():
    """Test metrics on a completely flat equity curve (no gain/loss)."""
    curve = np.full(50, 10_000.0)
    res = calculate_metrics(curve)
    assert res["Total Return %"] == 0.0
    assert res["Sharpe"] == 0.0
    assert res["Max Drawdown %"] == 0.0


def test_calculate_metrics_values():
    """Test metrics on known equity curve progressions."""
    # 1. 50% gain from 10k to 15k
    curve_up = np.linspace(10_000.0, 15_000.0, 100)
    res_up = calculate_metrics(curve_up)
    assert res_up["Total Return %"] == 50.0
    assert res_up["Sharpe"] > 0.0
    assert res_up["Max Drawdown %"] == 0.0

    # 2. 50% loss from 10k to 5k
    curve_down = np.linspace(10_000.0, 5_000.0, 100)
    res_down = calculate_metrics(curve_down)
    assert res_down["Total Return %"] == -50.0
    assert res_down["Sharpe"] < 0.0
    assert res_down["Max Drawdown %"] == -50.0


def test_plot_backtest_creates_image(tmp_path):
    """Test that plot_backtest generates a valid PNG file."""
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    agent_curve = np.linspace(10_000, 12_000, 30)
    bh_curve = np.linspace(10_000, 11_000, 30)
    prices = np.linspace(150, 165, 30)
    trades = [(9, "BUY", 150.0), (19, "SELL", 160.0)]

    plot_file = tmp_path / "test_plot.png"
    result_path = plot_backtest(
        dates=dates,
        agent_curve=agent_curve,
        bh_curve=bh_curve,
        close_prices=prices,
        trades=trades,
        window=10,
        ticker="AAPL",
        save_path=plot_file,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 1000  # Non-trivial image file
