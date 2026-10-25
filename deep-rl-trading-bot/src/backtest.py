"""Backtesting and strategy evaluation module.

Evaluates trained PPO models out-of-sample on the test partition,
benchmarks performance against Buy & Hold, computes key risk/return metrics
(Total Return, Sharpe Ratio, Max Drawdown), and produces two-panel visual charts.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Headless backend for scripts and servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data import download_data
from src.env import TradingEnv
from src.features import add_features, split_data


def calculate_metrics(curve: np.ndarray) -> Dict[str, float]:
    """Compute financial performance metrics for an equity curve.

    Metrics:
    - Total Return %: Total percentage gain or loss over the test window.
    - Sharpe Ratio: Annualized return / volatility (annualized with sqrt(252)).
    - Max Drawdown %: Worst peak-to-trough decline.

    Parameters
    ----------
    curve : np.ndarray
        Array of portfolio equity values over time.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'Total Return %', 'Sharpe', and 'Max Drawdown %'.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if len(curve) < 2:
        return {"Total Return %": 0.0, "Sharpe": 0.0, "Max Drawdown %": 0.0}

    # Daily percentage returns
    rets = np.diff(curve) / np.maximum(curve[:-1], 1e-8)

    # Total return
    total_return = (curve[-1] / max(curve[0], 1e-8)) - 1.0

    # Annualized Sharpe ratio (daily frequency, 252 trading days)
    std = rets.std()
    sharpe = (np.sqrt(252.0) * rets.mean() / (std + 1e-9)) if std > 1e-12 else 0.0

    # Maximum Drawdown
    peaks = np.maximum.accumulate(curve)
    drawdowns = (curve - peaks) / np.maximum(peaks, 1e-8)
    max_drawdown = float(drawdowns.min())

    return {
        "Total Return %": round(float(total_return * 100.0), 2),
        "Sharpe": round(float(sharpe), 2),
        "Max Drawdown %": round(float(max_drawdown * 100.0), 2),
    }


def run_backtest(
    model: PPO,
    df: pd.DataFrame,
    window: int = 10,
    init_cash: float = 10_000.0,
    fee: float = 0.001,
) -> TradingEnv:
    """Run deterministic policy rollout on given DataFrame.

    Parameters
    ----------
    model : PPO
        Trained Stable-Baselines3 model.
    df : pd.DataFrame
        OHLCV feature DataFrame for the backtesting period.
    window : int, default 10
        Observation window size.
    init_cash : float, default 10000.0
        Starting cash balance.
    fee : float, default 0.001
        Transaction fee rate.

    Returns
    -------
    TradingEnv
        Environment instance after completing the episode.
    """
    env = TradingEnv(df, window=window, init_cash=init_cash, fee=fee)
    obs, _ = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))

    return env


def plot_backtest(
    dates: pd.DatetimeIndex,
    agent_curve: np.ndarray,
    bh_curve: np.ndarray,
    close_prices: np.ndarray,
    trades: List[Tuple[int, str, float]],
    window: int = 10,
    ticker: str = "Asset",
    save_path: Union[str, Path] = "backtest.png",
) -> Path:
    """Generate and save two-panel backtest performance chart.

    Panel 1 (Top): Portfolio value of RL Agent vs Buy & Hold baseline.
    Panel 2 (Bottom): Asset price with green BUY (^) and red SELL (v) markers.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Date index corresponding to each step.
    agent_curve : np.ndarray
        RL Agent portfolio equity values.
    bh_curve : np.ndarray
        Buy & Hold baseline portfolio equity values.
    close_prices : np.ndarray
        Underlying asset close prices.
    trades : List[Tuple[int, str, float]]
        List of executed trades as (step, 'BUY'|'SELL', price).
    window : int, default 10
        Price window size.
    ticker : str, default "Asset"
        Ticker name for chart titles.
    save_path : str or Path, default "backtest.png"
        Output image file path.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Portfolio Value
    ax[0].plot(dates, agent_curve, label="RL Agent (PPO)", color="#1f77b4", lw=2)
    ax[0].plot(dates, bh_curve, label="Buy & Hold", color="#ff7f0e", alpha=0.75, lw=1.5, linestyle="--")
    ax[0].set_title(f"{ticker} - Out-of-Sample Portfolio Valuation", fontsize=14, fontweight="bold")
    ax[0].set_ylabel("Portfolio Value ($)", fontsize=11)
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)

    # Panel 2: Price and Trade Markers
    ax[1].plot(dates, close_prices, color="gray", lw=1.2, label=f"{ticker} Close Price")

    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    n = len(dates)

    for step, side, price in trades:
        idx = step - (window - 1)
        if 0 <= idx < n:
            if side == "BUY":
                buy_dates.append(dates[idx])
                buy_prices.append(price)
            elif side == "SELL":
                sell_dates.append(dates[idx])
                sell_prices.append(price)

    if buy_dates:
        ax[1].scatter(buy_dates, buy_prices, marker="^", color="green", s=70, label="BUY", zorder=5)
    if sell_dates:
        ax[1].scatter(sell_dates, sell_prices, marker="v", color="red", s=70, label="SELL", zorder=5)

    ax[1].set_title("Close Price & Trade Executions", fontsize=12)
    ax[1].set_ylabel("Price ($)", fontsize=11)
    ax[1].set_xlabel("Date", fontsize=11)
    ax[1].legend(loc="upper left")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_file, dpi=150)
    plt.close(fig)

    return save_file


def backtest(
    ticker: str = "AAPL",
    model_path: Optional[str] = None,
    output_plot: str = "backtest.png",
    output_csv: Optional[str] = "test_metrics.csv",
    window: int = 10,
    init_cash: float = 10_000.0,
    fee: float = 0.001,
) -> Tuple[pd.DataFrame, TradingEnv, Path]:
    """Perform complete out-of-sample backtest comparison.

    Parameters
    ----------
    ticker : str, default "AAPL"
        Stock ticker symbol.
    model_path : str, optional
        Path to saved PPO model .zip file. If None, checks default model directories.
    output_plot : str, default "backtest.png"
        Path to save output chart.
    output_csv : str, optional, default "test_metrics.csv"
        Path to save metrics comparison CSV.
    window : int, default 10
        Price window size.
    init_cash : float, default 10000.0
        Starting cash balance.
    fee : float, default 0.001
        Transaction fee rate.

    Returns
    -------
    Tuple[pd.DataFrame, TradingEnv, Path]
        (metrics_df, env, plot_path)
    """
    # Resolve model path
    if model_path is None:
        candidates = [
            Path("models/best_model/best_model.zip"),
            Path(f"models/ppo_{ticker.lower()}_final.zip"),
            Path("models/ppo_trading_best.zip"),
            Path("ppo_trading_best.zip"),
        ]
        for candidate in candidates:
            if candidate.exists():
                model_path = str(candidate)
                break

    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found. Checked: {model_path}")

    print(f"[Backtest] Loading trained agent from: {model_path}")
    agent = PPO.load(model_path, device="cpu")

    # Load data and prepare test partition
    raw_df = download_data(ticker=ticker, start="2012-01-01", end="2024-12-31")
    feat_df = add_features(raw_df)
    _, _, test_df = split_data(feat_df)

    if len(test_df) <= window:
        raise ValueError(f"Test split too small ({len(test_df)} rows) for window size ({window}).")

    print(f"[Backtest] Running deterministic rollout on {len(test_df)} test bars (2023 onward)...")
    env = run_backtest(agent, test_df, window=window, init_cash=init_cash, fee=fee)

    agent_curve = np.array(env.equity_curve, dtype=np.float64)

    # Buy & Hold baseline
    raw_prices = test_df["Close"].values[window - 1 :]
    bh_curve = init_cash * (raw_prices / raw_prices[0])

    # Align lengths
    n = min(len(agent_curve), len(bh_curve))
    agent_curve = agent_curve[:n]
    bh_curve = bh_curve[:n]
    dates = test_df.index[window - 1 : window - 1 + n]
    close_prices = raw_prices[:n]

    # Metrics computation
    metrics_df = pd.DataFrame(
        {
            "RL Agent": calculate_metrics(agent_curve),
            "Buy & Hold": calculate_metrics(bh_curve),
        }
    )

    print("\n========== Backtest Metrics ==========")
    print(metrics_df.to_string())
    print(f"Total Trades Executed: {len(env.trades)}")
    print("======================================\n")

    if output_csv:
        csv_file = Path(output_csv)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(csv_file)
        print(f"[Backtest] Saved metrics table to: {csv_file}")

    plot_file = plot_backtest(
        dates=dates,
        agent_curve=agent_curve,
        bh_curve=bh_curve,
        close_prices=close_prices,
        trades=env.trades,
        window=window,
        ticker=ticker,
        save_path=output_plot,
    )
    print(f"[Backtest] Saved plot to: {plot_file}")

    return metrics_df, env, plot_file


def parse_args() -> argparse.Namespace:
    """Parse backtesting CLI arguments."""
    parser = argparse.ArgumentParser(description="Backtest PPO Trading Agent against Buy & Hold")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to PPO model .zip (optional)")
    parser.add_argument("--output-plot", type=str, default="backtest.png", help="Output plot filename")
    parser.add_argument("--output-csv", type=str, default="test_metrics.csv", help="Output metrics CSV filename")
    parser.add_argument("--window", type=int, default=10, help="Observation window size")
    parser.add_argument("--init-cash", type=float, default=10_000.0, help="Initial cash balance")
    parser.add_argument("--fee", type=float, default=0.001, help="Transaction fee rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    backtest(
        ticker=args.ticker,
        model_path=args.model_path,
        output_plot=args.output_plot,
        output_csv=args.output_csv,
        window=args.window,
        init_cash=args.init_cash,
        fee=args.fee,
    )
