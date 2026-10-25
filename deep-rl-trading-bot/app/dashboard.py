"""Streamlit Dashboard for Deep RL Trading Bot.

Provides an interactive web interface for:
- Configuring ticker, date range, and uploading custom trained PPO models.
- Running deterministic backtests comparing the RL agent against Buy & Hold.
- Inspecting key performance metrics (Return, Sharpe, Max Drawdown, Trade Count).
- Visualizing equity curves and price charts with execution markers.
"""

import sys
from datetime import date, datetime
from pathlib import Path
import tempfile
from typing import Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import streamlit as st

from src.backtest import calculate_metrics, run_backtest
from src.data import download_data
from src.env import TradingEnv
from src.features import add_features


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Deep RL Trading Bot | Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1a1e29;
        border: 1px solid #2e384d;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        transition: border-color 0.2s ease-in-out;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-green { background-color: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-blue { background-color: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("Configure backtesting parameters and model checkpoints.")

# 1. Ticker Symbol
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL", help="Yahoo Finance stock ticker symbol").strip().upper()

# 2. Date Range
st.sidebar.markdown("### 📅 Date Range")
col_start, col_end = st.sidebar.columns(2)
with col_start:
    start_date = st.date_input("Start Date", value=date(2023, 1, 1), min_value=date(2010, 1, 1), max_value=date(2025, 12, 31))
with col_end:
    end_date = st.date_input("End Date", value=date(2024, 12, 31), min_value=date(2010, 1, 1), max_value=date(2025, 12, 31))

if start_date >= end_date:
    st.sidebar.error("Error: Start Date must precede End Date.")
    st.stop()

# 3. Model Checkpoint Uploader
st.sidebar.markdown("### 🤖 PPO Model Checkpoint")
uploaded_model = st.sidebar.file_uploader(
    "Upload Model (.zip)",
    type=["zip"],
    help="Upload your trained Stable-Baselines3 PPO zip checkpoint.",
)

default_model_paths = [
    Path("models/ppo_trading_best.zip"),
    Path("models/best_model/best_model.zip"),
    Path(f"models/ppo_{ticker.lower()}_final.zip"),
]
default_model_path: Optional[Path] = None
for p in default_model_paths:
    if p.exists():
        default_model_path = p
        break

if uploaded_model is not None:
    st.sidebar.markdown(f"<span class='status-badge badge-blue'>Active: {uploaded_model.name}</span>", unsafe_allow_html=True)
elif default_model_path:
    st.sidebar.markdown(f"<span class='status-badge badge-green'>Default: {default_model_path.as_posix()}</span>", unsafe_allow_html=True)
else:
    st.sidebar.warning("No default model checkpoint found. Please upload a model .zip.")

# 4. Simulation Parameters
with st.sidebar.expander("🛠 Advanced Parameters", expanded=False):
    init_cash = st.number_input("Initial Cash ($)", min_value=1_000.0, max_value=1_000_000.0, value=10_000.0, step=1000.0)
    fee_pct = st.number_input("Transaction Fee (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05)
    fee = fee_pct / 100.0
    window = st.number_input("Price Observation Window", min_value=5, max_value=60, value=10, step=1)

run_button = st.sidebar.button("🚀 Run Backtest", type="primary", width="stretch")


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_market_data(ticker_symbol: str, start_str: str, end_str: str) -> pd.DataFrame:
    """Download and prepare feature dataset with caching."""
    # Download slightly earlier to ensure sufficient warm-up period for 30-day SMA & RSI
    extended_start = (pd.to_datetime(start_str) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    raw = download_data(ticker=ticker_symbol, start=extended_start, end=end_str)
    featured = add_features(raw)
    # Filter strictly to requested date window
    sliced = featured.loc[start_str:end_str]
    return sliced


def load_agent(uploaded_file, default_path: Optional[Path]) -> PPO:
    """Load PPO agent from uploaded file buffer or local disk path."""
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        return PPO.load(tmp_path, device="cpu")

    if default_path and default_path.exists():
        return PPO.load(str(default_path), device="cpu")

    raise FileNotFoundError("No valid PPO model could be loaded.")


# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
st.markdown("<div class='main-title'>Deep RL Trading Bot Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sub-title'>Evaluating PPO Agent vs Buy & Hold Benchmark for <strong>{ticker}</strong> ({start_date} to {end_date})</div>",
    unsafe_allow_html=True,
)

# Run simulation
try:
    with st.spinner("Loading market data and running backtest rollout..."):
        test_df = load_market_data(ticker, str(start_date), str(end_date))

        if len(test_df) <= window + 2:
            st.error(f"Insufficient trading days ({len(test_df)} bars) for window size ({window}). Please select a wider date range.")
            st.stop()

        agent = load_agent(uploaded_model, default_model_path)
        env = run_backtest(agent, test_df, window=window, init_cash=init_cash, fee=fee)

        agent_curve = np.array(env.equity_curve, dtype=np.float64)
        raw_prices = test_df["Close"].values[window - 1 :]
        bh_curve = init_cash * (raw_prices / raw_prices[0])

        n = min(len(agent_curve), len(bh_curve))
        agent_curve = agent_curve[:n]
        bh_curve = bh_curve[:n]
        dates = test_df.index[window - 1 : window - 1 + n]
        close_prices = raw_prices[:n]

        agent_metrics = calculate_metrics(agent_curve)
        bh_metrics = calculate_metrics(bh_curve)

except Exception as e:
    st.error(f"Failed to execute backtest: {str(e)}")
    st.stop()


# ---------------------------------------------------------
# Section 1: KPI Metrics Cards
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    ret_agent = agent_metrics["Total Return %"]
    ret_bh = bh_metrics["Total Return %"]
    st.metric(
        label="RL Agent Total Return",
        value=f"{ret_agent:+.2f}%",
        delta=f"{ret_agent - ret_bh:+.2f}% vs B&H",
    )

with col2:
    st.metric(
        label="Buy & Hold Return",
        value=f"{ret_bh:+.2f}%",
        delta=f"${bh_curve[-1] - init_cash:+,.2f}",
    )

with col3:
    sharpe_agent = agent_metrics["Sharpe"]
    sharpe_bh = bh_metrics["Sharpe"]
    st.metric(
        label="Sharpe Ratio (Ann.)",
        value=f"{sharpe_agent:.2f}",
        delta=f"{sharpe_agent - sharpe_bh:+.2f} vs B&H",
    )

with col4:
    mdd_agent = agent_metrics["Max Drawdown %"]
    mdd_bh = bh_metrics["Max Drawdown %"]
    st.metric(
        label="Max Drawdown",
        value=f"{mdd_agent:.2f}%",
        delta=f"{mdd_agent - mdd_bh:+.2f}%",
        delta_color="inverse",
    )


# ---------------------------------------------------------
# Section 2: Detailed Performance Table
# ---------------------------------------------------------
st.markdown("### 📊 Performance Summary")
metrics_table = pd.DataFrame(
    {
        "Metric": [
            "Final Valuation ($)",
            "Net Profit ($)",
            "Total Return (%)",
            "Sharpe Ratio (Annualized)",
            "Max Drawdown (%)",
            "Total Trades Executed",
        ],
        "RL Agent (PPO)": [
            f"${agent_curve[-1]:,.2f}",
            f"${agent_curve[-1] - init_cash:+,.2f}",
            f"{agent_metrics['Total Return %']:+.2f}%",
            f"{agent_metrics['Sharpe']:.2f}",
            f"{agent_metrics['Max Drawdown %']:.2f}%",
            str(len(env.trades)),
        ],
        "Buy & Hold": [
            f"${bh_curve[-1]:,.2f}",
            f"${bh_curve[-1] - init_cash:+,.2f}",
            f"{bh_metrics['Total Return %']:+.2f}%",
            f"{bh_metrics['Sharpe']:.2f}",
            f"{bh_metrics['Max Drawdown %']:.2f}%",
            "1 (Entry)",
        ],
    }
).set_index("Metric")

st.dataframe(metrics_table, width="stretch")


# ---------------------------------------------------------
# Section 3: Equity Curve & Trade Marker Charts
# ---------------------------------------------------------
st.markdown("### 📈 Visual Strategy Analysis")

tab1, tab2 = st.tabs(["📊 Two-Panel Integrated Chart", "📋 Trade Log Table"])

with tab1:
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    fig.patch.set_facecolor("#0e1117")
    for a in ax:
        a.set_facecolor("#161b26")
        a.tick_params(colors="#cbd5e1")
        a.spines["bottom"].set_color("#334155")
        a.spines["top"].set_color("#334155")
        a.spines["right"].set_color("#334155")
        a.spines["left"].set_color("#334155")
        a.yaxis.label.set_color("#e2e8f0")
        a.xaxis.label.set_color("#e2e8f0")
        a.title.set_color("#f8fafc")
        a.grid(True, color="#334155", alpha=0.3, linestyle="--")

    # Panel 1: Equity Curve
    ax[0].plot(dates, agent_curve, label="RL Agent (PPO)", color="#38bdf8", lw=2.2)
    ax[0].plot(dates, bh_curve, label="Buy & Hold Benchmark", color="#f59e0b", lw=1.8, linestyle="--", alpha=0.85)
    ax[0].set_title(f"{ticker} — Portfolio Equity Trajectory", fontsize=14, fontweight="bold", pad=10)
    ax[0].set_ylabel("Portfolio Value (USD)", fontsize=11)
    ax[0].legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    # Panel 2: Price with BUY / SELL markers
    ax[1].plot(dates, close_prices, color="#94a3b8", lw=1.3, label=f"{ticker} Close Price")

    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []

    for step, side, price in env.trades:
        idx = step - (window - 1)
        if 0 <= idx < n:
            if side == "BUY":
                buy_dates.append(dates[idx])
                buy_prices.append(price)
            elif side == "SELL":
                sell_dates.append(dates[idx])
                sell_prices.append(price)

    if buy_dates:
        ax[1].scatter(buy_dates, buy_prices, marker="^", color="#22c55e", s=90, label="BUY Executed", zorder=6)
    if sell_dates:
        ax[1].scatter(sell_dates, sell_prices, marker="v", color="#ef4444", s=90, label="SELL Executed", zorder=6)

    ax[1].set_title("Close Price with Model Trade Executions", fontsize=13, fontweight="bold", pad=10)
    ax[1].set_ylabel("Price (USD)", fontsize=11)
    ax[1].set_xlabel("Date", fontsize=11)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax[1].legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    if len(env.trades) == 0:
        st.info("No trades were triggered during the selected test window (agent maintained HOLD position).")
    else:
        trades_records = []
        for step, side, price in env.trades:
            idx = step - (window - 1)
            date_val = dates[idx].strftime("%Y-%m-%d") if 0 <= idx < n else "N/A"
            eq_val = f"${agent_curve[idx]:,.2f}" if 0 <= idx < n else "N/A"
            trades_records.append(
                {
                    "Step": step,
                    "Execution Date": date_val,
                    "Action": side,
                    "Execution Price": f"${price:,.2f}",
                    "Portfolio Equity": eq_val,
                }
            )
        trades_df = pd.DataFrame(trades_records)
        st.dataframe(trades_df, width="stretch")
