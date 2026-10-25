# Deep RL Trading Bot (`deep-rl-trading-bot`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-008080.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0%2B-blueviolet.svg)](https://stable-baselines3.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)

A modular, production-ready Deep Reinforcement Learning (DRL) algorithmic trading system built with **Stable-Baselines3**, **Gymnasium**, **PyTorch**, and **Streamlit**. The framework trains an autonomous agent using Proximal Policy Optimization (PPO) to make sequential BUY / HOLD / SELL decisions on historical market data, evaluating performance against standard Buy & Hold benchmarks with transaction cost modeling.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [The 5-Phase Pipeline](#-the-5-phase-pipeline)
  - [Phase 1: Data Ingestion](#phase-1-data-ingestion-srcdatapy)
  - [Phase 2: Feature Engineering](#phase-2-feature-engineering-srcfeaturespy)
  - [Phase 3: Gymnasium Trading Environment](#phase-3-gymnasium-trading-environment-srcenvpy)
  - [Phase 4: PPO Agent Training](#phase-4-ppo-agent-training-srctrainpy--colab)
  - [Phase 5: Execution Engine & Backtesting](#phase-5-execution-engine--backtesting-srcbacktestpy--appdashboardpy)
- [Project Structure](#-project-structure)
- [Setup & Quickstart](#-setup--quickstart)
- [How to Train on Google Colab](#-how-to-train-on-google-colab)
- [Testing & Validation](#-testing--validation)
- [Interactive Streamlit Dashboard](#-interactive-streamlit-dashboard)
- [Honest Limitations & Real-World Caveats](#-honest-limitations--real-world-caveats)

---

## 🔍 Overview

Financial time series exhibit non-stationarity, regime changes, and low signal-to-noise ratios. Traditional rule-based strategies often struggle to dynamically adapt to shifting volatility and momentum. This project implements an end-to-end Reinforcement Learning pipeline that:

1. Ingests and cleans split-adjusted historical market price data.
2. Extracts scale-free momentum and trend indicators without lookahead bias.
3. Simulates realistic trading conditions (cash accounting, share positions, and transaction fees).
4. Trains an Actor-Critic PPO policy with generalized advantage estimation and entropy regularization.
5. Performs deterministic out-of-sample backtesting against a Buy & Hold benchmark with visual equity curves and trade execution markers.

---

## 🔄 The 5-Phase Pipeline

```text
┌─────────────────┐     ┌─────────────────────┐     ┌───────────────────────┐
│  Phase 1: Data  │ ──> │   Phase 2: Feature  │ ──> │   Phase 3: Gymnasium  │
│    Ingestion    │     │     Engineering     │     │      Environment      │
└─────────────────┘     └─────────────────────┘     └───────────────────────┘
                                                                │
                                                                ▼
┌─────────────────┐     ┌─────────────────────┐     ┌───────────────────────┐
│ Phase 5: Engine │ <── │    Phase 4: PPO     │ <───┘ (Observation Space &   │
│   & Backtest    │     │   Agent Training    │       Reward Signal)        │
└─────────────────┘     └─────────────────────┘
```

### Phase 1: Data Ingestion (`src/data.py`)
- **Source**: Yahoo Finance via `yfinance` with `auto_adjust=True` (accounts for stock splits and dividend distributions).
- **Multi-Index Normalization**: Automatically flattens multi-tiered column headers returned by modern `yfinance` versions.
- **Data Cleansing**: Retains standard OHLCV fields (`Open`, `High`, `Low`, `Close`, `Volume`), validates date ordering, strips timezone offsets, and eliminates missing/null values.
- **Local Caching**: Caches downloaded datasets to `data/{ticker}.csv` to enable fast, offline experimentation and eliminate redundant network requests.

### Phase 2: Feature Engineering (`src/features.py`)
Computes scale-invariant features designed for neural network stability:
1. **Trend (`sma_ratio`)**:
   $$\text{SMA}_{\text{ratio}} = \frac{\text{SMA}_{10}(\text{Close})}{\text{SMA}_{30}(\text{Close})} - 1.0$$
2. **Wilder's RSI (`rsi`)**:
   Computed using a 14-period exponential moving average ($\alpha = 1/14$) on positive and negative price diffs, scaled strictly to $[0.0, 1.0]$:
   $$\text{RS} = \frac{\text{EMA}_{14}(\text{gain})}{\text{EMA}_{14}(\text{loss}) + 10^{-9}}, \quad \text{RSI} = \frac{100 - \frac{100}{1 + \text{RS}}}{100}$$
3. **Momentum (`macd_delta`)**:
   Scale-free normalized difference between the MACD line ($\text{EMA}_{12} - \text{EMA}_{26}$) and its 9-period signal line:
   $$\text{MACD}_{\text{delta}} = \frac{\text{MACD} - \text{Signal}}{\text{Close}}$$
4. **Chronological Splitting**:
   Partitions the data without temporal leakage:
   - **Train**: Historical records up to **2021-12-31** ($\le 2021$).
   - **Validation**: Out-of-sample tuning set for **2022** ($= 2022$).
   - **Test**: Strictly unseen evaluation period from **2023 onward** ($\ge 2023$).

### Phase 3: Gymnasium Trading Environment (`src/env.py`)
Custom `TradingEnv` adhering strictly to Farama Gymnasium standards and validated via Stable-Baselines3 `check_env`:
- **Action Space**: `spaces.Discrete(3)`
  - `0 = HOLD`: Maintain current allocation.
  - `1 = BUY`: Invest 100% of available cash into asset shares (less 0.1% transaction fee).
  - `2 = SELL`: Liquidate 100% of asset holdings to cash (less 0.1% transaction fee).
- **Observation Space**: `spaces.Box(shape=(16,), dtype=np.float32)`
  - Last 10 normalized price steps: $(p / p_{\text{curr}} - 1.0) \times 10.0$
  - Technical indicators: `sma_ratio`, `rsi`, `macd_delta`
  - Portfolio metrics: `cash_fraction`, `position_fraction`, and `equity / init_cash - 1.0`
- **Reward Function**:
  $$R_t = \frac{V_{t+1} - V_t}{V_t}$$
  Direct percentage change in total portfolio equity between consecutive time steps.
- **Friction**: 0.1% proportional transaction fee per trade ($0.001$).
- **Logging**: Complete chronological history of executed trades (`step`, `action`, `price`).

### Phase 4: PPO Agent Training (`src/train.py` & Colab)
- **Algorithm**: Proximal Policy Optimization (PPO) via Stable-Baselines3.
- **Policy Network**: Multi-Layer Perceptron (`MlpPolicy`) with shared feature extraction layers of `[128, 128]` hidden units.
- **Exploration & Stability**: Entropy regularization coefficient (`ent_coef=0.01`), rollout buffer `n_steps=1024`, minibatch size `128`, learning rate $3 \times 10^{-4}$, and GAE parameter $\lambda = 0.95$.
- **Validation Checkpointing**: Synchronized `EvalCallback` monitoring the 2022 validation split, automatically saving the best checkpoint to `models/best_model/best_model.zip`.

### Phase 5: Execution Engine & Backtesting (`src/backtest.py` & `app/dashboard.py`)
- **Deterministic Rollout**: Runs `model.predict(obs, deterministic=True)` over the unseen test period (2023–2024).
- **Benchmark Comparison**: Computes simultaneous Buy & Hold trajectory from identical starting cash.
- **Quantitative Metrics**:
  - **Total Return (%)**: Overall capital growth.
  - **Sharpe Ratio**: Annualized risk-adjusted return ($\sqrt{252} \times \frac{\mu}{\sigma}$).
  - **Maximum Drawdown (%)**: Worst peak-to-trough equity decline.
- **Visualization**: Two-panel charts rendering portfolio equity trajectories alongside asset prices with color-coded trade markers (Green `▲` for BUY, Red `▼` for SELL).

---

## 📁 Project Structure

```text
deep-rl-trading-bot/
├── app/
│   ├── __init__.py           # Application package marker
│   └── dashboard.py          # Interactive Streamlit dashboard
├── data/
│   └── AAPL.csv              # Cached market OHLCV data
├── models/
│   ├── best_model/
│   │   ├── best_model.zip    # Checkpoint saved by EvalCallback on validation split
│   │   └── evaluations.npz   # Validation metrics over training steps
│   ├── ppo_trading_best.zip  # Pretrained Colab model checkpoint
│   ├── ppo_aapl_final.zip    # Final training weights
│   └── vec_normalize.pkl     # Vectorized environment statistics
├── notebooks/
│   └── deep_rl_trading_colab.ipynb  # Primary Colab training notebook
├── src/
│   ├── __init__.py           # Source package marker
│   ├── data.py               # yfinance data ingestion and caching
│   ├── features.py           # Technical indicators and dataset splitting
│   ├── env.py                # Gymnasium TradingEnv implementation
│   ├── train.py              # CLI training pipeline with PPO and EvalCallback
│   ├── backtest.py           # Strategy backtester, metrics, and plotting
│   └── utils.py              # Common utilities and helper routines
├── tests/
│   ├── __init__.py           # Tests package marker
│   ├── test_data.py          # Tests for data ingestion and caching
│   ├── test_features.py      # Tests for indicator correctness, RSI bounds, and splits
│   ├── test_env.py           # Tests for Gym API compliance and trade mechanics
│   ├── test_train.py         # Tests for training pipeline and CLI args
│   └── test_backtest.py      # Tests for financial metrics and plot generation
├── .gitignore                # Git ignore patterns
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## 🚀 Setup & Quickstart

### 1. Prerequisites
- **Python**: Version `3.10`, `3.11`, or `3.12` installed.
- **Operating System**: Windows, macOS, or Linux.

### 2. Virtual Environment Setup

#### Windows (PowerShell)
```powershell
# Navigate into project directory
cd deep-rl-trading-bot

# Create virtual environment
python -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS / Linux
```bash
cd deep-rl-trading-bot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ☁️ How to Train on Google Colab

The primary, high-performance training path uses Google Colab (GPU or CPU runtime) with the provided self-contained notebook:

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select [`notebooks/deep_rl_trading_colab.ipynb`](notebooks/deep_rl_trading_colab.ipynb).
3. (Optional) In Colab, navigate to **Runtime > Change runtime type** and select **T4 GPU** or **CPU** (MLP policies train efficiently on CPU as well).
4. Run all cells top to bottom (**Runtime > Run all**).
5. The notebook will automatically:
   - Install dependencies (`stable-baselines3`, `gymnasium`, `yfinance`).
   - Download historical OHLCV data.
   - Compute features and split datasets.
   - Train PPO with an `EvalCallback` for 300,000 steps.
   - Generate backtesting metrics and the `backtest.png` plot.
   - Package `ppo_trading_best.zip`, `config.json`, and `test_metrics.csv`.
6. Download the generated `.zip` model and place it into the local `models/` directory:
   ```powershell
   Copy-Item "path/to/downloaded/ppo_trading_best.zip" -Destination "deep-rl-trading-bot/models/"
   ```

---

## 🧪 Testing & Validation

The project includes a comprehensive test suite covering data ingestion, feature edge cases, Gymnasium compliance, training mechanics, and financial metrics.

Run all tests using `pytest`:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected test output:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.5, pytest-9.1.1
collected 14 items

tests/test_data.py::test_download_data_flatten_and_cache PASSED          [  7%]
tests/test_data.py::test_download_data_load_from_cache PASSED            [ 14%]
tests/test_env.py::test_trading_env_check_env PASSED                     [ 21%]
tests/test_env.py::test_random_agent_completes_episode PASSED            [ 28%]
tests/test_env.py::test_action_mechanics_and_fees PASSED                 [ 35%]
tests/test_features.py::test_add_features_no_nans PASSED                 [ 42%]
tests/test_features.py::test_rsi_bounds PASSED                           [ 50%]
tests/test_features.py::test_rsi_extremes PASSED                         [ 57%]
tests/test_features.py::test_split_data_partitions PASSED                [ 64%]
tests/test_train.py::test_parse_args PASSED                              [ 71%]
tests/test_train.py::test_train_pipeline_short_run PASSED                [ 78%]
tests/test_backtest.py::test_calculate_metrics_flat_curve PASSED         [ 85%]
tests/test_backtest.py::test_calculate_metrics_values PASSED             [ 92%]
tests/test_backtest.py::test_plot_backtest_creates_image PASSED          [100%]

============================= 14 passed in ~15s ===============================
```

---

## 📊 Interactive Streamlit Dashboard

Launch the interactive dashboard to evaluate models, inspect trade logs, and visualize performance:

```powershell
streamlit run app/dashboard.py
```

Open **`http://localhost:8501`** in your browser.

### Key Capabilities:
- **Sidebar Controls**: Modify ticker, select date ranges, adjust initial cash, tweak transaction fees, or upload new `.zip` checkpoints.
- **KPI Summary**: Instant inspection of Total Return %, Buy & Hold return, Sharpe Ratio, and Maximum Drawdown.
- **Two-Panel Integrated Chart**:
  - *Top Panel*: Portfolio equity curve comparison.
  - *Bottom Panel*: Price chart with green `▲ BUY` and red `▼ SELL` execution markers.
- **Trade Log Table**: Detailed breakdown of every executed transaction with timestamps, execution prices, and step equity.

---

## ⚠️ Honest Limitations & Real-World Caveats

While Deep Reinforcement Learning offers powerful function approximation for trading, deploying such models in live markets entails significant risks. Key limitations of this implementation include:

### 1. Overfitting & Market Regime Shifts
- **Non-Stationary Distributions**: Financial markets are non-stationary systems driven by macroeconomic policy, geopolitics, and investor psychology. Patterns learned during a prolonged secular bull market (such as 2012–2021 tech equities) often deteriorate rapidly during high-inflation regimes, interest-rate shock cycles, or liquidity crises.
- **Policy Brittleness**: RL agents trained without domain randomization or volatility noise can memorize idiosyncratic historical price trajectories rather than genuine generalizable signals.

### 2. Single-Stock Universe (Zero Diversification)
- **Concentration Risk**: The environment operates strictly on a single instrument (e.g., `AAPL`). It lacks the capacity to hedge systemic risk, allocate across uncorrelated assets, exploit cross-sectional momentum, or rebalance across sectors.
- **Idiosyncratic Shocks**: A single earnings disappointment, regulatory action, or supply chain disruption can induce catastrophic drawdown that cannot be mitigated through single-asset long/cash switching.

### 3. Lack of Slippage & Market Impact
- **Fixed Fee vs. Real Execution**: The simulation accounts for a constant 0.1% commission fee, but completely ignores:
  - **Bid-Ask Spread**: The cost of crossing the spread during volatile periods.
  - **Market Impact**: The price concession required to execute large orders against finite order book liquidity.
  - **Slippage**: Latency between order dispatch and exchange matching, which routinely degrades backtest results in live production.
- **Optimistic Fills**: The environment assumes instantaneous, full execution at the exact historical daily `Close` price, which is rarely achievable in live trading.

### 4. Discrete All-or-Nothing Position Sizing
- **Binary Allocations**: The agent is restricted to three discrete actions: 100% cash, 100% stock, or hold. Professional quantitative strategies rely on continuous, risk-weighted position sizing (e.g., Fractional Kelly, volatility targeting, or mean-variance optimization) to scale exposure based on confidence and prevailing market volatility.

---

## 📜 License

This project is licensed under the MIT License — see the repository for details.
