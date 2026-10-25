"""Tests for TradingEnv gymnasium trading environment."""

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from src.env import TradingEnv
from src.features import add_features


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Generate a clean feature dataframe for testing the environment."""
    dates = pd.date_range("2021-01-01", periods=120, freq="B")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, size=120)
    prices = 100.0 * np.exp(np.cumsum(returns))

    raw_df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.02,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 5_000_000, size=120),
        },
        index=dates,
    )
    return add_features(raw_df)


def test_trading_env_check_env(sample_feature_df):
    """Verify that TradingEnv adheres strictly to the Gymnasium API using SB3 check_env."""
    env = TradingEnv(sample_feature_df, window=10, init_cash=10_000.0, fee=0.001)
    # check_env raises AssertionError or Warning if the environment violates Gym standards
    check_env(env)


def test_random_agent_completes_episode(sample_feature_df):
    """Verify that a random agent completes a full episode without errors."""
    window = 10
    init_cash = 10_000.0
    fee = 0.001

    env = TradingEnv(sample_feature_df, window=window, init_cash=init_cash, fee=fee)
    obs, info = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert "equity" in info
    assert info["equity"] == pytest.approx(init_cash)

    steps_taken = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Invariance assertions per step
        assert env.observation_space.contains(obs), f"Observation at step {steps_taken} out of bounds"
        assert isinstance(reward, float)
        assert np.isfinite(reward), "Reward is non-finite (NaN or Inf)"
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "equity" in info
        assert np.isfinite(info["equity"])

        steps_taken += 1

    # Ensure the episode finished by reaching the end of the price data
    assert terminated is True
    expected_steps = len(sample_feature_df) - window
    assert steps_taken == expected_steps

    # Check equity curve tracking
    assert len(env.equity_curve) == steps_taken + 1
    assert all(np.isfinite(eq) for eq in env.equity_curve)

    # Check trade logging
    for trade in env.trades:
        step, side, price = trade
        assert isinstance(step, int)
        assert side in {"BUY", "SELL"}
        assert price > 0.0


def test_action_mechanics_and_fees(sample_feature_df):
    """Test trade execution logic, transaction fee application, and state updates."""
    window = 10
    init_cash = 10_000.0
    fee = 0.001  # 0.1%

    env = TradingEnv(sample_feature_df, window=window, init_cash=init_cash, fee=fee)
    env.reset(seed=42)

    p0 = env.prices[env.t]

    # 1. Action = BUY (1)
    obs, r, term, trunc, info = env.step(1)
    expected_shares = (init_cash * (1.0 - fee)) / p0
    assert env.cash == 0.0
    assert env.shares == pytest.approx(expected_shares)
    assert len(env.trades) == 1
    assert env.trades[0] == (window - 1, "BUY", p0)

    # 2. Action = HOLD (0)
    shares_before = env.shares
    p1 = env.prices[env.t]
    obs, r, term, trunc, info = env.step(0)
    assert env.cash == 0.0
    assert env.shares == shares_before
    assert len(env.trades) == 1

    # 3. Action = SELL (2)
    p2 = env.prices[env.t]
    obs, r, term, trunc, info = env.step(2)
    expected_cash = shares_before * p2 * (1.0 - fee)
    assert env.shares == 0.0
    assert env.cash == pytest.approx(expected_cash)
    assert len(env.trades) == 2
    assert env.trades[1] == (window + 1, "SELL", p2)
