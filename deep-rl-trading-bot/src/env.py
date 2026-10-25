"""Gymnasium trading environment module.

Implements a single-asset trading environment compatible with Gymnasium and
Stable-Baselines3.
"""

from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.features import add_features


class TradingEnv(gym.Env):
    """Single-asset financial trading environment for reinforcement learning.

    Actions:
        0: HOLD - Maintain current position.
        1: BUY  - Invest 100% of available cash into asset (minus transaction fee).
        2: SELL - Liquidate 100% of asset holdings to cash (minus transaction fee).

    Observation (Dimension: window + 3 + 3 = 16 by default):
        - Last `window` prices normalized by current price: (p / p[-1] - 1.0) * 10.0
        - Technical indicators: `sma_ratio`, `rsi`, `macd_delta`
        - Portfolio state: `cash_fraction`, `position_fraction`, `equity / initial - 1`

    Reward:
        Percentage change in total portfolio equity: (V_next - V_prev) / V_prev
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 10,
        init_cash: float = 10_000.0,
        fee: float = 0.001,
    ) -> None:
        """Initialize TradingEnv.

        Parameters
        ----------
        df : pd.DataFrame
            Historical price and feature DataFrame.
        window : int, default 10
            Number of historical price steps included in the state observation.
        init_cash : float, default 10000.0
            Starting cash balance.
        fee : float, default 0.001
            Proportional transaction fee per trade (0.1% = 0.001).
        """
        super().__init__()

        # Ensure features are present
        required_features = ["sma_ratio", "rsi", "macd_delta"]
        if not all(col in df.columns for col in required_features):
            df = add_features(df)

        if "Close" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'Close' price column.")

        self.df = df.reset_index(drop=True)
        self.window = int(window)
        self.init_cash = float(init_cash)
        self.fee = float(fee)

        if len(self.df) < self.window + 1:
            raise ValueError(
                f"DataFrame length ({len(self.df)}) must be strictly greater than window size ({self.window})."
            )

        self.prices = self.df["Close"].values.astype(np.float64)
        self.feats = self.df[required_features].values.astype(np.float32)

        # Action space: 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = spaces.Discrete(3)

        # Observation space: window + 3 technical indicators + 3 portfolio metrics
        obs_dim = self.window + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Internal state tracking
        self.t: int = self.window - 1
        self.cash: float = self.init_cash
        self.shares: float = 0.0
        self.equity_curve: List[float] = []
        self.trades: List[Tuple[int, str, float]] = []

    def _equity(self) -> float:
        """Calculate total current portfolio valuation (cash + shares * price)."""
        idx = min(self.t, len(self.prices) - 1)
        return float(self.cash + self.shares * self.prices[idx])

    def _obs(self) -> np.ndarray:
        """Construct current observation array."""
        idx = min(self.t, len(self.prices) - 1)

        # Sliced normalized price history (relative to current price, scaled by 10)
        p = self.prices[idx - self.window + 1 : idx + 1]
        p_curr = p[-1] if p[-1] > 1e-8 else 1e-8
        p_norm = (p / p_curr - 1.0) * 10.0

        # Portfolio metrics
        eq = self._equity()
        if eq > 1e-8:
            cash_fraction = self.cash / eq
            position_fraction = (self.shares * self.prices[idx]) / eq
        else:
            cash_fraction = 0.0
            position_fraction = 0.0

        equity_ratio = (eq / self.init_cash) - 1.0
        portfolio_state = np.array(
            [cash_fraction, position_fraction, equity_ratio],
            dtype=np.float32,
        )

        # Concatenate normalized price window, technical features, and portfolio metrics
        features_at_t = self.feats[idx]
        return np.concatenate([p_norm.astype(np.float32), features_at_t, portfolio_state]).astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state at step `window - 1`.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Initial observation and auxiliary info dict.
        """
        super().reset(seed=seed)

        self.t = self.window - 1
        self.cash = float(self.init_cash)
        self.shares = 0.0
        initial_equity = self._equity()
        self.equity_curve = [initial_equity]
        self.trades = []

        return self._obs(), {"equity": initial_equity}

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one trading step in the environment.

        Parameters
        ----------
        action : int
            0 = HOLD, 1 = BUY, 2 = SELL

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
            (observation, reward, terminated, truncated, info)
        """
        idx = min(self.t, len(self.prices) - 1)
        current_price = self.prices[idx]
        v_prev = self._equity()

        # Action Execution:
        # 0: HOLD - do nothing
        # 1: BUY  - invest all cash
        if action == 1 and self.cash > 1e-6:
            cost_factor = 1.0 - self.fee
            self.shares += (self.cash * cost_factor) / current_price
            self.cash = 0.0
            self.trades.append((self.t, "BUY", float(current_price)))

        # 2: SELL - liquidate all shares
        elif action == 2 and self.shares > 1e-9:
            proceeds_factor = 1.0 - self.fee
            self.cash += self.shares * current_price * proceeds_factor
            self.shares = 0.0
            self.trades.append((self.t, "SELL", float(current_price)))

        # Advance step
        self.t += 1
        terminated = self.t >= len(self.prices) - 1
        truncated = False

        # Calculate new equity and reward
        v_next = self._equity()
        denom = v_prev if abs(v_prev) > 1e-8 else 1e-8
        reward = (v_next - v_prev) / denom
        self.equity_curve.append(v_next)

        info = {
            "equity": v_next,
            "trades_count": len(self.trades),
            "step": self.t,
            "cash": self.cash,
            "shares": self.shares,
        }

        return self._obs(), float(reward), terminated, truncated, info
