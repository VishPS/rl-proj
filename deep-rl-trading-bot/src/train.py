"""PPO Agent training script.

Provides a lightweight, CPU-friendly training pipeline using Stable-Baselines3 PPO.
Uses an EvalCallback on the validation dataset to automatically checkpoint the best model.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data import download_data
from src.env import TradingEnv
from src.features import add_features, split_data


def train(
    ticker: str = "AAPL",
    steps: int = 10_240,
    seed: int = 42,
    start_date: str = "2012-01-01",
    end_date: str = "2024-12-31",
    models_dir: str = "models",
    eval_freq: int = 1024,
    verbose: int = 1,
) -> Tuple[PPO, Path, Path]:
    """Train PPO agent on historical stock data and checkpoint best validation model.

    Parameters
    ----------
    ticker : str, default "AAPL"
        Stock ticker symbol.
    steps : int, default 10240
        Total environment interaction steps for training.
    seed : int, default 42
        Random seed for reproducibility.
    start_date : str, default "2012-01-01"
        Start date for historical data download.
    end_date : str, default "2024-12-31"
        End date for historical data download.
    models_dir : str, default "models"
        Directory where best and final models are saved.
    eval_freq : int, default 1024
        Evaluate validation performance every `eval_freq` steps.
    verbose : int, default 1
        Verbosity level (0 = silent, 1 = info).

    Returns
    -------
    Tuple[PPO, Path, Path]
        (trained_model, best_model_path, final_model_path)
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[Train] Downloading/loading data for {ticker} ({start_date} to {end_date})...")
    raw_df = download_data(ticker=ticker, start=start_date, end=end_date)
    feat_df = add_features(raw_df)
    train_df, val_df, _ = split_data(feat_df)

    if verbose:
        print(f"[Train] Data splits - Train: {len(train_df)} bars, Val: {len(val_df)} bars")

    # Vectorized training environment with normalized rewards
    train_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(TradingEnv(train_df))]),
        norm_obs=False,
        norm_reward=True,
    )

    # Validation environment for deterministic evaluation synchronized with VecNormalize wrapper
    val_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(TradingEnv(val_df))]),
        training=False,
        norm_obs=False,
        norm_reward=False,
    )

    # EvalCallback checkpoints the best performing policy on the validation split
    best_model_dir = models_path / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(best_model_dir),
        eval_freq=max(1, eval_freq),
        n_eval_episodes=1,
        deterministic=True,
        verbose=verbose,
    )

    if verbose:
        print(f"[Train] Initializing PPO agent (MlpPolicy, net_arch=[128, 128], ent_coef=0.01, seed={seed})...")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=seed,
        device="cpu",
        verbose=verbose,
    )

    if verbose:
        print(f"[Train] Starting learning loop for {steps} steps on CPU...")
    model.learn(total_timesteps=steps, callback=eval_callback)

    final_model_path = models_path / f"ppo_{ticker.lower()}_final"
    model.save(str(final_model_path))

    # Save normalization statistics
    vec_norm_path = models_path / "vec_normalize.pkl"
    train_env.save(str(vec_norm_path))

    best_model_zip = best_model_dir / "best_model.zip"
    if verbose:
        print(f"[Train] Training complete!")
        print(f"[Train] Best model saved to: {best_model_zip}")
        print(f"[Train] Final model saved to: {final_model_path}.zip")

    return model, best_model_zip, final_model_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train PPO Trading Agent")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--steps", type=int, default=10_240, help="Total training timesteps (default: 10240)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for model checkpoints")
    parser.add_argument("--eval-freq", type=int, default=1024, help="Evaluation frequency in timesteps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        ticker=args.ticker,
        steps=args.steps,
        seed=args.seed,
        models_dir=args.models_dir,
        eval_freq=args.eval_freq,
    )
