"""Tests for the PPO training pipeline and CLI arguments."""

import pytest
from src.train import parse_args, train


def test_parse_args(monkeypatch):
    """Test CLI argument parsing with default and custom values."""
    # Test default arguments
    monkeypatch.setattr("sys.argv", ["train.py"])
    args = parse_args()
    assert args.ticker == "AAPL"
    assert args.steps == 10_240
    assert args.seed == 42
    assert args.models_dir == "models"

    # Test custom arguments
    custom_argv = [
        "train.py",
        "--ticker",
        "MSFT",
        "--steps",
        "2048",
        "--seed",
        "100",
        "--models-dir",
        "test_models",
        "--eval-freq",
        "512",
    ]
    monkeypatch.setattr("sys.argv", custom_argv)
    custom_args = parse_args()
    assert custom_args.ticker == "MSFT"
    assert custom_args.steps == 2048
    assert custom_args.seed == 100
    assert custom_args.models_dir == "test_models"
    assert custom_args.eval_freq == 512


def test_train_pipeline_short_run(tmp_path):
    """Test that train() executes an end-to-end CPU run and checkpoints models."""
    models_dir = tmp_path / "models"
    model, best_zip, final_path = train(
        ticker="AAPL",
        steps=1024,
        seed=42,
        models_dir=str(models_dir),
        eval_freq=1024,
        verbose=0,
    )

    assert best_zip.exists(), "Best model checkpoint was not created"
    assert final_path.with_suffix(".zip").exists(), "Final model was not saved"
    assert (models_dir / "vec_normalize.pkl").exists(), "VecNormalize statistics were not saved"
