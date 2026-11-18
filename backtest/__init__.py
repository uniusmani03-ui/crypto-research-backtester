"""Backtesting harness: costs, sizing, metrics, walk-forward validation."""

from .costs import PRESETS, CostModel
from .harness import (
    BacktestConfig,
    FoldResult,
    SweepResult,
    WalkForwardResult,
    run_strategy,
    sharpe_with_min_trades,
    simulate,
    sweep,
    train_test_split,
    walk_forward,
    walk_forward_folds,
)
from .metrics import Metrics, compute_metrics, degradation_note, format_comparison
from .sizing import RiskModel

__all__ = [
    "CostModel",
    "PRESETS",
    "RiskModel",
    "BacktestConfig",
    "Metrics",
    "compute_metrics",
    "format_comparison",
    "degradation_note",
    "simulate",
    "run_strategy",
    "sweep",
    "SweepResult",
    "train_test_split",
    "walk_forward",
    "walk_forward_folds",
    "WalkForwardResult",
    "FoldResult",
    "sharpe_with_min_trades",
]
