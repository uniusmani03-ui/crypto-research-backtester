"""
Strategy interface.

A strategy's only job is to turn a price DataFrame into two boolean Series:
`entries` and `exits`. It does not size positions, model costs, or decide when
orders fill — the harness owns all of that, so every strategy is costed and
sized identically and comparisons between them mean something.

Signals returned here are *raw*: computed from the close of bar t. The harness
shifts them forward one bar before simulating, so a signal generated from bar t
is acted on at bar t+1. Do not shift inside a strategy or you will double-shift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

import pandas as pd


class Strategy(ABC):
    """Base class for example strategies."""

    #: Human-readable name, used in reports.
    name: str = "unnamed"

    #: Parameter grid used by the walk-forward optimiser. Keep it small — every
    #: extra combination is another chance to overfit the training window.
    param_grid: dict[str, list[Any]] = {}

    def __init__(self, **params: Any) -> None:
        self.params = {**self.defaults(), **params}

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        """First value of each grid entry, used when no params are supplied."""
        return {k: v[0] for k, v in cls.param_grid.items()}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Return (entries, exits) as boolean Series aligned to `df.index`.

        `df` has columns [open, high, low, close, volume].
        """

    # ------------------------------------------------------------------ utils

    @property
    def warmup(self) -> int:
        """
        Bars of history the strategy needs before its signals are trustworthy.

        The harness uses this to check that every walk-forward fold is long
        enough to be meaningful.
        """
        numeric = [v for v in self.params.values() if isinstance(v, (int, float))]
        return int(max(numeric)) if numeric else 0

    def describe(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({args})"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.describe()}>"

    @classmethod
    def grid(cls) -> Iterator[dict[str, Any]]:
        """Iterate every combination in `param_grid`."""
        import itertools

        if not cls.param_grid:
            yield {}
            return
        keys = list(cls.param_grid)
        for combo in itertools.product(*(cls.param_grid[k] for k in keys)):
            yield dict(zip(keys, combo))
