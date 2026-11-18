"""
EXAMPLE STRATEGY — REFERENCE IMPLEMENTATION ONLY. NOT A RECOMMENDATION.
=======================================================================

EMA crossover is here because it is the "hello world" of systematic trading:
it is short enough to read in one sitting and it exercises every part of the
harness (signals -> shift -> ATR sizing -> costs -> metrics).

It is not an edge. Dual moving-average crossover on liquid crypto has been
public knowledge for decades; whatever trend premium it captured is largely
arbitraged, and after realistic fees and slippage it typically underperforms
buy-and-hold on the same asset. Treat any good-looking backtest of this as a
property of your chosen window, not a discovery.

Known weaknesses, stated so they are not a surprise later:
  - Whipsaws in ranging markets: many small losses, high trade count, fees eat you.
  - Highly sensitive to the fast/slow pair — which is exactly what makes it easy
    to overfit. Run it through walk-forward before believing any parameter.
  - Long-only here. It has no view on downtrends; it just sits in cash.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import ema


class EmaCrossover(Strategy):
    name = "ema_crossover"

    param_grid = {
        "fast": [10, 20, 30, 50],
        "slow": [50, 100, 150, 200],
    }

    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        fast_n = int(self.params["fast"])
        slow_n = int(self.params["slow"])
        if fast_n >= slow_n:
            # Degenerate combination; return no trades rather than nonsense.
            empty = pd.Series(False, index=df.index)
            return empty, empty.copy()

        close = df["close"]
        fast = ema(close, fast_n)
        slow = ema(close, slow_n)

        # Cast explicitly and shift with fill_value rather than shift-then-
        # fillna. Shifting a boolean Series introduces NaN, which promotes the
        # column to object dtype, and `~` on an object column inverts Python
        # bools as integers instead of logically. That produces signals that
        # look plausible and are wrong.
        above = (fast > slow).fillna(False).astype(bool)
        prev_above = above.shift(1, fill_value=False).astype(bool)

        entries = above & ~prev_above   # cross up
        exits = (~above) & prev_above   # cross down

        # Suppress signals in the warm-up region where the EMAs are still NaN.
        valid = fast.notna() & slow.notna()
        return (entries & valid).fillna(False), (exits & valid).fillna(False)

    @property
    def warmup(self) -> int:
        return int(self.params["slow"])
