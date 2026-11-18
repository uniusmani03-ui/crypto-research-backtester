"""
EXAMPLE STRATEGY — REFERENCE IMPLEMENTATION ONLY. NOT A RECOMMENDATION.
=======================================================================

Donchian channel breakout: go long on a new N-bar high, exit on a new M-bar low.
This is the skeleton of the original Turtle system.

Included as a third reference because it has a different trade profile again:
low win rate, a handful of large winners carrying everything. That shape breaks
people's intuitions about what a "good" strategy looks like, and it is a useful
counterexample when reading the metrics table.

Not an edge. Notes:
  - Breakout systems are the most sensitive of the three to slippage: by
    definition you are buying at the moment price is moving fastest against you.
    If your slippage assumption is optimistic, this strategy is where you will
    be most wrong.
  - Expect win rates in the 30-40% range. That is not a bug.
  - Trade counts are low, so the "statistically meaningful?" flag in the metrics
    report will often fire. Believe the flag.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import rolling_max, rolling_min


class DonchianBreakout(Strategy):
    name = "donchian_breakout"

    param_grid = {
        "entry_window": [20, 40, 55],
        "exit_window": [10, 20],
    }

    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        entry_n = int(self.params["entry_window"])
        exit_n = int(self.params["exit_window"])

        close = df["close"]
        # Channel excludes the current bar, otherwise every new high trivially
        # equals the rolling max and the signal is degenerate.
        upper = rolling_max(df["high"], entry_n).shift(1)
        lower = rolling_min(df["low"], exit_n).shift(1)

        # These are level conditions, not edges: price can sit above the
        # channel for many bars. That is intentional for a breakout system -
        # the harness and the paper runner both treat repeated signals as
        # "stay long", not "buy again".
        entries = (close > upper).fillna(False).astype(bool)
        exits = (close < lower).fillna(False).astype(bool)

        valid = upper.notna() & lower.notna()
        return (entries & valid).fillna(False), (exits & valid).fillna(False)

    @property
    def warmup(self) -> int:
        return max(int(self.params["entry_window"]), int(self.params["exit_window"])) + 1
