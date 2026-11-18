"""
EXAMPLE STRATEGY — REFERENCE IMPLEMENTATION ONLY. NOT A RECOMMENDATION.
=======================================================================

RSI mean reversion: buy when RSI drops below an oversold threshold, exit when it
recovers past a neutral/overbought one.

It is included as a deliberate contrast to the trend-following example. Mean
reversion and trend following fail in opposite regimes, and seeing both run
through the same harness on the same data is the fastest way to internalise that
a backtest measures a strategy *and a period*, not a strategy alone.

It is not an edge either. Specific hazards of this shape of strategy:
  - It sells volatility. The equity curve looks smooth and reliable right up
    until a sustained downtrend, where "oversold" keeps getting more oversold
    and the strategy buys the whole way down.
  - Win rate is flattering and largely meaningless here: many small wins, rare
    large losses. Read profit factor and max drawdown instead.
  - The ATR stop in the harness is doing real work for this one. Without it,
    the tail risk is unbounded.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import rsi


class RsiMeanReversion(Strategy):
    name = "rsi_mean_reversion"

    param_grid = {
        "period": [7, 14, 21],
        "oversold": [20, 25, 30],
        "exit_level": [50, 55, 60],
    }

    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        period = int(self.params["period"])
        oversold = float(self.params["oversold"])
        exit_level = float(self.params["exit_level"])

        if exit_level <= oversold:
            empty = pd.Series(False, index=df.index)
            return empty, empty.copy()

        r = rsi(df["close"], period)

        # Explicit bool casts and fill_value shifts: see the note in
        # ema_crossover.py for why shift-then-fillna is a trap.
        below = (r < oversold).fillna(False).astype(bool)
        prev_below = below.shift(1, fill_value=False).astype(bool)
        entries = below & ~prev_below            # first bar of the oversold run

        recovered = (r > exit_level).fillna(False).astype(bool)
        prev_recovered = recovered.shift(1, fill_value=False).astype(bool)
        exits = recovered & ~prev_recovered      # first bar back above the exit level

        valid = r.notna()
        return (entries & valid).fillna(False), (exits & valid).fillna(False)

    @property
    def warmup(self) -> int:
        return int(self.params["period"]) * 3
