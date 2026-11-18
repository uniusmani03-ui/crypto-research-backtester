"""
ATR-based fixed-fractional position sizing.

The idea, in one sentence: risk the same *fraction of equity* on every trade by
making position size inversely proportional to the instrument's volatility.

Mechanically:

    stop distance  = atr_mult * ATR(atr_period)          (in price units)
    risk budget    = equity * risk_per_trade             (in cash)
    position size  = risk budget / stop distance         (in units)

Expressed as a fraction of equity, which is what vectorbt wants:

    fraction = risk_per_trade * price / (atr_mult * ATR)

So a 1% risk budget with a 2xATR stop on an instrument whose ATR is 2% of price
gives a position of 1% * 1 / (2 * 0.02) = 25% of equity. If volatility doubles,
the position halves, and the cash at risk stays at 1%.

Two things worth being explicit about:

1. The stop is not optional. The whole risk calculation assumes you exit at
   `atr_mult * ATR` against you. The harness therefore passes a matching
   `sl_stop` to vectorbt. Sizing by ATR without the corresponding stop is not
   a risk model, it is just a volatility-scaled position.

2. Real losses exceed the stop. Gaps, illiquidity, and stop slippage all mean
   "risk 1%" is really "risk about 1%, occasionally more". Fixed-fractional
   sizing bounds your *typical* loss, not your worst one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.indicators import atr as _atr


@dataclass(frozen=True)
class RiskModel:
    """Fixed-fractional risk-per-trade sizing driven by ATR."""

    #: Fraction of equity risked per trade, assuming the stop is hit. 0.01 == 1%.
    risk_per_trade: float = 0.01
    #: ATR lookback.
    atr_period: int = 14
    #: Stop distance in ATR multiples.
    atr_mult: float = 2.0
    #: Cap on position size as a fraction of equity. 1.0 == no leverage.
    max_position_frac: float = 1.0
    #: Floor, to avoid dust-sized positions when volatility spikes.
    min_position_frac: float = 0.01

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade < 1:
            raise ValueError("risk_per_trade must be in (0, 1).")
        if self.atr_mult <= 0:
            raise ValueError("atr_mult must be positive.")
        if not 0 < self.max_position_frac <= 1.0:
            raise ValueError(
                "max_position_frac must be in (0, 1]. Leverage is out of scope "
                "for this research harness."
            )

    def atr(self, df: pd.DataFrame) -> pd.Series:
        return _atr(df["high"], df["low"], df["close"], self.atr_period)

    def stop_distance_frac(self, df: pd.DataFrame) -> pd.Series:
        """Stop distance as a fraction of price — this is vectorbt's `sl_stop`."""
        a = self.atr(df)
        frac = (self.atr_mult * a) / df["close"]
        return frac.replace([np.inf, -np.inf], np.nan)

    def position_frac(self, df: pd.DataFrame) -> pd.Series:
        """
        Position size as a fraction of available equity, per bar.

        Because this is a fraction rather than a fixed unit count, the risk
        budget compounds with the account: as equity grows, so does the cash
        risked per trade, which is what "fixed fractional" is supposed to mean.
        """
        stop_frac = self.stop_distance_frac(df)
        frac = self.risk_per_trade / stop_frac
        frac = frac.clip(lower=self.min_position_frac, upper=self.max_position_frac)
        # Warm-up bars have no ATR yet; there should be no signals there anyway.
        return frac.fillna(self.min_position_frac)

    def describe(self) -> str:
        return (
            f"risk {self.risk_per_trade:.2%}/trade, "
            f"stop {self.atr_mult:g}xATR({self.atr_period}), "
            f"max {self.max_position_frac:.0%} of equity"
        )
