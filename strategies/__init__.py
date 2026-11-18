"""
Example strategies.

EVERY STRATEGY IN THIS PACKAGE IS A REFERENCE IMPLEMENTATION, NOT A
RECOMMENDATION. They exist to demonstrate the harness and to give you a
working template to write your own against. None of them is expected to be
profitable after costs. Read the docstring at the top of each file.
"""

from .base import Strategy
from .donchian_breakout import DonchianBreakout
from .ema_crossover import EmaCrossover
from .rsi_mean_reversion import RsiMeanReversion

#: Registry, so scripts can look strategies up by name.
STRATEGIES: dict[str, type[Strategy]] = {
    EmaCrossover.name: EmaCrossover,
    RsiMeanReversion.name: RsiMeanReversion,
    DonchianBreakout.name: DonchianBreakout,
}

__all__ = [
    "Strategy",
    "EmaCrossover",
    "RsiMeanReversion",
    "DonchianBreakout",
    "STRATEGIES",
]
