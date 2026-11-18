"""
Trading cost model.

The single most common way a beginner's backtest lies to them is by assuming
trades are free and fill at the printed price. They are not, and they do not.
Every simulation in this project is costed; there is no zero-cost code path.

Two components:

  fee_pct       Exchange commission, per side, as a fraction of notional.
                Defaults to 10 bps (0.10%), roughly Binance spot taker without
                fee discounts. Check your own venue and tier — this varies by
                an order of magnitude between retail spot and institutional.

  slippage_pct  Everything else that moves your fill away from the reference
                price: half the bid/ask spread, plus market impact, plus the
                fact that on a signal bar you are usually trading in the same
                direction as everyone else who saw the same thing.

Slippage is the parameter people get most wrong. 5 bps on hourly BTC/USDT with
retail size is a defensible starting point. On an illiquid altcoin, in a fast
market, on a breakout signal, or with size, it can be 10x that. The honest test
is not "what does my strategy earn at my favourite assumption" but "at what
cost level does the edge disappear" — see `CostModel.breakeven_scan` usage in
the example script.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Per-side trading costs, as fractions of notional (0.001 == 10 bps)."""

    fee_pct: float = 0.001
    slippage_pct: float = 0.0005

    def __post_init__(self) -> None:
        if self.fee_pct < 0 or self.slippage_pct < 0:
            raise ValueError("Costs cannot be negative.")
        if self.fee_pct == 0 and self.slippage_pct == 0:
            # Allowed, but only ever as a deliberate diagnostic.
            import warnings

            warnings.warn(
                "CostModel with zero fees AND zero slippage. This is a fantasy "
                "equity curve — useful only to measure how much cost is eating.",
                stacklevel=2,
            )

    @property
    def round_trip_pct(self) -> float:
        """Total cost of getting in and out once, as a fraction of notional."""
        return 2.0 * (self.fee_pct + self.slippage_pct)

    @property
    def round_trip_bps(self) -> float:
        return self.round_trip_pct * 10_000

    def scaled(self, factor: float) -> "CostModel":
        """A copy with both components multiplied — for cost-sensitivity scans."""
        return CostModel(fee_pct=self.fee_pct * factor, slippage_pct=self.slippage_pct * factor)

    def describe(self) -> str:
        return (
            f"fees {self.fee_pct * 1e4:.1f}bps/side, "
            f"slippage {self.slippage_pct * 1e4:.1f}bps/side "
            f"({self.round_trip_bps:.1f}bps round trip)"
        )


#: A few named presets. These are starting points, not truth. Verify against
#: your own venue's fee schedule and your own fill data before trusting them.
PRESETS = {
    # Retail crypto spot, taker, no fee discounts.
    "crypto_retail": CostModel(fee_pct=0.0010, slippage_pct=0.0005),
    # Higher-tier / maker-ish crypto spot.
    "crypto_low": CostModel(fee_pct=0.0004, slippage_pct=0.0003),
    # Deliberately harsh: illiquid pairs, fast markets, or size.
    "crypto_harsh": CostModel(fee_pct=0.0015, slippage_pct=0.0025),
    # Retail spot FX via a spread-only broker (commission folded into spread).
    "fx_retail": CostModel(fee_pct=0.0000, slippage_pct=0.0004),
}
