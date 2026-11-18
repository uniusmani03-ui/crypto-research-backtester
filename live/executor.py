"""
Live order execution — DELIBERATELY NOT IMPLEMENTED.
====================================================

This file exists so that the shape of the boundary is visible: research code
produces signals, and something else, that you write and review, turns signals
into orders. It is a stub. It will stay a stub.

Nothing in this repository holds exchange API keys or broker credentials. The
data layer uses public endpoints only. If you implement this class you are
taking on, at minimum:

  - Key management. Keys with trade permission, never withdrawal permission,
    IP-allowlisted, stored outside the repository, never in a notebook.
  - Order lifecycle. Placement, partial fills, rejections, cancellations,
    reconciliation after a disconnect, and duplicate-order prevention on retry.
  - State reconciliation. Your idea of the position and the exchange's idea of
    the position will diverge. The exchange is right.
  - Kill switches. A hard daily loss limit, a max-position limit, and a manual
    stop that works when your code is wedged.
  - Idempotency. Every order needs a client order id so a retry after a timeout
    cannot double your size.
  - Paper trading first, then the smallest size the venue permits, for weeks.

A backtest that looks good is not evidence that any of the above is handled.
The failure modes of live trading are almost entirely operational, and none of
them appear in a backtest.

If you want help building this, that is a separate piece of work with its own
review — not something to bolt onto a research repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OrderIntent:
    """
    What a strategy wants to happen. Note this is an *intent*, not an order:
    nothing here reaches an exchange.
    """

    symbol: str
    side: str  # "buy" | "sell"
    size: float
    reason: str = ""


class LiveExecutor:
    """Unimplemented on purpose. See the module docstring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Live execution is not implemented in this research environment, and "
            "no credentials are configured anywhere in it. Implementing this is "
            "your decision and your review — read live/executor.py's docstring "
            "for what that entails before you start."
        )

    def submit(self, intent: OrderIntent) -> None:
        raise NotImplementedError("Live execution is not implemented. See module docstring.")

    def cancel_all(self) -> None:
        raise NotImplementedError("Live execution is not implemented. See module docstring.")

    def reconcile(self) -> None:
        raise NotImplementedError("Live execution is not implemented. See module docstring.")
