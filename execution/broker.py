"""
Broker interface and a simulated implementation.

The `Broker` abstract base class is the seam. Everything above it - strategy,
risk governor, engine - is identical whether the broker underneath is a
simulator, a paper account, or a live one. That is the whole reason the seam
exists: you debug the hard parts against the simulator, where a bug costs
nothing.

What ships here:

  SimulatedBroker   Fills market orders against a price you feed it, applying
                    the same fee and slippage model the backtester uses. Tracks
                    cash and positions. No network, no credentials, no account.

  AlpacaPaperBroker A documented stub. Paper trading needs your own API keys,
                    and wiring those is your decision, not mine. The class
                    lists exactly what implementing it requires.

Long-only, market orders only, single-venue. Those limits are deliberate: each
one you remove adds a class of failure (borrow, partial fills at multiple
price levels, cross-venue reconciliation) that deserves its own tests.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

log = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """
    One order. `client_order_id` is the idempotency key.

    A retry after a timeout must reuse the same client_order_id, or you will
    double your position. This is the single most expensive bug in live
    trading and it is trivially avoidable, so the field is required.
    """

    client_order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    status: str = OrderStatus.PENDING.value
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    reject_reason: str = ""
    broker_order_id: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Order":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    @property
    def notional(self) -> float:
        return abs(self.filled_quantity * self.avg_fill_price)

    def describe(self) -> str:
        return (
            f"{self.side} {self.quantity:.8g} {self.symbol} "
            f"[{self.status}]"
            + (f" @ {self.avg_fill_price:,.2f}" if self.filled_quantity else "")
            + (f" ({self.reject_reason})" if self.reject_reason else "")
        )


@dataclass
class Position:
    """
    A holding. `avg_price` is fee-inclusive: buy commissions are folded into
    the cost basis, so realised PnL on the way out is net of both sides.
    """

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realised_pnl: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    def market_value(self, mark: float) -> float:
        return self.quantity * mark

    def unrealised_pnl(self, mark: float) -> float:
        return self.quantity * (mark - self.avg_price)


class Broker(ABC):
    """Minimum surface the engine needs. Implement all of it or none of it."""

    name: str = "abstract"

    @abstractmethod
    def submit(self, order: Order) -> Order:
        """
        Send an order. MUST be idempotent on `client_order_id`: submitting the
        same id twice returns the existing order and does not trade again.
        """

    @abstractmethod
    def cancel(self, client_order_id: str) -> Order:
        ...

    @abstractmethod
    def positions(self) -> dict[str, Position]:
        """Broker's view of holdings. This is the authoritative one."""

    @abstractmethod
    def cash(self) -> float:
        ...

    @abstractmethod
    def orders(self) -> dict[str, Order]:
        ...

    def equity(self, marks: dict[str, float]) -> float:
        total = self.cash()
        for symbol, position in self.positions().items():
            mark = marks.get(symbol)
            if mark is None:
                raise ValueError(f"No mark price supplied for open position in {symbol}")
            total += position.market_value(mark)
        return total


class SimulatedBroker(Broker):
    """
    In-process broker. Fills market orders immediately at the last price you
    gave it, moved against you by `slippage_pct` and charged `fee_pct`.

    It is intentionally strict: it rejects orders it cannot afford and sells it
    cannot cover, rather than quietly going negative. Silent negative cash is
    how a simulator tells you a strategy works when it does not.
    """

    name = "simulated"

    def __init__(
        self,
        starting_cash: float = 10_000.0,
        fee_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        clock=None,
    ) -> None:
        self._cash = float(starting_cash)
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._prices: dict[str, float] = {}
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._seq = 0

    # ------------------------------------------------------------ price feed

    def set_price(self, symbol: str, price: float) -> None:
        if price <= 0:
            raise ValueError(f"Price for {symbol} must be positive, got {price}")
        self._prices[symbol] = float(price)

    def price(self, symbol: str) -> Optional[float]:
        return self._prices.get(symbol)

    # ---------------------------------------------------------------- broker

    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, Position]:
        return {s: p for s, p in self._positions.items() if abs(p.quantity) > 1e-12}

    def orders(self) -> dict[str, Order]:
        return dict(self._orders)

    def cancel(self, client_order_id: str) -> Order:
        order = self._orders.get(client_order_id)
        if order is None:
            raise KeyError(f"Unknown client_order_id {client_order_id!r}")
        if order.status == OrderStatus.PENDING.value:
            order.status = OrderStatus.CANCELLED.value
            order.updated_at = self._clock().isoformat()
        return order

    def submit(self, order: Order) -> Order:
        # Idempotency: a resubmitted id is answered from the ledger, not traded.
        existing = self._orders.get(order.client_order_id)
        if existing is not None:
            log.warning(
                "Duplicate client_order_id %s - returning existing order, not trading again.",
                order.client_order_id,
            )
            return existing

        now = self._clock().isoformat()
        order.created_at = order.created_at or now
        order.updated_at = now
        self._seq += 1
        order.broker_order_id = f"sim-{self._seq:06d}"
        self._orders[order.client_order_id] = order

        price = self._prices.get(order.symbol)
        if price is None:
            return self._reject(order, f"no price available for {order.symbol}")
        if order.quantity <= 0:
            return self._reject(order, f"quantity must be positive, got {order.quantity}")
        if order.order_type != "market":
            return self._reject(order, f"only market orders are simulated, got {order.order_type}")

        if order.side == Side.BUY.value:
            return self._fill_buy(order, price)
        if order.side == Side.SELL.value:
            return self._fill_sell(order, price)
        return self._reject(order, f"unknown side {order.side!r}")

    # --------------------------------------------------------------- filling

    def _reject(self, order: Order, reason: str) -> Order:
        order.status = OrderStatus.REJECTED.value
        order.reject_reason = reason
        order.updated_at = self._clock().isoformat()
        log.warning("Order %s rejected: %s", order.client_order_id, reason)
        return order

    def _fill_buy(self, order: Order, price: float) -> Order:
        fill_price = price * (1.0 + self.slippage_pct)
        cost = order.quantity * fill_price
        fees = cost * self.fee_pct
        if cost + fees > self._cash + 1e-9:
            return self._reject(
                order,
                f"insufficient cash: need {cost + fees:,.2f}, have {self._cash:,.2f}",
            )

        self._cash -= cost + fees
        position = self._positions.setdefault(order.symbol, Position(order.symbol))
        # Fold entry fees into the cost basis so exits report PnL net of both sides.
        total_cost = position.quantity * position.avg_price + cost + fees
        position.quantity += order.quantity
        position.avg_price = total_cost / position.quantity if position.quantity else 0.0

        order.status = OrderStatus.FILLED.value
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.fees = fees
        order.updated_at = self._clock().isoformat()
        return order

    def _fill_sell(self, order: Order, price: float) -> Order:
        position = self._positions.get(order.symbol)
        held = position.quantity if position else 0.0
        if order.quantity > held + 1e-12:
            return self._reject(
                order,
                f"insufficient position: selling {order.quantity:.8g}, hold {held:.8g} "
                f"(shorting is not supported)",
            )

        fill_price = price * (1.0 - self.slippage_pct)
        proceeds = order.quantity * fill_price
        fees = proceeds * self.fee_pct
        self._cash += proceeds - fees

        realised = proceeds - fees - order.quantity * position.avg_price
        position.realised_pnl += realised
        position.quantity -= order.quantity
        if abs(position.quantity) < 1e-12:
            position.quantity = 0.0
            position.avg_price = 0.0

        order.status = OrderStatus.FILLED.value
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.fees = fees
        order.updated_at = self._clock().isoformat()
        order.reason = order.reason or ""
        return order

    # ---------------------------------------------------------- persistence

    def snapshot(self) -> dict[str, Any]:
        return {
            "cash": self._cash,
            "seq": self._seq,
            "positions": {s: p.to_dict() for s, p in self._positions.items()},
            "orders": {k: o.to_dict() for k, o in self._orders.items()},
        }

    def restore(self, data: dict[str, Any]) -> None:
        self._cash = float(data.get("cash", self._cash))
        self._seq = int(data.get("seq", 0))
        self._positions = {
            s: Position.from_dict(p) for s, p in (data.get("positions") or {}).items()
        }
        self._orders = {k: Order.from_dict(o) for k, o in (data.get("orders") or {}).items()}


class AlpacaPaperBroker(Broker):
    """
    NOT IMPLEMENTED. Paper trading against Alpaca requires your own API keys.

    This class exists to pin down the interface, not to be filled in casually.
    To implement it you need:

      1. A paper account and paper keys from https://alpaca.markets. Paper keys
         are separate from live keys. Store them in the environment or a
         secrets manager, never in this repository - `.gitignore` already
         excludes `.env` and `*.key`.

      2. The base URL. Paper and live differ ONLY by hostname:
             paper  https://paper-api.alpaca.markets
             live   https://api.alpaca.markets
         One character of carelessness separates them. Assert on the hostname
         at construction and refuse anything that is not the paper host until
         you have deliberately decided otherwise.

      3. `submit()` must pass `client_order_id` through to Alpaca and handle
         the duplicate-id error as success, returning the existing order. That
         is what makes a retry after a network timeout safe.

      4. `positions()` must return Alpaca's view, not a cached local one. The
         engine reconciles against it and the broker is always right.

      5. Orders are asynchronous. Alpaca accepts an order and fills it later,
         so `submit()` returning is not a fill. You need to poll or stream
         order updates and reflect partial fills, which this project's
         synchronous `Order` model does not currently express.

    Point 5 is the real work, and the reason this is a stub rather than a
    thin wrapper: the engine above it currently assumes immediate fills, which
    is true of the simulator and false of every real broker.
    """

    name = "alpaca-paper"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "AlpacaPaperBroker is not implemented and no credentials are configured "
            "anywhere in this project. Read the class docstring for what implementing "
            "it involves - in particular, real brokers fill asynchronously and the "
            "engine currently assumes immediate fills."
        )

    def submit(self, order: Order) -> Order:  # pragma: no cover
        raise NotImplementedError

    def cancel(self, client_order_id: str) -> Order:  # pragma: no cover
        raise NotImplementedError

    def positions(self) -> dict[str, Position]:  # pragma: no cover
        raise NotImplementedError

    def cash(self) -> float:  # pragma: no cover
        raise NotImplementedError

    def orders(self) -> dict[str, Order]:  # pragma: no cover
        raise NotImplementedError
