"""
Execution engine: the thing that turns a decision into an order, safely.

Responsibilities, in the order they matter:

  1. NOTHING REACHES A BROKER WITHOUT PASSING THE RISK GOVERNOR. The gate is
     inside `place()`. There is no bypass argument and no "just this once"
     path, because that is what those always become.

  2. IDEMPOTENCY. Every order carries a deterministic `client_order_id`
     derived from what the order *is* - symbol, side, quantity, and the bar
     that triggered it. Replay the same bar and you get the same id, which the
     broker answers from its ledger instead of trading again. This is what
     makes a crash-and-restart safe.

  3. RECONCILIATION. On every sync, the engine compares its own view of
     positions with the broker's. If they diverge it adopts the broker's
     numbers and halts. The broker is always right; your process is the thing
     that crashed.

  4. PERSISTENCE. Engine state is written atomically after every change, so a
     restart resumes rather than starting from an imaginary flat book.

Deliberate limitation: this assumes orders fill immediately and completely.
That is true of `SimulatedBroker` and false of every real broker. Wiring a
real one means modelling asynchronous and partial fills first - see the
`AlpacaPaperBroker` docstring.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .atomic import atomic_write_text
from .broker import Broker, Order, OrderStatus, Position, Side
from .risk import RiskDecision, RiskGovernor

log = logging.getLogger(__name__)

#: Positions closer than this are treated as equal during reconciliation.
POSITION_TOLERANCE = 1e-8


def make_client_order_id(symbol: str, side: str, quantity: float, stamp: str) -> str:
    """
    Deterministic idempotency key.

    Same symbol, side, size and triggering bar produce the same id, so a retry
    or a replay cannot open a second position. Keep this a pure function of
    its arguments - adding a timestamp or a random suffix defeats the purpose.
    """
    raw = f"{symbol}|{side}|{quantity:.12g}|{stamp}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{side}-{digest}"


@dataclass
class EngineState:
    """The engine's own view, mirrored against the broker on every sync."""

    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    submitted_keys: list[str] = field(default_factory=list)
    last_sync: str = ""
    broker_snapshot: dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    """
    Wires strategy intent -> risk governor -> broker, with persistence.

        engine = ExecutionEngine(broker, governor, state_dir="state")
        engine.mark("BTC/USDT", 42_000.0)
        engine.sync()
        engine.target_exposure("BTC/USDT", 0.20, reason="ema cross up", stamp=bar_time)
    """

    def __init__(
        self,
        broker: Broker,
        governor: RiskGovernor,
        state_dir: Path | str = "state",
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.broker = broker
        self.governor = governor
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "engine_state.json"
        self.clock = clock
        self.marks: dict[str, float] = {}
        self._last_saved = ""
        self._last_block_reason = ""
        self._blocked_count = 0
        self.state = self._load()
        self._restore_broker()

    # ---------------------------------------------------------- persistence

    def _load(self) -> EngineState:
        if not self.state_path.exists():
            return EngineState()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return EngineState(
                positions=data.get("positions", {}),
                submitted_keys=data.get("submitted_keys", []),
                last_sync=data.get("last_sync", ""),
                broker_snapshot=data.get("broker_snapshot", {}),
            )
        except Exception as exc:
            # Fail closed, exactly as the governor does. An engine that cannot
            # read its own state does not know what it holds.
            log.error("Unreadable engine state at %s (%s). Halting.", self.state_path, exc)
            self.governor.halt(f"engine state unreadable ({type(exc).__name__})")
            return EngineState()

    def _save(self) -> None:
        """
        Persist atomically, skipping writes that would change nothing.

        `last_sync` is excluded from the comparison on purpose - it ticks every
        bar, and including it would defeat the whole point of the check.
        Losing the last few seconds of a timestamp costs nothing; the positions
        and the submitted-order keys are what must survive.

        A persistence failure halts rather than raising. A bot that cannot
        record what it did must stop trading, but crashing the process loses
        the session for no benefit.
        """
        payload = {
            "positions": self.state.positions,
            "submitted_keys": self.state.submitted_keys[-5000:],
            "broker_snapshot": (
                self.broker.snapshot() if hasattr(self.broker, "snapshot") else {}
            ),
        }
        serialised = json.dumps(payload, indent=2, sort_keys=True, default=str)
        if serialised == self._last_saved:
            return

        payload["last_sync"] = self.state.last_sync
        try:
            atomic_write_text(
                self.state_path, json.dumps(payload, indent=2, sort_keys=True, default=str)
            )
            self._last_saved = serialised
        except Exception as exc:
            log.error("Could not persist engine state (%s). Halting.", exc)
            self.governor.halt(f"could not persist engine state ({type(exc).__name__})")

    def _restore_broker(self) -> None:
        """Simulated brokers keep their book in our state file; real ones do not."""
        if self.state.broker_snapshot and hasattr(self.broker, "restore"):
            try:
                self.broker.restore(self.state.broker_snapshot)
                log.info("Restored simulated broker state from %s", self.state_path)
            except Exception as exc:
                log.error("Could not restore broker state (%s). Halting.", exc)
                self.governor.halt(f"broker state restore failed: {exc}")

    # ---------------------------------------------------------------- marks

    def mark(self, symbol: str, price: float) -> None:
        """Record the current price. Feeds equity, sizing, and the simulator."""
        if price <= 0:
            raise ValueError(f"mark price for {symbol} must be positive, got {price}")
        self.marks[symbol] = float(price)
        if hasattr(self.broker, "set_price"):
            self.broker.set_price(symbol, price)

    def equity(self) -> float:
        return self.broker.equity(self.marks)

    def exposure(self) -> float:
        return sum(
            abs(p.market_value(self.marks[s]))
            for s, p in self.broker.positions().items()
            if s in self.marks
        )

    # --------------------------------------------------------------- syncing

    def sync(self) -> None:
        """
        Refresh equity into the governor and reconcile against the broker.

        Call this before any decision. It is cheap and it is the only thing
        standing between you and trading on a stale view of your own account.
        """
        self.reconcile()
        self.governor.update_equity(self.equity(), now=self.clock())
        self.state.last_sync = self.clock().isoformat()
        self._save()

    def reconcile(self) -> bool:
        """
        Compare our view with the broker's. Returns True if they agree.

        On divergence the broker wins and the governor halts. Trading on after
        discovering you do not know your own position is how a small bug
        becomes a large loss.
        """
        broker_positions = self.broker.positions()
        ours = self.state.positions
        divergences: list[str] = []

        symbols = set(broker_positions) | set(ours)
        for symbol in sorted(symbols):
            theirs_qty = broker_positions[symbol].quantity if symbol in broker_positions else 0.0
            ours_qty = float(ours.get(symbol, {}).get("quantity", 0.0))
            if abs(theirs_qty - ours_qty) > POSITION_TOLERANCE:
                divergences.append(
                    f"{symbol}: broker {theirs_qty:.8g}, engine {ours_qty:.8g}"
                )

        # Adopt the broker's view regardless.
        self.state.positions = {s: p.to_dict() for s, p in broker_positions.items()}

        if divergences:
            reason = "position reconciliation failed - " + "; ".join(divergences)
            log.error("%s", reason)
            self.governor.halt(reason, now=self.clock())
            self._save()
            return False
        return True

    # -------------------------------------------------------------- ordering

    def place(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stamp: str,
        reason: str = "",
    ) -> tuple[Optional[Order], RiskDecision]:
        """
        Risk-check and send one order.

        Returns (order, decision). `order` is None when the risk governor
        blocked it - the decision explains why.
        """
        price = self.marks.get(symbol)
        if price is None:
            raise ValueError(f"No mark for {symbol}; call mark() before placing orders.")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")

        notional = quantity * price

        # Exits are risk-reducing. Blocking them because exposure is too high
        # would trap you in exactly the position you are trying to leave, so
        # sells are checked with zero incremental notional.
        is_exit = side == Side.SELL.value
        decision = self.governor.check_order(
            notional=0.0 if is_exit else notional,
            current_exposure=self.exposure(),
            now=self.clock(),
        )
        if not decision.allowed and not is_exit:
            # Rate-limited: a latched halt would otherwise print an identical
            # warning on every bar for the rest of the run, burying anything
            # useful underneath it.
            self._blocked_count += 1
            reason = decision.describe()
            if reason != self._last_block_reason:
                log.warning("Order blocked for %s: %s", symbol, reason)
                self._last_block_reason = reason
            elif self._blocked_count % 100 == 0:
                log.warning(
                    "Still blocked after %d attempts: %s", self._blocked_count, reason
                )
            return None, decision
        if not decision.allowed and is_exit:
            # A halt must not prevent flattening, but it must be visible.
            log.warning(
                "Risk governor is blocking (%s) but this is an exit; allowing it through.",
                decision.describe(),
            )

        key = make_client_order_id(symbol, side, quantity, stamp)
        if key in self.state.submitted_keys:
            log.info("Order %s already submitted (idempotent no-op).", key)
            return self.broker.orders().get(key), decision

        # Capture the cost basis BEFORE the fill. A fully closed position
        # vanishes from positions(), so reading realised PnL back afterwards
        # is not possible for exactly the trades that matter most.
        before = self.broker.positions().get(symbol)
        entry_basis = before.avg_price if before else 0.0

        order = Order(
            client_order_id=key,
            symbol=symbol,
            side=side,
            quantity=quantity,
            reason=reason,
            created_at=self.clock().isoformat(),
        )
        order = self.broker.submit(order)

        self.state.submitted_keys.append(key)

        if order.status == OrderStatus.FILLED.value:
            if is_exit:
                # Realised PnL is derived from the fill itself rather than read
                # back off the broker's position record. Two reasons: a fully
                # closed position no longer appears in positions(), and not
                # every broker reports realised PnL at all. The entry basis is
                # fee-inclusive, so subtracting exit fees nets both sides.
                realised = (
                    order.filled_quantity * (order.avg_fill_price - entry_basis) - order.fees
                )
                self.governor.record_trade(pnl=realised, now=self.clock())
                log.info(
                    "Filled %s (%s) realised %.2f",
                    order.describe(), reason or "no reason given", realised,
                )
            else:
                log.info("Filled %s (%s)", order.describe(), reason or "no reason given")
        else:
            log.warning("Order not filled: %s", order.describe())

        self.state.positions = {s: p.to_dict() for s, p in self.broker.positions().items()}
        self._save()
        return order, decision

    def target_exposure(
        self,
        symbol: str,
        target_fraction: float,
        stamp: str,
        reason: str = "",
    ) -> tuple[Optional[Order], RiskDecision]:
        """
        Move toward holding `target_fraction` of equity in `symbol`.

        0.0 flattens. This is the bridge from a strategy's allocation decision
        to an actual order: it computes the delta and places only that.
        """
        if not 0.0 <= target_fraction <= 1.0:
            raise ValueError(f"target_fraction must be in [0, 1], got {target_fraction}")

        price = self.marks.get(symbol)
        if price is None:
            raise ValueError(f"No mark for {symbol}; call mark() first.")

        equity = self.equity()
        current = self.broker.positions().get(symbol)
        current_qty = current.quantity if current else 0.0
        target_qty = (target_fraction * equity) / price
        delta = target_qty - current_qty

        # Ignore rebalances too small to be worth the fees.
        if abs(delta) * price < max(1.0, 0.001 * equity):
            return None, self.governor.check_order(
                notional=0.0, current_exposure=self.exposure(), now=self.clock()
            )

        side = Side.BUY.value if delta > 0 else Side.SELL.value
        return self.place(symbol, side, abs(delta), stamp=stamp, reason=reason)

    # -------------------------------------------------------------- display

    def report(self) -> str:
        lines = [
            f"engine [broker={self.broker.name}]",
            f"  cash {self.broker.cash():,.2f}   equity {self.equity():,.2f}",
        ]
        positions = self.broker.positions()
        if positions:
            for symbol, p in sorted(positions.items()):
                mark = self.marks.get(symbol, p.avg_price)
                lines.append(
                    f"  {symbol}: {p.quantity:.8g} @ {p.avg_price:,.2f} "
                    f"(mark {mark:,.2f}, unrealised {p.unrealised_pnl(mark):,.2f}, "
                    f"realised {p.realised_pnl:,.2f})"
                )
        else:
            lines.append("  flat")
        lines.append(f"  orders submitted: {len(self.state.submitted_keys)}")
        return "\n".join(lines)
