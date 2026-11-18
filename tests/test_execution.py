"""
Tests for the simulated broker and the execution engine.

    python tests\test_execution.py
    pytest tests\test_execution.py

No network, no credentials, no real state directory.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution import (  # noqa: E402
    ExecutionEngine,
    OrderStatus,
    RiskGovernor,
    RiskLimits,
    Side,
    SimulatedBroker,
    make_client_order_id,
)

T0 = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
SYMBOL = "BTC/USDT"


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


def build(tmp: str, clock: FakeClock, cash: float = 10_000.0, **limits):
    defaults = dict(max_position_frac=0.50, max_trades_per_day=50, max_consecutive_losses=50)
    defaults.update(limits)
    broker = SimulatedBroker(starting_cash=cash, fee_pct=0.001, slippage_pct=0.0005, clock=clock)
    governor = RiskGovernor(RiskLimits(**defaults), state_dir=Path(tmp), clock=clock)
    engine = ExecutionEngine(broker, governor, state_dir=Path(tmp), clock=clock)
    return broker, governor, engine


# ------------------------------------------------------------ broker tests


def test_buy_updates_cash_and_position():
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        broker, _, engine = build(tmp, clock)
        engine.mark(SYMBOL, 100.0)
        engine.sync()

        order, decision = engine.place(SYMBOL, Side.BUY.value, 10.0, stamp="bar-1")
        assert order is not None and order.status == OrderStatus.FILLED.value, decision.describe()

        # Filled above the mark because of slippage, plus fees.
        assert abs(order.avg_fill_price - 100.05) < 1e-9
        expected_cost = 10.0 * 100.05
        expected_fees = expected_cost * 0.001
        assert abs(broker.cash() - (10_000.0 - expected_cost - expected_fees)) < 1e-6

        position = broker.positions()[SYMBOL]
        assert abs(position.quantity - 10.0) < 1e-12
        # Cost basis includes entry fees.
        assert position.avg_price > 100.05


def test_insufficient_cash_rejected():
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        # Exposure cap wide open, so the governor passes it and the BROKER is
        # the thing that refuses: notional is exactly equity, but slippage and
        # fees push the actual cost above available cash.
        broker, _, engine = build(tmp, clock, cash=500.0, max_position_frac=1.0)
        engine.mark(SYMBOL, 100.0)
        engine.sync()

        order, decision = engine.place(SYMBOL, Side.BUY.value, 5.0, stamp="bar-1")
        assert order is not None, decision.describe()
        assert order.status == OrderStatus.REJECTED.value
        assert "insufficient cash" in order.reject_reason
        assert broker.cash() == 500.0
        assert not broker.positions()


def test_oversized_sell_rejected():
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        broker, _, engine = build(tmp, clock)
        engine.mark(SYMBOL, 100.0)
        engine.sync()
        engine.place(SYMBOL, Side.BUY.value, 5.0, stamp="bar-1")

        order, _ = engine.place(SYMBOL, Side.SELL.value, 50.0, stamp="bar-2")
        assert order.status == OrderStatus.REJECTED.value
        assert "insufficient position" in order.reject_reason
        assert abs(broker.positions()[SYMBOL].quantity - 5.0) < 1e-12


def test_round_trip_realised_pnl_is_net_of_both_sides():
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        broker, _, engine = build(tmp, clock)
        engine.mark(SYMBOL, 100.0)
        engine.sync()
        engine.place(SYMBOL, Side.BUY.value, 10.0, stamp="bar-1")

        engine.mark(SYMBOL, 110.0)
        engine.sync()
        engine.place(SYMBOL, Side.SELL.value, 10.0, stamp="bar-2")

        position = broker._positions[SYMBOL]
        assert abs(position.quantity) < 1e-12

        # Gross move is +10%, but slippage and two lots of fees eat into it.
        gross = 10.0 * (110.0 - 100.0)
        assert 0 < position.realised_pnl < gross
        # Cash should equal starting cash plus realised PnL.
        assert abs(broker.cash() - (10_000.0 + position.realised_pnl)) < 1e-6


# ------------------------------------------------------------ engine tests


def test_client_order_id_is_deterministic():
    a = make_client_order_id(SYMBOL, "buy", 1.5, "2026-04-01T09:00")
    b = make_client_order_id(SYMBOL, "buy", 1.5, "2026-04-01T09:00")
    c = make_client_order_id(SYMBOL, "buy", 1.6, "2026-04-01T09:00")
    assert a == b
    assert a != c


def test_replayed_bar_does_not_trade_twice():
    """The crash-and-restart case: same bar, same intent, one position."""
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        broker, _, engine = build(tmp, clock)
        engine.mark(SYMBOL, 100.0)
        engine.sync()

        first, _ = engine.place(SYMBOL, Side.BUY.value, 10.0, stamp="bar-1")
        second, _ = engine.place(SYMBOL, Side.BUY.value, 10.0, stamp="bar-1")

        assert first.client_order_id == second.client_order_id
        assert abs(broker.positions()[SYMBOL].quantity - 10.0) < 1e-12
        assert len([o for o in broker.orders().values() if o.status == "filled"]) == 1


def test_risk_gate_blocks_entries_but_not_exits():
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        broker, governor, engine = build(tmp, clock)
        engine.mark(SYMBOL, 100.0)
        engine.sync()
        engine.place(SYMBOL, Side.BUY.value, 10.0, stamp="bar-1")

        governor.halt("manual test halt")

        blocked, decision = engine.place(SYMBOL, Side.BUY.value, 5.0, stamp="bar-2")
        assert blocked is None
        assert not decision.allowed

        # You must always be able to get out.
        exit_order, _ = engine.place(SYMBOL, Side.SELL.value, 10.0, stamp="bar-3")
        assert exit_order is not None
        assert exit_order.status == OrderStatus.FILLED.value
        assert not broker.positions()


def test_exposure_cap_blocks_oversized_entry():
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        _, _, engine = build(tmp, clock, max_position_frac=0.25)
        engine.mark(SYMBOL, 100.0)
        engine.sync()

        # 30 units at 100 = 3,000 = 30% of 10,000 equity, over the 25% cap.
        blocked, decision = engine.place(SYMBOL, Side.BUY.value, 30.0, stamp="bar-1")
        assert blocked is None
        assert "exposure would be" in decision.describe()


def test_exit_records_trade_with_governor():
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        _, governor, engine = build(tmp, clock)
        engine.mark(SYMBOL, 100.0)
        engine.sync()
        engine.place(SYMBOL, Side.BUY.value, 10.0, stamp="bar-1")
        assert governor.state.trades_today == 0  # entries are not closed trades

        engine.mark(SYMBOL, 90.0)
        engine.sync()
        engine.place(SYMBOL, Side.SELL.value, 10.0, stamp="bar-2")

        assert governor.state.trades_today == 1
        assert governor.state.consecutive_losses == 1, (
            "a fully closed losing position must be recorded as a loss - if this "
            "reads 0, realised PnL is being lost when the position goes flat"
        )

        # And a winning full exit must clear the streak, not read as zero either.
        engine.mark(SYMBOL, 100.0)
        engine.sync()
        engine.place(SYMBOL, Side.BUY.value, 5.0, stamp="bar-3")
        engine.mark(SYMBOL, 120.0)
        engine.sync()
        engine.place(SYMBOL, Side.SELL.value, 5.0, stamp="bar-4")

        assert governor.state.trades_today == 2
        assert governor.state.consecutive_losses == 0


def test_reconciliation_divergence_halts():
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        _, governor, engine = build(tmp, clock)
        engine.mark(SYMBOL, 100.0)
        engine.sync()
        engine.place(SYMBOL, Side.BUY.value, 10.0, stamp="bar-1")

        # Simulate our view drifting from the broker's.
        engine.state.positions[SYMBOL]["quantity"] = 3.0

        assert engine.reconcile() is False
        assert governor.state.halted
        assert "reconciliation" in governor.state.halt_reason
        # The broker's number wins.
        assert abs(engine.state.positions[SYMBOL]["quantity"] - 10.0) < 1e-12


def test_state_survives_restart():
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        broker, _, engine = build(tmp, clock)
        engine.mark(SYMBOL, 100.0)
        engine.sync()
        engine.place(SYMBOL, Side.BUY.value, 10.0, stamp="bar-1")
        cash_before = broker.cash()

        # New process: fresh broker object, same state directory.
        broker2, _, engine2 = build(tmp, clock)
        engine2.mark(SYMBOL, 100.0)

        assert abs(broker2.cash() - cash_before) < 1e-9
        assert abs(broker2.positions()[SYMBOL].quantity - 10.0) < 1e-12
        # And the replayed bar still does not trade twice.
        engine2.sync()
        engine2.place(SYMBOL, Side.BUY.value, 10.0, stamp="bar-1")
        assert abs(broker2.positions()[SYMBOL].quantity - 10.0) < 1e-12


def test_target_exposure_moves_to_the_right_size():
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        broker, _, engine = build(tmp, clock)
        engine.mark(SYMBOL, 100.0)
        engine.sync()

        engine.target_exposure(SYMBOL, 0.20, stamp="bar-1", reason="test")
        position = broker.positions()[SYMBOL]
        # 20% of 10,000 equity at 100 = 20 units, within rounding for fees.
        assert 19.0 < position.quantity < 20.5

        engine.target_exposure(SYMBOL, 0.0, stamp="bar-2", reason="flatten")
        assert not broker.positions()


def test_tiny_rebalance_is_skipped():
    clock = FakeClock(T0)
    with tempfile.TemporaryDirectory() as tmp:
        broker, _, engine = build(tmp, clock)
        engine.mark(SYMBOL, 100.0)
        engine.sync()
        engine.target_exposure(SYMBOL, 0.20, stamp="bar-1")
        qty = broker.positions()[SYMBOL].quantity

        # A 0.01% target change is not worth paying fees for.
        order, _ = engine.target_exposure(SYMBOL, 0.2001, stamp="bar-2")
        assert order is None
        assert abs(broker.positions()[SYMBOL].quantity - qty) < 1e-12


def test_alpaca_paper_broker_is_a_stub():
    from execution import AlpacaPaperBroker

    try:
        AlpacaPaperBroker()
    except NotImplementedError as exc:
        assert "not implemented" in str(exc).lower()
    else:
        raise AssertionError("AlpacaPaperBroker should refuse to construct")


# ---------------------------------------------------------------- runner


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for test in tests:
        try:
            test()
            print(f"  [ok]   {test.__name__}")
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"  [FAIL] {test.__name__}: {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} of {len(tests)} tests FAILED")
        return 1
    print(f"All {len(tests)} execution tests passed.")
    return 0


if __name__ == "__main__":
    import logging

    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
