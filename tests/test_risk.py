"""
Tests for the risk governor.

Runnable two ways, so there is no excuse not to run them:

    python tests\test_risk.py       (no dependencies)
    pytest tests\test_risk.py       (if you have pytest)

Every test uses a temporary state directory and an injected clock, so nothing
here touches real state or depends on the wall clock.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution import RiskGovernor, RiskHalted, RiskLimits  # noqa: E402

DAY1 = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
DAY2 = DAY1 + timedelta(days=1)


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now = self.now + timedelta(**kwargs)


def make(tmpdir: str, clock: FakeClock, **limit_kwargs) -> RiskGovernor:
    return RiskGovernor(RiskLimits(**limit_kwargs), state_dir=Path(tmpdir), clock=clock)


# --------------------------------------------------------------------- tests


def test_normal_order_allowed():
    with tempfile.TemporaryDirectory() as tmp:
        gov = make(tmp, FakeClock(DAY1))
        gov.update_equity(10_000)
        decision = gov.check_order(notional=2_000, current_exposure=0)
        assert decision.allowed, decision.describe()
        assert bool(decision) is True


def test_blocks_before_equity_is_known():
    """Fail closed: a governor that has not seen equity must not approve anything."""
    with tempfile.TemporaryDirectory() as tmp:
        gov = make(tmp, FakeClock(DAY1))
        decision = gov.check_order(notional=100)
        assert not decision.allowed
        assert "equity is zero or unset" in decision.describe()


def test_position_size_cap():
    with tempfile.TemporaryDirectory() as tmp:
        gov = make(tmp, FakeClock(DAY1), max_position_frac=0.25)
        gov.update_equity(10_000)

        assert gov.check_order(notional=2_500).allowed
        assert not gov.check_order(notional=2_501).allowed
        # Existing exposure counts toward the cap.
        assert not gov.check_order(notional=1_000, current_exposure=2_000).allowed


def test_daily_loss_blocks_then_resets_next_day():
    clock = FakeClock(DAY1)
    with tempfile.TemporaryDirectory() as tmp:
        gov = make(tmp, clock, max_daily_loss_pct=0.03)
        gov.update_equity(10_000)
        assert gov.check_order(notional=100).allowed

        gov.update_equity(9_650)  # -3.5% on the day
        blocked = gov.check_order(notional=100)
        assert not blocked.allowed
        assert "daily loss" in blocked.describe()

        # New UTC day: the daily breaker clears without human intervention.
        clock.now = DAY2
        gov.update_equity(9_650)
        assert gov.check_order(notional=100).allowed
        assert gov.state.trades_today == 0


def test_drawdown_latches_and_survives_restart():
    clock = FakeClock(DAY1)
    with tempfile.TemporaryDirectory() as tmp:
        gov = make(tmp, clock, max_drawdown_pct=0.20)
        gov.update_equity(10_000)
        gov.update_equity(7_900)  # -21% from peak

        assert not gov.check_order(notional=100).allowed
        assert gov.state.halted

        # A new day does NOT clear a latching breaker.
        clock.now = DAY2
        gov.update_equity(7_900)
        assert not gov.check_order(notional=100).allowed

        # Nor does restarting the process.
        reloaded = make(tmp, clock, max_drawdown_pct=0.20)
        assert reloaded.state.halted
        assert not reloaded.check_order(notional=100).allowed
        assert "drawdown" in reloaded.state.halt_reason


def test_reset_requires_exact_acknowledgement():
    with tempfile.TemporaryDirectory() as tmp:
        gov = make(tmp, FakeClock(DAY1), max_drawdown_pct=0.20)
        gov.update_equity(10_000)
        gov.update_equity(7_000)
        gov.check_order(notional=100)
        assert gov.state.halted

        for wrong in ("yes", "", "i have investigated", "I have investigated "):
            try:
                gov.reset(wrong)
            except RiskHalted:
                pass
            else:
                raise AssertionError(f"reset accepted {wrong!r}")

        gov.reset("I have investigated")
        assert not gov.state.halted
        gov.update_equity(10_000)
        assert gov.check_order(notional=100).allowed


def test_consecutive_losses_latch():
    with tempfile.TemporaryDirectory() as tmp:
        gov = make(tmp, FakeClock(DAY1), max_consecutive_losses=3, max_trades_per_day=100)
        gov.update_equity(10_000)

        gov.record_trade(-10)
        gov.record_trade(-10)
        assert gov.check_order(notional=100).allowed

        gov.record_trade(-10)
        assert not gov.check_order(notional=100).allowed
        assert gov.state.halted

        # A winner resets the streak - but only before the latch trips.
        gov.reset("I have investigated")
        gov.record_trade(+50)
        assert gov.state.consecutive_losses == 0


def test_trades_per_day_cap():
    with tempfile.TemporaryDirectory() as tmp:
        gov = make(tmp, FakeClock(DAY1), max_trades_per_day=2, max_consecutive_losses=99)
        gov.update_equity(10_000)

        gov.record_trade(+5)
        assert gov.check_order(notional=100).allowed
        gov.record_trade(+5)

        blocked = gov.check_order(notional=100)
        assert not blocked.allowed
        assert "trades today" in blocked.describe()


def test_kill_switch_file():
    with tempfile.TemporaryDirectory() as tmp:
        gov = make(tmp, FakeClock(DAY1))
        gov.update_equity(10_000)
        assert gov.check_order(notional=100).allowed

        gov.engage_kill_switch("testing")
        assert not gov.check_order(notional=100).allowed

        # A file created by hand works identically - that is the whole point.
        gov.release_kill_switch()
        assert gov.check_order(notional=100).allowed

        (Path(tmp) / "HALT").write_text("stopped by hand", encoding="utf-8")
        assert not gov.check_order(notional=100).allowed


def test_corrupt_state_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "risk_state.json").write_text("{not valid json", encoding="utf-8")
        gov = make(tmp, FakeClock(DAY1))
        assert gov.state.halted
        assert "unreadable" in gov.state.halt_reason
        gov.update_equity(10_000)
        assert not gov.check_order(notional=100).allowed


def test_peak_equity_persists():
    clock = FakeClock(DAY1)
    with tempfile.TemporaryDirectory() as tmp:
        gov = make(tmp, clock)
        gov.update_equity(10_000)
        gov.update_equity(12_000)
        assert gov.state.peak_equity == 12_000

        reloaded = make(tmp, clock)
        assert reloaded.state.peak_equity == 12_000

        # Drawdown is measured from the persisted peak, not this session's high.
        reloaded.update_equity(11_000)
        assert abs(reloaded.current_drawdown() - (1 - 11_000 / 12_000)) < 1e-9


def test_state_file_is_valid_json():
    with tempfile.TemporaryDirectory() as tmp:
        gov = make(tmp, FakeClock(DAY1))
        gov.update_equity(10_000)
        data = json.loads((Path(tmp) / "risk_state.json").read_text(encoding="utf-8"))
        assert data["peak_equity"] == 10_000
        assert data["halted"] is False


def test_warnings_before_the_wall():
    with tempfile.TemporaryDirectory() as tmp:
        # Daily limit set wide so only the drawdown warning is under test.
        gov = make(tmp, FakeClock(DAY1), max_drawdown_pct=0.20, max_daily_loss_pct=0.90)
        gov.update_equity(10_000)
        gov.update_equity(8_400)  # 16% drawdown, 80% of the way to the halt
        decision = gov.check_order(notional=100)
        assert decision.allowed, decision.describe()
        assert any("within 25%" in note for note in decision.notes), decision.notes


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
    print(f"All {len(tests)} risk-governor tests passed.")
    return 0


if __name__ == "__main__":
    import logging

    logging.disable(logging.CRITICAL)  # the governor logs errors on purpose
    raise SystemExit(main())
