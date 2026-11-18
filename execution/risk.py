"""
Risk governor: circuit breakers that sit between a strategy and a broker.

This module knows nothing about strategies, models, regimes or signals. It sees
account equity, open exposure, today's trade count, and a proposed order. That
ignorance is the point. A breaker that depends on the model being right is not
a breaker, it is a second opinion from the same source.

Four design rules, each of which exists because the alternative fails badly:

1. FAIL CLOSED. Any error - unreadable state file, corrupt JSON, a limit that
   cannot be evaluated - blocks trading. A risk system that opens up when it is
   confused is worse than no risk system, because you trust it.

2. SOME BREAKERS LATCH. A daily loss limit resets tomorrow; that is a speed
   bump. A max-drawdown breach latches and stays latched until a human calls
   `reset()`. If your account is down 20%, something is wrong that a new
   calendar day does not fix, and the system should not quietly resume.

3. THE KILL SWITCH IS A FILE. Not a config flag, not an API call, not a
   keyboard interrupt. Creating a file named HALT in the project directory
   stops everything on the next check. It works when the process is wedged,
   when you are on your phone over SSH, and when you cannot remember how any
   of the rest of this works.

4. STATE IS PERSISTENT AND WRITTEN ATOMICALLY. Daily counters and peak equity
   have to survive a restart, or "max drawdown" resets every time the process
   dies - which, conveniently, is exactly when things are going wrong.

Day boundaries are UTC. That matches crypto, which trades continuously. If you
ever point this at equities, the trading day is an exchange session in the
venue's local time, and this needs changing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .atomic import atomic_write_text

log = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------- limits


@dataclass(frozen=True)
class RiskLimits:
    """
    Hard limits. Every one of these blocks orders when breached.

    The defaults are deliberately tight. Loosening them should be a decision
    you make on purpose, having thought about the number, not something you
    inherit from an example file.
    """

    #: Loss from the day's starting equity that stops new entries. Resets daily.
    max_daily_loss_pct: float = 0.03

    #: Drawdown from all-time peak equity that halts trading. LATCHES.
    max_drawdown_pct: float = 0.20

    #: Cap on total exposure as a fraction of equity, across all positions.
    max_position_frac: float = 0.25

    #: Cap on new trades per UTC day. Catches runaway signal loops.
    max_trades_per_day: int = 10

    #: Consecutive losing trades that halt trading. LATCHES.
    max_consecutive_losses: int = 5

    #: Absolute equity floor. Below this, stop. LATCHES.
    min_equity: float = 0.0

    #: Presence of this file halts everything. Relative to the state directory.
    kill_switch_filename: str = "HALT"

    def __post_init__(self) -> None:
        for name in ("max_daily_loss_pct", "max_drawdown_pct", "max_position_frac"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day must be at least 1")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be at least 1")
        if self.min_equity < 0:
            raise ValueError("min_equity cannot be negative")

    def describe(self) -> str:
        return (
            f"daily loss {self.max_daily_loss_pct:.1%}, "
            f"max DD {self.max_drawdown_pct:.1%} (latching), "
            f"max exposure {self.max_position_frac:.0%}, "
            f"max {self.max_trades_per_day} trades/day, "
            f"max {self.max_consecutive_losses} consecutive losses (latching)"
        )


# -------------------------------------------------------------------- state


@dataclass
class RiskState:
    """Everything the governor must remember across restarts."""

    equity: float = 0.0
    peak_equity: float = 0.0
    day: str = ""  # ISO date, UTC
    day_start_equity: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ""
    halted_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskState":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


# ---------------------------------------------------------------- decisions


@dataclass(frozen=True)
class RiskDecision:
    """The answer to 'may I send this order?', with its reasoning attached."""

    allowed: bool
    breaches: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.allowed

    def describe(self) -> str:
        if self.allowed:
            return "ALLOWED" + (f" ({'; '.join(self.notes)})" if self.notes else "")
        return "BLOCKED: " + "; ".join(self.breaches)


class RiskHalted(RuntimeError):
    """Raised when code tries to trade through a latched halt."""


# ---------------------------------------------------------------- governor


class RiskGovernor:
    """
    Stateful circuit breaker. Construct once, consult before every order,
    update after every fill.

    Typical use:

        gov = RiskGovernor(RiskLimits(), Path("state"))
        gov.update_equity(account_equity)

        decision = gov.check_order(notional=250.0, current_exposure=0.0)
        if not decision:
            log.warning("order blocked: %s", decision.describe())
            return

        # ... broker fills the order ...
        gov.record_trade(pnl=-12.40)
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        state_dir: Path | str = "state",
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "risk_state.json"
        self.kill_switch_path = self.state_dir / self.limits.kill_switch_filename
        self.clock = clock
        self._last_saved = ""
        self.state = self._load()

    # ----------------------------------------------------------- persistence

    def _load(self) -> RiskState:
        if not self.state_path.exists():
            return RiskState()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return RiskState.from_dict(data)
        except Exception as exc:
            # Fail closed. A corrupt state file means we do not know the peak
            # equity or today's losses, so we must not trade.
            log.error("Unreadable risk state at %s (%s). Halting.", self.state_path, exc)
            state = RiskState()
            state.halted = True
            state.halt_reason = f"risk state file unreadable ({type(exc).__name__})"
            state.halted_at = self.clock().isoformat()
            return state

    def _save(self) -> None:
        """
        Persist state, atomically, and only when it has actually changed.

        The no-op check matters more than it looks: a bar-by-bar run calls this
        on every bar, and an fsync per bar into a cloud-synced folder is both
        slow and a good way to collide with the sync client.
        """
        payload = self.state.to_json()
        if payload == self._last_saved:
            return
        try:
            atomic_write_text(self.state_path, payload)
            self._last_saved = payload
        except Exception as exc:
            log.error("Could not persist risk state (%s). Halting.", exc)
            self.state.halted = True
            self.state.halt_reason = f"could not persist risk state ({type(exc).__name__})"

    # -------------------------------------------------------------- updates

    def update_equity(self, equity: float, now: Optional[datetime] = None) -> None:
        """
        Record current account equity. Call this before checking an order.

        Rolls the UTC day when needed and tracks the all-time peak, which is
        what max drawdown is measured against.
        """
        now = now or self.clock()
        today = now.date().isoformat()

        if not self.state.day:
            self.state.day = today
            self.state.day_start_equity = equity
            self.state.peak_equity = equity
        elif self.state.day != today:
            log.info(
                "New UTC day %s. Daily counters reset (was %s, %d trades).",
                today, self.state.day, self.state.trades_today,
            )
            self.state.day = today
            self.state.day_start_equity = equity
            self.state.trades_today = 0

        self.state.equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self._save()

    def record_trade(self, pnl: float, now: Optional[datetime] = None) -> None:
        """Call after every closed trade. Feeds the daily and streak counters."""
        now = now or self.clock()
        self.state.trades_today += 1
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        self._save()

    # --------------------------------------------------------------- checks

    def kill_switch_engaged(self) -> bool:
        return self.kill_switch_path.exists()

    def current_drawdown(self) -> float:
        if self.state.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.state.equity / self.state.peak_equity)

    def daily_loss(self) -> float:
        if self.state.day_start_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.state.equity / self.state.day_start_equity)

    def check_order(
        self,
        notional: float = 0.0,
        current_exposure: float = 0.0,
        now: Optional[datetime] = None,
    ) -> RiskDecision:
        """
        May this order be sent?

        `notional` is the cash value of the proposed order. `current_exposure`
        is the cash value of positions already held. Both are absolute values.

        Latching breaches are recorded to state so they survive a restart.
        """
        now = now or self.clock()
        breaches: list[str] = []
        notes: list[str] = []

        # 1. Kill switch first: cheapest check, highest authority.
        if self.kill_switch_engaged():
            return RiskDecision(
                False,
                (f"kill switch file present: {self.kill_switch_path}",),
            )

        # 2. Already halted by a previous latching breach.
        if self.state.halted:
            return RiskDecision(
                False,
                (f"halted: {self.state.halt_reason} (at {self.state.halted_at}); "
                 f"call reset() after investigating",),
            )

        # 3. Sanity: we cannot evaluate anything without equity.
        if self.state.equity <= 0:
            return RiskDecision(
                False,
                ("equity is zero or unset - call update_equity() first",),
            )

        if notional < 0 or current_exposure < 0:
            return RiskDecision(False, ("negative notional or exposure passed to check_order",))

        # 4. Latching breaches.
        if self.state.equity < self.limits.min_equity:
            self._latch(
                f"equity {self.state.equity:,.2f} below floor {self.limits.min_equity:,.2f}", now
            )
            breaches.append(self.state.halt_reason)

        drawdown = self.current_drawdown()
        if drawdown >= self.limits.max_drawdown_pct:
            self._latch(
                f"drawdown {drawdown:.2%} at or beyond limit "
                f"{self.limits.max_drawdown_pct:.2%} (peak {self.state.peak_equity:,.2f})",
                now,
            )
            breaches.append(self.state.halt_reason)

        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            self._latch(
                f"{self.state.consecutive_losses} consecutive losses at or beyond limit "
                f"{self.limits.max_consecutive_losses}",
                now,
            )
            breaches.append(self.state.halt_reason)

        # 5. Non-latching breaches: block today, clear tomorrow.
        daily_loss = self.daily_loss()
        if daily_loss >= self.limits.max_daily_loss_pct:
            breaches.append(
                f"daily loss {daily_loss:.2%} at or beyond limit "
                f"{self.limits.max_daily_loss_pct:.2%} (resets at UTC midnight)"
            )

        if self.state.trades_today >= self.limits.max_trades_per_day:
            breaches.append(
                f"{self.state.trades_today} trades today, limit is "
                f"{self.limits.max_trades_per_day} (resets at UTC midnight)"
            )

        # 6. Position sizing.
        projected = current_exposure + notional
        max_allowed = self.limits.max_position_frac * self.state.equity
        if projected > max_allowed:
            breaches.append(
                f"exposure would be {projected:,.2f} "
                f"({projected / self.state.equity:.1%} of equity), limit is "
                f"{max_allowed:,.2f} ({self.limits.max_position_frac:.0%})"
            )

        # 7. Warnings that do not block, so you see a wall before you hit it.
        if not breaches:
            if drawdown >= 0.75 * self.limits.max_drawdown_pct:
                notes.append(f"drawdown {drawdown:.2%} is within 25% of the halt threshold")
            if daily_loss >= 0.75 * self.limits.max_daily_loss_pct:
                notes.append(f"daily loss {daily_loss:.2%} is within 25% of the daily limit")
            if self.state.trades_today >= self.limits.max_trades_per_day - 1:
                notes.append("one trade left today")

        return RiskDecision(not breaches, tuple(breaches), tuple(notes))

    # ------------------------------------------------------------- controls

    def _latch(self, reason: str, now: datetime) -> None:
        if not self.state.halted:
            log.error("RISK HALT: %s", reason)
            self.state.halted = True
            self.state.halt_reason = reason
            self.state.halted_at = now.isoformat()
            self._save()

    def halt(self, reason: str, now: Optional[datetime] = None) -> None:
        """Manual halt. Latches like any other."""
        self._latch(f"manual halt: {reason}", now or self.clock())

    def reset(self, acknowledgement: str) -> None:
        """
        Clear a latched halt.

        Deliberately awkward: you must pass the exact string "I have
        investigated". A halt means something went wrong, and a one-argument
        `reset()` is too easy to call reflexively from a retry loop.
        """
        expected = "I have investigated"
        if acknowledgement != expected:
            raise RiskHalted(
                f"reset() requires the exact acknowledgement {expected!r}. "
                f"The halt reason was: {self.state.halt_reason or '(none)'}"
            )
        log.warning("Risk halt cleared. Previous reason: %s", self.state.halt_reason)
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.halted_at = ""
        self.state.consecutive_losses = 0
        self._save()

    def engage_kill_switch(self, reason: str = "") -> Path:
        """Create the kill-switch file. Also doable by hand, which is the point."""
        self.kill_switch_path.write_text(
            f"{self.clock().isoformat()}\n{reason}\n", encoding="utf-8"
        )
        log.error("Kill switch engaged: %s", self.kill_switch_path)
        return self.kill_switch_path

    def release_kill_switch(self) -> None:
        if self.kill_switch_path.exists():
            self.kill_switch_path.unlink()
            log.warning("Kill switch released: %s", self.kill_switch_path)

    # -------------------------------------------------------------- display

    def snapshot(self) -> dict[str, Any]:
        """Flat dict for logging or a dashboard."""
        return {
            "equity": self.state.equity,
            "peak_equity": self.state.peak_equity,
            "drawdown": self.current_drawdown(),
            "daily_loss": self.daily_loss(),
            "day": self.state.day,
            "trades_today": self.state.trades_today,
            "consecutive_losses": self.state.consecutive_losses,
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "kill_switch": self.kill_switch_engaged(),
        }

    def report(self) -> str:
        s = self.snapshot()
        status = "HALTED" if s["halted"] else ("KILLED" if s["kill_switch"] else "running")
        return (
            f"risk governor [{status}]\n"
            f"  equity {s['equity']:,.2f}  peak {s['peak_equity']:,.2f}  "
            f"drawdown {s['drawdown']:.2%}\n"
            f"  today {s['day']}: loss {s['daily_loss']:.2%}, "
            f"{s['trades_today']} trades, {s['consecutive_losses']} consecutive losses\n"
            f"  limits: {self.limits.describe()}"
            + (f"\n  reason: {s['halt_reason']}" if s["halt_reason"] else "")
        )
