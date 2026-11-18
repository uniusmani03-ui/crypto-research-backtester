"""
Paper trading demo: real public data, real execution path, simulated broker.

    python scripts\run_paper.py                     # replay the last 500 bars
    python scripts\run_paper.py --reset             # start from a clean state dir
    python scripts\run_paper.py --mode live         # poll for new bars and act on them
    python scripts\run_paper.py --strategy rsi_mean_reversion --bars 1000

WHAT THIS IS
    Every bar goes through the full production path: strategy signal ->
    ATR-based allocation -> risk governor -> broker -> persisted state. The
    only simulated component is the broker itself. Nothing here touches an
    exchange account, and no credentials exist anywhere in this project.

WHAT THIS IS NOT
    It is not a performance test, and its numbers will NOT match
    run_example.py. The backtester is vectorised, fills at the bar close, and
    applies intrabar ATR stops. This walks bar by bar, sizes from live equity,
    and has no intrabar stop at all. Two different fill models produce two
    different answers, and neither is the truth. Use run_example.py to judge a
    strategy; use this to judge the plumbing.

    --mode live polls public market data and trades a simulated account in
    real time. It is a rehearsal of the loop, not a live bot.

KILL SWITCH
    Create a file called HALT in the state directory (printed at startup) and
    everything stops on the next check. It works while this is running, from
    another terminal, without touching this process.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from backtest import RiskModel  # noqa: E402
from data import load_ohlcv  # noqa: E402
from execution import (  # noqa: E402
    ExecutionEngine,
    RiskGovernor,
    RiskLimits,
    SimulatedBroker,
)
from strategies import STRATEGIES  # noqa: E402

REPORTS = ROOT / "reports"
log = logging.getLogger("paper")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=("replay", "live"), default="replay")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="4h")
    p.add_argument("--strategy", default="ema_crossover", choices=sorted(STRATEGIES))
    p.add_argument("--bars", type=int, default=500, help="replay: how many recent bars")
    p.add_argument("--since", default="2021-01-01")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--risk", type=float, default=0.01, help="fraction of equity risked per trade")
    p.add_argument("--atr-mult", type=float, default=2.0)
    p.add_argument("--max-exposure", type=float, default=0.50)
    p.add_argument("--max-daily-loss", type=float, default=0.05)
    p.add_argument("--max-drawdown", type=float, default=0.20)
    p.add_argument("--max-trades-per-day", type=int, default=20)
    p.add_argument("--state-dir", default="state")
    p.add_argument("--reset", action="store_true", help="wipe the state directory first")
    p.add_argument("--poll-seconds", type=int, default=60, help="live mode: seconds between polls")
    p.add_argument("--max-iterations", type=int, default=0, help="live mode: 0 means run forever")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


class Clock:
    """
    Wall clock, unless pinned to a bar timestamp.

    Replay needs this. A 500-bar replay spanning June to August finishes in
    seconds of real time, so a wall-clock governor sees every trade as
    happening on the same day: the daily-loss and trades-per-day breakers
    never roll over and never fire. Pinning the clock to each bar's timestamp
    makes the replay a faithful rehearsal instead of a compressed one.

    Live mode never pins, so it gets the real time.
    """

    def __init__(self) -> None:
        self._pinned: datetime | None = None

    def __call__(self) -> datetime:
        return self._pinned or datetime.now(timezone.utc)

    def pin(self, moment: datetime) -> None:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        self._pinned = moment


def build(args) -> tuple[SimulatedBroker, RiskGovernor, ExecutionEngine, RiskModel, Clock]:
    state_dir = ROOT / args.state_dir
    if args.reset and state_dir.exists():
        shutil.rmtree(state_dir)
        print(f"Wiped state directory {state_dir}")
    state_dir.mkdir(parents=True, exist_ok=True)

    clock = Clock()
    broker = SimulatedBroker(
        starting_cash=args.cash,
        fee_pct=args.fee_bps / 1e4,
        slippage_pct=args.slippage_bps / 1e4,
        clock=clock,
    )
    governor = RiskGovernor(
        RiskLimits(
            max_daily_loss_pct=args.max_daily_loss,
            max_drawdown_pct=args.max_drawdown,
            max_position_frac=args.max_exposure,
            max_trades_per_day=args.max_trades_per_day,
        ),
        state_dir=state_dir,
        clock=clock,
    )
    engine = ExecutionEngine(broker, governor, state_dir=state_dir, clock=clock)
    risk_model = RiskModel(
        risk_per_trade=args.risk,
        atr_mult=args.atr_mult,
        max_position_frac=args.max_exposure,
    )
    return broker, governor, engine, risk_model, clock


def prepare_signals(df: pd.DataFrame, strategy, risk_model: RiskModel) -> pd.DataFrame:
    """
    Signals and target sizes for every bar, lagged one bar.

    The lag matters as much here as in the backtester: a signal computed from
    bar t's close is acted on at bar t+1, because at the moment bar t closes
    you cannot also have traded on it.
    """
    entries, exits = strategy.generate_signals(df)
    size_frac = risk_model.position_frac(df)

    plan = pd.DataFrame(
        {
            "close": df["close"],
            "entry": entries.shift(1).fillna(False).astype(bool),
            "exit": exits.shift(1).fillna(False).astype(bool),
            "size_frac": size_frac.shift(1),
        }
    )
    return plan


def step(engine, governor, symbol: str, row, stamp: str, desired: float) -> dict:
    """One bar through the whole path. Returns a log record."""
    engine.mark(symbol, float(row["close"]))
    engine.sync()

    order = None
    decision = None
    if desired is not None:
        order, decision = engine.target_exposure(
            symbol, desired, stamp=stamp, reason=f"target {desired:.1%}"
        )

    position = engine.broker.positions().get(symbol)
    return {
        "timestamp": stamp,
        "price": float(row["close"]),
        "target_frac": desired,
        "action": order.side if order else "",
        "status": order.status if order else "",
        "reject_reason": order.reject_reason if order else "",
        "blocked": bool(decision is not None and not decision.allowed and order is None),
        "block_reason": decision.describe() if (decision and not decision.allowed) else "",
        "quantity": position.quantity if position else 0.0,
        "cash": engine.broker.cash(),
        "equity": engine.equity(),
        "halted": governor.state.halted,
    }


def resolve_target(row, want_long: bool) -> tuple[bool, float | None]:
    """
    Decide the desired position state, and whether it needs an order.

    This is a STATE MACHINE, not an edge handler, and the distinction is the
    whole point. If you re-issue a target every time a signal is true, the
    ATR-derived size drifts a little each bar and the bot rebalances forever -
    dozens of tiny orders, each paying full fees, none of them expressing a
    new opinion about anything.

    So: an order is placed only when the desired state actually changes.
    Flat -> long, or long -> flat. Everything else is "hold".

    Returns (new_want_long, target_fraction_or_None).
    """
    new_want = want_long
    if row["exit"]:
        new_want = False
    elif row["entry"]:
        new_want = True

    if new_want == want_long:
        return new_want, None

    if not new_want:
        return new_want, 0.0

    frac = row["size_frac"]
    if pd.isna(frac):
        # Still inside the ATR warm-up; we cannot size the position yet.
        return want_long, None
    return new_want, float(min(max(frac, 0.0), 1.0))


def run_replay(args, engine, governor, plan: pd.DataFrame, clock: Clock) -> pd.DataFrame:
    records = []
    want_long = False
    total = len(plan)

    for i, (timestamp, row) in enumerate(plan.iterrows(), start=1):
        if governor.kill_switch_engaged():
            print(f"\nKill switch detected at bar {i}. Stopping.")
            break

        # Governor sees bar time, not wall time, so daily breakers roll over
        # across the replayed period the way they would in real operation.
        clock.pin(timestamp.to_pydatetime())

        want_long, desired = resolve_target(row, want_long)
        record = step(engine, governor, args.symbol, row, timestamp.isoformat(), desired)
        records.append(record)

        # Intent follows reality. If an entry was blocked by the governor or
        # rejected by the broker we are still flat, and must be free to try
        # again on the next signal rather than believing we are long.
        want_long = record["quantity"] > 1e-12

        if not args.quiet and (i % 50 == 0 or i == total):
            print(
                f"  bar {i:>5}/{total}  {timestamp:%Y-%m-%d %H:%M}  "
                f"px {record['price']:>10,.2f}  eq {record['equity']:>11,.2f}  "
                f"pos {record['quantity']:>10.6f}"
                + ("  HALTED" if record["halted"] else "")
            )
    return pd.DataFrame(records)


def run_live(args, engine, governor, strategy, risk_model) -> pd.DataFrame:
    """
    Poll public market data and act on each newly closed bar.

    Deliberately conservative: it only ever acts on bars that have closed, and
    it refetches history each cycle rather than trying to maintain incremental
    state. Slower, and much harder to get subtly wrong.
    """
    records = []
    seen: set[str] = set()
    iterations = 0

    print(f"Polling every {args.poll_seconds}s. Ctrl+C to stop.")
    print(f"Kill switch: create {governor.kill_switch_path}")

    try:
        while True:
            iterations += 1
            if args.max_iterations and iterations > args.max_iterations:
                print(f"\nReached --max-iterations {args.max_iterations}. Stopping.")
                break
            if governor.kill_switch_engaged():
                print("\nKill switch engaged. Stopping.")
                break

            try:
                df, _ = load_ohlcv(args.symbol, args.timeframe, since=args.since)
            except Exception as exc:
                log.error("Fetch failed (%s). Retrying next cycle.", exc)
                time.sleep(args.poll_seconds)
                continue

            plan = prepare_signals(df, strategy, risk_model)
            timestamp = plan.index[-1]
            stamp = timestamp.isoformat()

            if stamp in seen:
                time.sleep(args.poll_seconds)
                continue
            seen.add(stamp)

            row = plan.iloc[-1]
            position = engine.broker.positions().get(args.symbol)
            holding = bool(position and position.quantity > 0)
            _, desired = resolve_target(row, holding)

            record = step(engine, governor, args.symbol, row, stamp, desired)
            records.append(record)

            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(
                f"  [{now}] bar {timestamp:%Y-%m-%d %H:%M}  px {record['price']:,.2f}  "
                f"eq {record['equity']:,.2f}  pos {record['quantity']:.6f}"
                + (f"  {record['action']} {record['status']}" if record["action"] else "")
                + ("  HALTED" if record["halted"] else "")
            )
            if record["block_reason"]:
                print(f"           blocked: {record['block_reason']}")

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nInterrupted. State has been persisted; rerun to resume.")

    return pd.DataFrame(records)


def summarise(args, broker, governor, engine, session: pd.DataFrame) -> None:
    banner("SESSION SUMMARY")
    print(engine.report())
    print()
    print(governor.report())

    if session.empty:
        print("\nNo bars processed.")
        return

    start_equity = args.cash
    end_equity = float(session["equity"].iloc[-1])
    peak = float(session["equity"].cummax().iloc[-1])
    drawdown = abs(float((session["equity"] / session["equity"].cummax() - 1).min()))

    filled = session[session["status"] == "filled"]
    rejected = session[session["status"] == "rejected"]
    blocked = session[session["blocked"]]

    # Split the result into what was actually banked and what is still riding
    # on an open position. A total return carried entirely by one open trade
    # is not the same claim as a total return made of closed ones.
    realised = sum(p.realised_pnl for p in broker._positions.values())
    unrealised = sum(
        p.unrealised_pnl(engine.marks.get(s, p.avg_price))
        for s, p in broker.positions().items()
    )

    print()
    print(f"  bars processed     {len(session):,}")
    print(f"  starting equity    {start_equity:,.2f}")
    print(f"  ending equity      {end_equity:,.2f}  ({end_equity / start_equity - 1:+.2%})")
    print(f"  peak equity        {peak:,.2f}")
    print(f"  max drawdown       {drawdown:.2%}")
    print(f"  realised PnL       {realised:,.2f}  (closed trades)")
    print(f"  unrealised PnL     {unrealised:,.2f}  (still open at the last bar)")
    print(f"  closed trades      {governor.state.trades_today} recorded by the governor")
    print(f"  orders filled      {len(filled)}")
    print(f"  orders rejected    {len(rejected)}")
    print(f"  orders blocked     {len(blocked)}  (risk governor)")

    if realised < 0 <= unrealised and end_equity > start_equity:
        print(
            "\n  NOTE: every closed trade taken together LOST money; the gain is "
            "an open position marked at the last bar. Nothing is banked until "
            "it is closed, and the exit is the part that has not been tested."
        )

    if len(blocked):
        print("\n  reasons the governor blocked orders:")
        for reason, count in blocked["block_reason"].value_counts().head(5).items():
            print(f"    {count:>4}x  {reason[:100]}")

    if len(rejected):
        print("\n  reasons the broker rejected orders:")
        for reason, count in rejected["reject_reason"].value_counts().head(5).items():
            print(f"    {count:>4}x  {reason[:100]}")

    REPORTS.mkdir(exist_ok=True)
    stamp = f"{args.strategy}_{args.symbol.replace('/', '-')}_{args.timeframe}_{args.mode}"
    out = REPORTS / f"paper_session_{stamp}.csv"
    session.to_csv(out, index=False)
    print(f"\n  session log -> {out}")

    print(
        "\nThese numbers describe the execution path, not the strategy. "
        "For strategy evaluation use run_example.py, which applies intrabar "
        "stops and a different fill model."
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    logging.getLogger("ccxt").setLevel(logging.WARNING)

    broker, governor, engine, risk_model, clock = build(args)
    strategy = STRATEGIES[args.strategy]()

    banner("PAPER TRADING (SIMULATED BROKER, REAL PUBLIC DATA, NO CREDENTIALS)")
    print(f"  strategy      {strategy.describe()}")
    print(f"  symbol        {args.symbol} {args.timeframe}")
    print(f"  broker        {broker.name}, starting cash {args.cash:,.2f}")
    print(f"  costs         {args.fee_bps:.1f}bps fee + {args.slippage_bps:.1f}bps slippage per side")
    print(f"  sizing        {risk_model.describe()}")
    print(f"  limits        {governor.limits.describe()}")
    print(f"  state dir     {engine.state_dir}")
    print(f"  KILL SWITCH   create {governor.kill_switch_path}")

    if governor.state.halted:
        print(f"\n  NOTE: governor is HALTED ({governor.state.halt_reason})")
        print("  Entries will be blocked. Use --reset for a clean run.")

    if args.mode == "live":
        session = run_live(args, engine, governor, strategy, risk_model)
    else:
        banner(f"FETCHING {args.symbol} {args.timeframe}")
        df, exchange = load_ohlcv(args.symbol, args.timeframe, since=args.since)
        print(f"{len(df):,} bars from {exchange}")

        plan = prepare_signals(df, strategy, risk_model).tail(args.bars)
        banner(f"REPLAYING {len(plan):,} BARS THROUGH THE EXECUTION PATH")
        print(f"  {plan.index[0]:%Y-%m-%d %H:%M} -> {plan.index[-1]:%Y-%m-%d %H:%M} UTC\n")
        session = run_replay(args, engine, governor, plan, clock)

    summarise(args, broker, governor, engine, session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
