"""
End-to-end example: fetch real public data, backtest, validate, report.

Run from the project root with the venv active:

    python scripts/run_example.py
    python scripts/run_example.py --strategy rsi_mean_reversion --timeframe 1d
    python scripts/run_example.py --symbol ETH/USDT --since 2020-01-01 --splits 6

What it does, in order:

  1. Fetches OHLCV from a public exchange (no API key), caching to Parquet.
  2. Runs a single costed, ATR-sized backtest on a simple train/test split.
  3. Runs walk-forward validation: parameters are chosen inside each training
     window and scored on the following untouched window. Prints in-sample and
     out-of-sample side by side, because the gap between them is the result.
  4. Runs a cost-sensitivity scan, because "does this survive realistic costs"
     is a more useful question than "what does it earn at my assumptions".
  5. Writes an equity-curve plot and CSVs to reports/.

Everything it prints is a property of the strategy AND the period AND the cost
assumptions. Change any of the three and the numbers change.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from backtest import (  # noqa: E402
    BacktestConfig,
    CostModel,
    RiskModel,
    format_comparison,
    run_strategy,
    train_test_split,
    walk_forward,
)
from backtest.metrics import extract_equity, extract_trades  # noqa: E402
from data import load_ohlcv  # noqa: E402
from strategies import STRATEGIES  # noqa: E402

REPORTS = ROOT / "reports"


def write_csv(frame: pd.DataFrame | pd.Series, path: Path, **kwargs) -> bool:
    """
    Write one report, and survive it failing.

    A backtest can take minutes; the writes at the end take milliseconds. Losing
    the whole run because the disk filled up, or OneDrive held a handle on the
    reports folder, is a bad trade. Each artefact is written independently so one
    failure costs one file instead of all of them, and the error is reported
    rather than swallowed.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, **kwargs)
    except OSError as exc:
        print(f"  FAILED {path.name}: {exc.__class__.__name__}: {exc}")
        return False
    print(f"  {path}")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="4h", help="1h, 4h, 1d, ...")
    p.add_argument("--since", default="2021-01-01")
    p.add_argument("--strategy", default="ema_crossover", choices=sorted(STRATEGIES))
    p.add_argument("--splits", type=int, default=5, help="walk-forward folds")
    p.add_argument("--train-frac", type=float, default=0.5, help="fraction of history per training window")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--fee-bps", type=float, default=10.0, help="commission per side, basis points")
    p.add_argument("--slippage-bps", type=float, default=5.0, help="slippage per side, basis points")
    p.add_argument("--risk", type=float, default=0.01, help="fraction of equity risked per trade")
    p.add_argument("--atr-mult", type=float, default=2.0)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    REPORTS.mkdir(exist_ok=True)

    strategy_cls = STRATEGIES[args.strategy]

    cfg = BacktestConfig(
        cost=CostModel(fee_pct=args.fee_bps / 1e4, slippage_pct=args.slippage_bps / 1e4),
        risk=RiskModel(risk_per_trade=args.risk, atr_mult=args.atr_mult),
        init_cash=args.cash,
        timeframe=args.timeframe,
    )

    # ------------------------------------------------------------- 1. data
    banner(f"1. FETCHING {args.symbol} {args.timeframe} SINCE {args.since}")
    df, exchange_used = load_ohlcv(args.symbol, args.timeframe, since=args.since)
    print(
        f"{len(df):,} bars from {exchange_used}: "
        f"{df.index[0]:%Y-%m-%d %H:%M} -> {df.index[-1]:%Y-%m-%d %H:%M} UTC"
    )
    print(f"close: {df['close'].min():,.2f} .. {df['close'].max():,.2f}")
    print(f"config: {cfg.describe()}")

    gaps = df.index.to_series().diff().value_counts().head(3)
    print(f"\nbar spacing (top 3):\n{gaps.to_string()}")

    # ------------------------------------- 2. single split, default params
    banner("2. SINGLE TRAIN/TEST SPLIT (default parameters, no optimisation)")
    train_df, test_df = train_test_split(df, train_frac=0.7)
    strategy = strategy_cls()
    print(f"strategy: {strategy.describe()}")
    print(f"train: {len(train_df):,} bars  |  test: {len(test_df):,} bars")

    _, m_train = run_strategy(train_df, strategy, cfg, label="train (in-sample)")
    _, m_test = run_strategy(test_df, strategy, cfg, label="test (out-of-sample)")
    print()
    print(format_comparison([m_train, m_test], title="Default parameters, 70/30 chronological split"))

    # ------------------------------------------------ 3. walk-forward
    banner(f"3. WALK-FORWARD VALIDATION ({args.splits} folds, parameters refit per fold)")
    n_combos = len(list(strategy_cls.grid()))
    print(f"parameter grid: {n_combos} combinations x {args.splits} folds = {n_combos * args.splits} simulations")
    print("parameters are selected on training data only; test windows are never optimised on.\n")

    wf = walk_forward(
        df,
        strategy_cls,
        cfg,
        n_splits=args.splits,
        train_frac=args.train_frac,
    )
    print()
    print(wf.report())

    # ------------------------------------------------ 4. cost sensitivity
    banner("4. COST SENSITIVITY (same strategy, same data, different assumptions)")
    print("If the edge only exists at low cost assumptions, it is not an edge.\n")
    rows = []
    for label, factor in [("zero cost", 0.0), ("half", 0.5), ("baseline", 1.0), ("double", 2.0), ("4x", 4.0)]:
        scan_cfg = BacktestConfig(
            cost=cfg.cost.scaled(factor) if factor else CostModel(0.0, 0.0),
            risk=cfg.risk,
            init_cash=cfg.init_cash,
            timeframe=cfg.timeframe,
        )
        try:
            _, m = run_strategy(test_df, strategy, scan_cfg, label=label)
            rows.append(
                {
                    "assumption": label,
                    "round_trip_bps": scan_cfg.cost.round_trip_bps,
                    "total_return": m.total_return,
                    "sharpe": m.sharpe,
                    "trades": m.trade_count,
                }
            )
        except Exception as exc:
            print(f"  {label}: failed ({exc})")
    cost_table = pd.DataFrame(rows)
    print(cost_table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    # ------------------------------------------------------- 5. artefacts
    banner("5. WRITING REPORTS")
    stamp = f"{args.strategy}_{args.symbol.replace('/', '-')}_{args.timeframe}"

    written = [
        write_csv(wf.summary_table(), REPORTS / f"walkforward_{stamp}.csv", index=False),
        write_csv(cost_table, REPORTS / f"costscan_{stamp}.csv", index=False),
        write_csv(wf.stitched_oos_equity, REPORTS / f"oos_equity_{stamp}.csv", header=["equity"]),
    ]

    # Full-period run, used for both the trade log and the plot.
    pf_all, m_all = run_strategy(df, strategy, cfg, label="full period")

    # The trade log is the audit trail: one row per trade, so any figure in the
    # summary tables can be traced back to the trades that produced it.
    trades_df = extract_trades(pf_all)
    if not trades_df.empty:
        written.append(write_csv(trades_df, REPORTS / f"trades_{stamp}.csv", index=False))
        print(
            f"    full period: {m_all.total_trades} trades "
            f"({m_all.trade_count} closed, {m_all.open_trade_count} open at final bar, "
            f"unrealised {m_all.unrealised_pnl:,.2f})"
        )
        for warning in m_all.warnings:
            print(f"    note: {warning}")

    equity_all_series = extract_equity(pf_all)
    written.append(
        write_csv(equity_all_series, REPORTS / f"equity_full_{stamp}.csv", header=["equity"])
    )

    if not all(written):
        print(
            f"\n  WARNING: {written.count(False)} of {len(written)} reports failed to write. "
            f"The backtest results above are still valid — only the files are missing."
        )

    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            equity_all = equity_all_series
            bh = cfg.init_cash * (df["close"] / df["close"].iloc[0])

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[3, 1])
            axes[0].plot(equity_all.index, equity_all.values, label="strategy (full period, fixed params)", lw=1.2)
            axes[0].plot(
                wf.stitched_oos_equity.index,
                wf.stitched_oos_equity.values,
                label="walk-forward out-of-sample (stitched)",
                lw=1.6,
            )
            axes[0].plot(bh.index, bh.values, label="buy & hold", lw=1.0, alpha=0.6)
            axes[0].set_yscale("log")
            axes[0].set_ylabel("equity (log)")
            axes[0].set_title(f"{args.symbol} {args.timeframe} — {strategy.describe()} — {cfg.cost.describe()}")
            axes[0].legend(loc="upper left", fontsize=8)
            axes[0].grid(alpha=0.3)

            dd = wf.stitched_oos_equity / wf.stitched_oos_equity.cummax() - 1.0
            axes[1].fill_between(dd.index, dd.values * 100, 0, alpha=0.4, color="crimson")
            axes[1].set_ylabel("OOS drawdown %")
            axes[1].grid(alpha=0.3)

            fig.tight_layout()
            out = REPORTS / f"equity_{stamp}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"  {out}")
        except Exception as exc:
            print(f"  plot skipped: {type(exc).__name__}: {exc}")

    banner("READ THIS BEFORE DRAWING CONCLUSIONS")
    print(
        "These are EXAMPLE strategies chosen to exercise the harness, not\n"
        "recommendations. The out-of-sample column is the only one worth much,\n"
        "and even that is one sample from one asset over one regime.\n"
        "See the 'Why a good backtest is not a good strategy' section of README.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
