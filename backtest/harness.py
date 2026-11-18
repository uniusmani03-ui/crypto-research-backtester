"""
The backtest harness: vectorbt simulation, train/test split, walk-forward.

Design rules this file enforces, so individual strategies cannot quietly break
them:

1. SIGNALS ARE LAGGED. A signal computed from bar t's close is acted on at
   bar t+1. Backtests that enter at the close of the bar that generated the
   signal are trading on information they did not have. That single mistake
   accounts for a large share of implausibly good backtests.

2. COSTS ARE ALWAYS APPLIED. Fees and slippage are passed to every simulation.
   There is no uncosted path through this code.

3. POSITIONS ARE ATR-SIZED AND STOPPED. Size and stop come from the same
   RiskModel, so the risk-per-trade number actually means something.

4. PARAMETER SEARCH ONLY EVER SEES TRAINING DATA. `walk_forward` selects
   parameters inside each training window and scores them on the following,
   untouched, test window. In-sample and out-of-sample results are reported
   side by side because the gap between them is the actual finding.

The first bars of every simulation are a warm-up region where indicators are
still NaN and no signals fire. That is intentional, not a bug in the data.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vectorbt is not installed. Run setup.ps1, or: pip install -r requirements.txt"
    ) from exc

from data.fetch import BARS_PER_YEAR, PANDAS_FREQ
from strategies.base import Strategy

from .costs import CostModel
from .metrics import Metrics, compute_metrics, degradation_note, format_comparison
from .sizing import RiskModel

log = logging.getLogger(__name__)


# ------------------------------------------------------------------- config


@dataclass
class BacktestConfig:
    cost: CostModel = field(default_factory=CostModel)
    risk: RiskModel = field(default_factory=RiskModel)
    init_cash: float = 10_000.0
    timeframe: str = "1h"
    #: Bars between signal generation and execution. 1 is the honest minimum.
    signal_lag: int = 1
    #: Long-only. Shorting has borrow costs and funding this harness does not model.
    direction: str = "longonly"

    @property
    def bars_per_year(self) -> float:
        return float(BARS_PER_YEAR.get(self.timeframe, 365.0))

    @property
    def freq(self) -> str:
        return PANDAS_FREQ.get(self.timeframe, "1D")

    def describe(self) -> str:
        return (
            f"init_cash={self.init_cash:,.0f} | {self.cost.describe()} | "
            f"{self.risk.describe()} | lag={self.signal_lag} bar(s) | {self.timeframe}"
        )


@dataclass
class Prepared:
    """Per-bar sizing inputs, computed once on the full series and then sliced."""

    size_frac: pd.Series
    stop_frac: pd.Series


def prepare(df: pd.DataFrame, cfg: BacktestConfig) -> Prepared:
    return Prepared(
        size_frac=cfg.risk.position_frac(df),
        stop_frac=cfg.risk.stop_distance_frac(df),
    )


# --------------------------------------------------------------- simulation


def _from_signals(close, entries, exits, kwargs: dict) -> Any:
    """
    Call vectorbt with graceful degradation.

    vectorbt is pinned in requirements.txt, but this project is meant to survive
    the user upgrading it. If an optional keyword disappears or is renamed, drop
    it loudly rather than crashing — except for fees and slippage, which are
    never dropped, because a silently uncosted backtest is worse than an error.
    """
    attempts: list[tuple[dict, str]] = [
        (kwargs, "full"),
        ({k: v for k, v in kwargs.items() if k != "sl_stop"}, "without ATR stop"),
        (
            {k: v for k, v in kwargs.items() if k not in {"sl_stop", "size", "size_type"}},
            "without ATR stop or ATR sizing",
        ),
    ]
    last_exc: Optional[Exception] = None
    for attempt_kwargs, label in attempts:
        assert "fees" in attempt_kwargs and "slippage" in attempt_kwargs
        try:
            pf = vbt.Portfolio.from_signals(close, entries, exits, **attempt_kwargs)
            if label != "full":
                warnings.warn(
                    f"vectorbt rejected some arguments; simulated {label}. "
                    f"Results are NOT comparable to a full run. Check your vectorbt "
                    f"version against requirements.txt.",
                    stacklevel=2,
                )
            return pf
        except (TypeError, ValueError, KeyError) as exc:
            last_exc = exc
            log.debug("from_signals failed (%s): %s", label, exc)
    raise RuntimeError(
        f"vectorbt.Portfolio.from_signals failed for every argument set. "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def simulate(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    cfg: BacktestConfig,
    prepared: Optional[Prepared] = None,
    label: str = "",
) -> tuple[Any, Metrics]:
    """
    Run one costed, sized, lagged backtest.

    `entries` / `exits` are RAW strategy signals. The lag is applied here.
    """
    if prepared is None:
        prepared = prepare(df, cfg)

    close = df["close"].astype("float64")
    idx = close.index

    entries = entries.reindex(idx).fillna(False).astype(bool)
    exits = exits.reindex(idx).fillna(False).astype(bool)

    if cfg.signal_lag:
        entries = entries.shift(cfg.signal_lag).fillna(False).astype(bool)
        exits = exits.shift(cfg.signal_lag).fillna(False).astype(bool)

    size_frac = prepared.size_frac.reindex(idx).ffill()
    stop_frac = prepared.stop_frac.reindex(idx).ffill()

    kwargs: dict[str, Any] = {
        "init_cash": cfg.init_cash,
        "fees": cfg.cost.fee_pct,
        "slippage": cfg.cost.slippage_pct,
        "size": size_frac.values,
        "size_type": "percent",
        "sl_stop": stop_frac.values,
        "direction": cfg.direction,
        "freq": cfg.freq,
    }

    pf = _from_signals(close, entries, exits, kwargs)
    m = compute_metrics(pf, cfg.bars_per_year, label=label, price=close)
    return pf, m


def run_strategy(
    df: pd.DataFrame,
    strategy: Strategy,
    cfg: BacktestConfig,
    prepared: Optional[Prepared] = None,
    label: str = "",
) -> tuple[Any, Metrics]:
    entries, exits = strategy.generate_signals(df)
    return simulate(df, entries, exits, cfg, prepared=prepared, label=label or strategy.describe())


# ------------------------------------------------------------------ splitting


def train_test_split(
    df: pd.DataFrame, train_frac: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split. Never shuffle time series — that is lookahead by another name."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1).")
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


@dataclass
class Fold:
    index: int
    train: slice
    test: slice


def walk_forward_folds(
    n_bars: int, n_splits: int = 5, train_frac: float = 0.5, anchored: bool = False
) -> list[Fold]:
    """
    Rolling (or anchored) walk-forward windows.

    `train_frac` is the fraction of the whole series used for each training
    window. The remainder is divided evenly into `n_splits` test windows, each
    following its own training window.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1.")
    train_size = int(n_bars * train_frac)
    test_size = (n_bars - train_size) // n_splits
    if train_size < 2 or test_size < 2:
        raise ValueError(
            f"Not enough data for {n_splits} folds at train_frac={train_frac} "
            f"({n_bars} bars gives train={train_size}, test={test_size}). "
            f"Fetch a longer history or reduce n_splits."
        )
    folds: list[Fold] = []
    for i in range(n_splits):
        train_start = 0 if anchored else i * test_size
        train_end = train_size + i * test_size
        test_end = min(train_end + test_size, n_bars)
        if test_end <= train_end:
            break
        folds.append(Fold(i, slice(train_start, train_end), slice(train_end, test_end)))
    return folds


# ---------------------------------------------------------------- optimisation


def sharpe_with_min_trades(m: Metrics, min_trades: int = 10) -> float:
    """
    Default objective: Sharpe, but a parameter set that barely trades is
    disqualified rather than allowed to win on three lucky trades.
    """
    if m.trade_count < min_trades:
        return -np.inf
    return m.sharpe if np.isfinite(m.sharpe) else -np.inf


@dataclass
class SweepResult:
    rows: pd.DataFrame
    best_params: dict[str, Any]
    best_metrics: Metrics


def sweep(
    df: pd.DataFrame,
    strategy_cls: type[Strategy],
    cfg: BacktestConfig,
    signal_cache: Optional[dict] = None,
    objective: Callable[[Metrics], float] = sharpe_with_min_trades,
    param_sets: Optional[Iterable[dict]] = None,
    prepared: Optional[Prepared] = None,
) -> SweepResult:
    """
    Evaluate every parameter combination on `df` and return them ranked.

    `signal_cache` lets the caller compute signals once on the full history and
    reuse the slices across folds — indicators are causal, so slicing after the
    fact introduces no lookahead and saves a lot of recomputation.
    """
    if prepared is None:
        prepared = prepare(df, cfg)
    param_sets = list(param_sets if param_sets is not None else strategy_cls.grid())

    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best: Optional[tuple[dict, Metrics]] = None

    for params in param_sets:
        key = (strategy_cls.name, tuple(sorted(params.items())))
        if signal_cache is not None and key in signal_cache:
            entries, exits = signal_cache[key]
            entries = entries.reindex(df.index).fillna(False)
            exits = exits.reindex(df.index).fillna(False)
        else:
            entries, exits = strategy_cls(**params).generate_signals(df)

        try:
            _, m = simulate(df, entries, exits, cfg, prepared=prepared, label=str(params))
        except Exception as exc:  # one bad combo should not kill the sweep
            log.warning("Params %s failed: %s", params, exc)
            continue

        score = objective(m)
        rows.append(
            {
                **params,
                "score": score,
                "sharpe": m.sharpe,
                "total_return": m.total_return,
                "max_drawdown": m.max_drawdown,
                "profit_factor": m.profit_factor,
                "win_rate": m.win_rate,
                "trades": m.trade_count,
            }
        )
        if score > best_score:
            best_score, best = score, (params, m)

    if best is None:
        raise RuntimeError("Every parameter combination failed or was disqualified.")

    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return SweepResult(rows=table, best_params=best[0], best_metrics=best[1])


# --------------------------------------------------------------- walk-forward


@dataclass
class FoldResult:
    fold: int
    params: dict[str, Any]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    in_sample: Metrics
    out_of_sample: Metrics


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    stitched_oos_equity: pd.Series
    stitched_oos: Metrics
    aggregate_is: Metrics
    config: BacktestConfig

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            rows.append(
                {
                    "fold": f.fold,
                    "params": ", ".join(f"{k}={v}" for k, v in sorted(f.params.items())),
                    "test_from": f.test_start.date(),
                    "test_to": f.test_end.date(),
                    "IS_sharpe": f.in_sample.sharpe,
                    "OOS_sharpe": f.out_of_sample.sharpe,
                    "IS_return": f.in_sample.total_return,
                    "OOS_return": f.out_of_sample.total_return,
                    "OOS_maxdd": f.out_of_sample.max_drawdown,
                    "OOS_trades": f.out_of_sample.trade_count,
                }
            )
        return pd.DataFrame(rows)

    def report(self) -> str:
        parts = [
            format_comparison(
                [self.aggregate_is, self.stitched_oos],
                title="IN-SAMPLE vs OUT-OF-SAMPLE (walk-forward, stitched)",
            ),
            "",
            degradation_note(self.aggregate_is, self.stitched_oos),
            "",
            "Per-fold detail:",
            self.summary_table().to_string(
                index=False,
                float_format=lambda v: f"{v:,.3f}",
            ),
        ]
        return "\n".join(parts)


def walk_forward(
    df: pd.DataFrame,
    strategy_cls: type[Strategy],
    cfg: BacktestConfig,
    n_splits: int = 5,
    train_frac: float = 0.5,
    anchored: bool = False,
    objective: Callable[[Metrics], float] = sharpe_with_min_trades,
) -> WalkForwardResult:
    """
    Select parameters in each training window, score them on the next test
    window, and stitch the untouched out-of-sample periods into one curve.

    This is the part that stops a parameter sweep from flattering itself. A
    sweep alone will always find *something* that worked on the data it was
    shown; the only question worth asking is whether it kept working on data it
    was not shown.
    """
    folds = walk_forward_folds(len(df), n_splits, train_frac, anchored)
    prepared_full = prepare(df, cfg)
    param_sets = list(strategy_cls.grid())

    # Compute each combination's signals once on the full (causal) series.
    signal_cache: dict = {}
    for params in param_sets:
        key = (strategy_cls.name, tuple(sorted(params.items())))
        signal_cache[key] = strategy_cls(**params).generate_signals(df)

    def slice_prepared(sl: slice) -> Prepared:
        return Prepared(
            size_frac=prepared_full.size_frac.iloc[sl],
            stop_frac=prepared_full.stop_frac.iloc[sl],
        )

    results: list[FoldResult] = []
    oos_return_chunks: list[pd.Series] = []
    is_return_chunks: list[pd.Series] = []

    for fold in folds:
        train_df = df.iloc[fold.train]
        test_df = df.iloc[fold.test]

        probe = strategy_cls(**strategy_cls.defaults())
        if len(test_df) < probe.warmup * 2:
            log.warning(
                "Fold %d test window (%d bars) is short relative to the strategy "
                "warm-up (%d bars). Its metrics will be thin.",
                fold.index, len(test_df), probe.warmup,
            )

        sweep_result = sweep(
            train_df,
            strategy_cls,
            cfg,
            signal_cache=signal_cache,
            objective=objective,
            param_sets=param_sets,
            prepared=slice_prepared(fold.train),
        )
        best_params = sweep_result.best_params

        key = (strategy_cls.name, tuple(sorted(best_params.items())))
        entries, exits = signal_cache[key]

        pf_is, m_is = simulate(
            train_df,
            entries.iloc[fold.train],
            exits.iloc[fold.train],
            cfg,
            prepared=slice_prepared(fold.train),
            label=f"fold {fold.index} IS",
        )
        pf_oos, m_oos = simulate(
            test_df,
            entries.iloc[fold.test],
            exits.iloc[fold.test],
            cfg,
            prepared=slice_prepared(fold.test),
            label=f"fold {fold.index} OOS",
        )

        from .metrics import extract_equity

        oos_return_chunks.append(extract_equity(pf_oos).pct_change().dropna())
        is_return_chunks.append(extract_equity(pf_is).pct_change().dropna())

        results.append(
            FoldResult(
                fold=fold.index,
                params=best_params,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                in_sample=m_is,
                out_of_sample=m_oos,
            )
        )
        log.info(
            "Fold %d: params=%s  IS Sharpe %.2f -> OOS Sharpe %.2f (%d OOS trades)",
            fold.index, best_params, m_is.sharpe, m_oos.sharpe, m_oos.trade_count,
        )

    close = df["close"]
    stitched_equity, stitched = _stitch(
        oos_return_chunks, results, cfg, "out-of-sample (stitched)", price=close
    )
    _, aggregate_is = _stitch(
        is_return_chunks, results, cfg, "in-sample (aggregate)", price=close, is_sample=True
    )

    return WalkForwardResult(
        folds=results,
        stitched_oos_equity=stitched_equity,
        stitched_oos=stitched,
        aggregate_is=aggregate_is,
        config=cfg,
    )


def _stitch(
    chunks: list[pd.Series],
    results: list[FoldResult],
    cfg: BacktestConfig,
    label: str,
    price: Optional[pd.Series] = None,
    is_sample: bool = False,
) -> tuple[pd.Series, Metrics]:
    """Chain per-fold returns into one equity curve and summarise it."""
    if not chunks:
        return pd.Series(dtype="float64"), Metrics(label=label, significance="no data")

    rets = pd.concat(chunks)
    rets = rets[~rets.index.duplicated(keep="first")].sort_index()
    equity = cfg.init_cash * (1.0 + rets).cumprod()

    m = Metrics(label=label)
    m.start, m.end, m.bars = equity.index[0], equity.index[-1], len(equity)
    m.init_cash = cfg.init_cash
    m.final_equity = float(equity.iloc[-1])
    m.total_return = m.final_equity / m.init_cash - 1.0

    years = max(m.bars / cfg.bars_per_year, 1e-9)
    m.cagr = (m.final_equity / m.init_cash) ** (1 / years) - 1 if m.final_equity > 0 else float("nan")
    dd = equity / equity.cummax() - 1.0
    m.max_drawdown = float(-dd.min())

    if len(rets) > 1 and rets.std(ddof=1) > 0:
        ann = np.sqrt(cfg.bars_per_year)
        m.volatility = float(rets.std(ddof=1) * ann)
        m.sharpe = float(rets.mean() / rets.std(ddof=1) * ann)
        down = rets[rets < 0]
        m.sortino = (
            float(rets.mean() / down.std(ddof=1) * ann)
            if len(down) > 1 and down.std(ddof=1) > 0
            else float("nan")
        )
    else:
        m.volatility = m.sharpe = m.sortino = float("nan")
    m.calmar = float(m.cagr / m.max_drawdown) if m.max_drawdown > 0 else float("nan")

    # Trade statistics are POOLED across folds, not averaged.
    #
    # Averaging per-fold profit factors is a genuinely misleading statistic: a
    # fold with three trades and no losers has an enormous (or infinite) profit
    # factor, and averaging lets it drag the headline number up regardless of
    # how little money moved through it. Summing gross wins and gross losses
    # across all folds and dividing once weights every trade equally, which is
    # what the reader assumes the number means.
    source = [f.in_sample if is_sample else f.out_of_sample for f in results]

    m.trade_count = int(sum(s.trade_count for s in source))
    win_count = int(round(sum(s.win_rate * s.trade_count for s in source)))
    loss_count = m.trade_count - win_count
    m.win_rate = float(win_count / m.trade_count) if m.trade_count else 0.0

    m.gross_win = float(sum(s.gross_win for s in source))
    m.gross_loss = float(sum(s.gross_loss for s in source))
    m.profit_factor = (
        m.gross_win / m.gross_loss if m.gross_loss > 0 else float("nan")
    )

    m.avg_win = m.gross_win / win_count if win_count else 0.0
    m.avg_loss = -m.gross_loss / loss_count if loss_count else 0.0
    m.payoff_ratio = abs(m.avg_win / m.avg_loss) if m.avg_loss else float("nan")

    total_pnl = m.gross_win - m.gross_loss
    m.expectancy = total_pnl / m.trade_count if m.trade_count else float("nan")
    m.largest_loss = float(min([s.largest_loss for s in source], default=0.0))

    # Each fold ends at its own boundary, so several folds can each leave a
    # position open. These are summed rather than netted.
    m.open_trade_count = int(sum(s.open_trade_count for s in source))
    m.unrealised_pnl = float(sum(s.unrealised_pnl for s in source))

    # Buy & hold over the same window. Without this the table cannot answer the
    # first question anyone should ask: did the strategy beat just holding it?
    if price is not None:
        window = price.loc[(price.index >= equity.index[0]) & (price.index <= equity.index[-1])]
        window = window.dropna()
        if len(window) > 1:
            m.buy_hold_return = float(window.iloc[-1] / window.iloc[0] - 1.0)

    from .metrics import assess_significance

    m.significance, notes = assess_significance(m)
    m.warnings.extend(notes)

    if not is_sample:
        thin = [f.fold for f in results if f.out_of_sample.trade_count < 10]
        if thin:
            m.warnings.append(
                f"Fold(s) {thin} contributed fewer than 10 out-of-sample trades each. "
                f"Their returns still carry full weight in this stitched curve, so a "
                f"single lucky trade in a thin fold can dominate the headline figure."
            )
    return equity, m
