"""
Performance metrics, computed from the equity curve and the trade list.

These are calculated here in plain pandas rather than read off vectorbt's
`Portfolio.stats()` for two reasons: the numbers are then version-independent,
and you can see exactly what each one means. vectorbt's own stats table is still
available via `Portfolio.stats()` if you want a second opinion — and comparing
the two is a decent sanity check.

Metrics chosen for what they tell you:

  max_drawdown   The one that decides whether you can actually run the strategy.
                 Total return is a number; a 60% drawdown is an experience.
  sharpe         Return per unit of volatility. Annualised. Widely abused: it
                 assumes roughly normal returns, which trading strategies with
                 stops and fat tails do not have.
  sortino        Sharpe but only penalising downside deviation.
  calmar         CAGR divided by max drawdown. Blunt, hard to game.
  profit_factor  Gross wins / gross losses. Below 1.0 the strategy loses money.
                 Above ~1.5 on out-of-sample data is genuinely interesting.
  win_rate       The most-quoted and least-informative metric here. A 90% win
                 rate with a 1:20 payoff ratio is a losing strategy.
  trade_count    Determines whether anything above is meaningful at all.

On significance: with N trades, the standard error of the win rate is roughly
0.5/sqrt(N). At 25 trades that is +/-10 percentage points — wide enough that a
55% win rate and a 45% win rate are indistinguishable. `Metrics.significance`
flags this so a thin result cannot quietly pass as a real one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

#: Below this many closed trades, results are noise.
MIN_TRADES_MEANINGFUL = 30
#: Below this many, treat conclusions as provisional.
MIN_TRADES_CONFIDENT = 100


# --------------------------------------------------------------------- vectorbt
# Small tolerant accessors. vectorbt's public surface is stable in the parts we
# use, but this project must not fall over on a point release.


def extract_equity(pf: Any) -> pd.Series:
    """Equity curve as a Series, whatever the vectorbt version calls it."""
    for attr in ("value", "equity"):
        obj = getattr(pf, attr, None)
        if obj is None:
            continue
        try:
            series = obj() if callable(obj) else obj
        except Exception:
            continue
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        if isinstance(series, pd.Series) and len(series):
            return series.astype("float64")
    raise AttributeError(
        "Could not read an equity curve off this Portfolio object. "
        "vectorbt's API may have changed; check Portfolio.value()."
    )


def _find_column(df: pd.DataFrame, names: set[str]) -> Optional[Any]:
    """Locate a column by case-insensitive name, tolerating vectorbt's spacing."""
    for col in df.columns:
        if str(col).strip().lower() in names:
            return col
    return None


def extract_trades(pf: Any) -> pd.DataFrame:
    """
    The full trade list as a readable DataFrame: one row per trade, with entry
    and exit timestamps, prices, size, PnL and return.

    This is the audit trail. If a metric in the summary table looks wrong, this
    is where you go to find out why, trade by trade.
    """
    trades = getattr(pf, "trades", None)
    if trades is None:
        return pd.DataFrame()
    for attr in ("records_readable", "readable"):
        recs = getattr(trades, attr, None)
        if recs is None:
            continue
        try:
            df = recs() if callable(recs) else recs
        except Exception:
            continue
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return pd.DataFrame()


def split_trade_pnl(trades_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Return (realised PnL of closed trades, unrealised PnL of still-open trades).

    Keeping these apart matters. Win rate, profit factor and expectancy are
    statistics about *completed* trades; the equity curve includes the open one.
    If a backtest ends mid-position, those two views of the same run disagree,
    and a reader who does not know that will think one of them is broken.
    """
    empty = pd.Series(dtype="float64")
    if trades_df.empty:
        return empty, empty

    pnl_col = _find_column(trades_df, {"pnl", "profit", "net pnl"})
    if pnl_col is None:
        return empty, empty

    pnl = pd.to_numeric(trades_df[pnl_col], errors="coerce")
    status_col = _find_column(trades_df, {"status"})

    if status_col is None:
        # No status field: assume everything is closed rather than silently
        # dropping trades.
        return pnl.dropna().reset_index(drop=True), empty

    is_closed = trades_df[status_col].astype(str).str.strip().str.lower().eq("closed")
    closed = pnl[is_closed].dropna().reset_index(drop=True)
    open_ = pnl[~is_closed].dropna().reset_index(drop=True)
    return closed, open_


def extract_trade_pnl(pf: Any) -> pd.Series:
    """Realised PnL per closed trade, in cash. Kept for backwards compatibility."""
    closed, _ = split_trade_pnl(extract_trades(pf))
    return closed


# ---------------------------------------------------------------------- metrics


@dataclass
class Metrics:
    label: str = ""
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    bars: int = 0
    init_cash: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    volatility: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0
    expectancy: float = 0.0
    largest_loss: float = 0.0
    open_trade_count: int = 0
    unrealised_pnl: float = 0.0
    buy_hold_return: float = float("nan")
    significance: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        """
        Every position the strategy took, closed or still open.

        `trade_count` deliberately counts closed trades only, because win rate,
        profit factor and expectancy are undefined for a position that has not
        exited yet. That makes it the wrong number to quote for "how much did
        this thing actually trade" — a strategy with 3 closed trades and 1 open
        one traded 4 times. Keep the two separate and report both.
        """
        return int(self.trade_count) + int(self.open_trade_count)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["start"] = str(self.start) if self.start is not None else ""
        d["end"] = str(self.end) if self.end is not None else ""
        d["warnings"] = "; ".join(self.warnings)
        d["total_trades"] = self.total_trades  # property, so asdict() misses it
        return d

    # -- presentation ------------------------------------------------------

    ROWS = [
        ("Total return", "total_return", "pct"),
        ("Buy & hold return", "buy_hold_return", "pct"),
        ("CAGR", "cagr", "pct"),
        ("Max drawdown", "max_drawdown", "pct"),
        ("Sharpe (ann.)", "sharpe", "num"),
        ("Sortino (ann.)", "sortino", "num"),
        ("Calmar", "calmar", "num"),
        ("Volatility (ann.)", "volatility", "pct"),
        ("Trades (closed)", "trade_count", "int"),
        ("Trades (incl. open)", "total_trades", "int"),
        ("Win rate", "win_rate", "pct"),
        ("Profit factor", "profit_factor", "num"),
        ("Payoff ratio", "payoff_ratio", "num"),
        ("Expectancy / trade", "expectancy", "cash"),
        ("Largest loss", "largest_loss", "cash"),
        ("Open trades at end", "open_trade_count", "int"),
        ("Unrealised PnL (open)", "unrealised_pnl", "cash"),
        ("Final equity", "final_equity", "cash"),
    ]

    @staticmethod
    def _fmt(value: Any, kind: str) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "n/a"
        if kind == "pct":
            return f"{value * 100:,.2f}%"
        if kind == "int":
            return f"{int(value):,d}"
        if kind == "cash":
            return f"{value:,.2f}"
        return f"{value:,.3f}"


def compute_metrics(
    pf: Any,
    bars_per_year: float,
    label: str = "",
    price: Optional[pd.Series] = None,
    risk_free: float = 0.0,
) -> Metrics:
    """Build a `Metrics` from a vectorbt Portfolio."""
    equity = extract_equity(pf)
    equity = equity.dropna()
    m = Metrics(label=label)

    if equity.empty:
        m.warnings.append("Empty equity curve.")
        m.significance = "no data"
        return m

    m.start, m.end, m.bars = equity.index[0], equity.index[-1], len(equity)
    m.init_cash = float(equity.iloc[0])
    m.final_equity = float(equity.iloc[-1])
    m.total_return = m.final_equity / m.init_cash - 1.0 if m.init_cash else float("nan")

    rets = equity.pct_change().dropna()
    years = max(m.bars / bars_per_year, 1e-9)

    # CAGR
    if m.init_cash > 0 and m.final_equity > 0:
        m.cagr = (m.final_equity / m.init_cash) ** (1.0 / years) - 1.0
    else:
        m.cagr = float("nan")

    # Drawdown
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    m.max_drawdown = float(-drawdown.min()) if len(drawdown) else 0.0

    # Risk-adjusted
    if len(rets) > 1 and rets.std(ddof=1) > 0:
        ann = np.sqrt(bars_per_year)
        excess = rets - (risk_free / bars_per_year)
        m.volatility = float(rets.std(ddof=1) * ann)
        m.sharpe = float(excess.mean() / rets.std(ddof=1) * ann)
        downside = rets[rets < 0]
        if len(downside) > 1 and downside.std(ddof=1) > 0:
            m.sortino = float(excess.mean() / downside.std(ddof=1) * ann)
        else:
            m.sortino = float("nan")
    else:
        m.volatility = m.sharpe = m.sortino = float("nan")

    m.calmar = float(m.cagr / m.max_drawdown) if m.max_drawdown > 0 else float("nan")

    # Trade-level. Closed and open trades are tallied separately: the stats
    # below describe completed trades, while the equity curve above includes
    # any position still open at the final bar.
    pnl, open_pnl = split_trade_pnl(extract_trades(pf))
    m.trade_count = int(len(pnl))
    m.open_trade_count = int(len(open_pnl))
    m.unrealised_pnl = float(open_pnl.sum()) if len(open_pnl) else 0.0
    if m.trade_count:
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        m.win_rate = float(len(wins) / m.trade_count)
        m.gross_win = float(wins.sum())
        m.gross_loss = float(-losses.sum())
        m.profit_factor = m.gross_win / m.gross_loss if m.gross_loss > 0 else float("inf")
        m.avg_win = float(wins.mean()) if len(wins) else 0.0
        m.avg_loss = float(losses.mean()) if len(losses) else 0.0
        m.payoff_ratio = abs(m.avg_win / m.avg_loss) if m.avg_loss else float("inf")
        m.expectancy = float(pnl.mean())
        m.largest_loss = float(pnl.min())
    else:
        m.profit_factor = float("nan")
        m.payoff_ratio = float("nan")

    if price is not None and len(price.dropna()) > 1:
        p = price.dropna()
        m.buy_hold_return = float(p.iloc[-1] / p.iloc[0] - 1.0)

    m.significance, extra = assess_significance(m)
    m.warnings.extend(extra)
    m.warnings.extend(_reconcile(m))
    return m


def _reconcile(m: Metrics) -> list[str]:
    """
    Cross-check the trade list against the equity curve.

    Realised PnL plus unrealised PnL should account for the change in equity.
    When it does not, one of the two views is wrong, and silently reporting
    both is how a broken backtest passes for a working one. This catches that.
    """
    notes: list[str] = []
    if not m.trade_count and not m.open_trade_count:
        return notes

    realised = m.gross_win - m.gross_loss
    accounted = realised + m.unrealised_pnl
    actual = m.final_equity - m.init_cash
    tolerance = max(0.005 * m.init_cash, 0.02 * abs(actual))

    if abs(accounted - actual) > tolerance:
        notes.append(
            f"Trade list and equity curve disagree: trades account for "
            f"{accounted:,.2f} (realised {realised:,.2f} + unrealised "
            f"{m.unrealised_pnl:,.2f}) but equity moved {actual:,.2f}. "
            f"Difference {accounted - actual:,.2f}. Inspect the exported trade log."
        )
    elif m.open_trade_count and abs(m.unrealised_pnl) > 0.05 * m.init_cash:
        notes.append(
            f"{m.open_trade_count} position(s) still open at the final bar, carrying "
            f"{m.unrealised_pnl:,.2f} unrealised. Win rate, profit factor and "
            f"expectancy describe closed trades only, so they will not reconcile "
            f"with total return until that position is closed."
        )
    return notes


def assess_significance(m: Metrics) -> tuple[str, list[str]]:
    """Judge whether the sample supports any conclusion at all."""
    notes: list[str] = []
    n = m.trade_count

    if n == 0:
        return "NO TRADES", ["Strategy never traded — check signals and warm-up period."]

    if n < MIN_TRADES_MEANINGFUL:
        level = "NOT MEANINGFUL"
        notes.append(
            f"Only {n} trades (< {MIN_TRADES_MEANINGFUL}). These metrics are noise. "
            f"Win-rate standard error is about {50 / np.sqrt(n):.0f} percentage points."
        )
    elif n < MIN_TRADES_CONFIDENT:
        level = "WEAK"
        notes.append(
            f"{n} trades (< {MIN_TRADES_CONFIDENT}). Treat as provisional; "
            f"win-rate standard error is about {50 / np.sqrt(n):.0f} percentage points."
        )
    else:
        level = "OK"

    if np.isfinite(m.max_drawdown) and m.max_drawdown > 0.5:
        notes.append(f"Max drawdown {m.max_drawdown:.0%} — few people can sit through this.")
    if np.isfinite(m.buy_hold_return) and np.isfinite(m.total_return):
        if m.total_return < m.buy_hold_return:
            notes.append(
                f"Underperforms buy & hold ({m.total_return:.1%} vs {m.buy_hold_return:.1%}) "
                f"— the strategy is doing work for no reward."
            )
    if np.isfinite(m.sharpe) and m.sharpe > 3:
        notes.append(
            f"Sharpe of {m.sharpe:.2f} on a simple strategy is a red flag, not a result. "
            f"Check for lookahead bias before celebrating."
        )
    return level, notes


# ------------------------------------------------------------------ formatting


def format_comparison(metrics: list[Metrics], title: str = "") -> str:
    """Render several Metrics side by side — the in-sample vs out-of-sample table."""
    if not metrics:
        return "(no results)"

    labels = [m.label or f"run {i}" for i, m in enumerate(metrics)]
    name_w = max(len(r[0]) for r in Metrics.ROWS) + 2
    col_w = max(max(len(l) for l in labels), 14) + 2

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * (name_w + col_w * len(labels)))
    lines.append("".ljust(name_w) + "".join(l.rjust(col_w) for l in labels))
    lines.append("-" * (name_w + col_w * len(labels)))

    for display, attr, kind in Metrics.ROWS:
        row = display.ljust(name_w)
        for m in metrics:
            row += Metrics._fmt(getattr(m, attr), kind).rjust(col_w)
        lines.append(row)

    lines.append("-" * (name_w + col_w * len(labels)))
    lines.append("Significance".ljust(name_w) + "".join(m.significance.rjust(col_w) for m in metrics))

    flagged = [(m.label, w) for m in metrics for w in m.warnings]
    if flagged:
        lines.append("")
        lines.append("Flags:")
        for label, warning in flagged:
            lines.append(f"  [{label}] {warning}")
    return "\n".join(lines)


def degradation_note(is_m: Metrics, oos_m: Metrics) -> str:
    """Plain-language read on how far out-of-sample fell short of in-sample."""
    if not np.isfinite(is_m.sharpe) or not np.isfinite(oos_m.sharpe):
        return "Cannot compare Sharpe (insufficient data in one of the windows)."
    if is_m.sharpe <= 0:
        return "In-sample Sharpe is non-positive — the parameter search found nothing to overfit."

    retained = oos_m.sharpe / is_m.sharpe
    head = (
        f"Out-of-sample Sharpe is {oos_m.sharpe:.2f} against {is_m.sharpe:.2f} in-sample "
        f"({retained:.0%} retained)."
    )
    if retained < 0:
        tail = "Sign flipped out of sample. The in-sample result was curve fitting."
    elif retained < 0.3:
        tail = "Most of the in-sample edge did not survive. Assume overfitting."
    elif retained < 0.6:
        tail = "Substantial decay. Normal for a real edge, but also what overfitting looks like."
    else:
        tail = "Held up reasonably. Necessary but not sufficient — one split is one sample."
    return f"{head} {tail}"
