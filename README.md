# trading-bot — a research and backtesting environment

A small, coherent stack for researching systematic trading strategies on public
market data.

## What this does

- Fetches historical OHLCV from public exchange endpoints via CCXT, with
  pagination, rate-limit handling, and a Parquet cache.
- Backtests strategies with VectorBT, applying realistic fees and slippage to
  every simulation.
- Sizes positions by ATR with a fixed-fractional risk-per-trade budget, and
  places a matching ATR stop.
- Validates with a chronological train/test split and walk-forward analysis, so
  a parameter sweep cannot quietly grade its own homework.
- Reports max drawdown, Sharpe, Sortino, Calmar, win rate, profit factor, payoff
  ratio, expectancy and trade count — and flags when the trade count is too low
  for any of it to mean anything.

## What this does not do

**It does not place trades.** There is no live execution, no exchange
credentials, no broker integration, no paper-trading loop, and no scheduler.

`live/executor.py` exists only to mark the boundary. Every method raises
`NotImplementedError`, and its docstring lists what implementing it actually
entails — key management, order lifecycle, reconciliation, idempotency, kill
switches. That list is not boilerplate. The failure modes of live trading are
overwhelmingly operational, and none of them show up in a backtest.

Nothing here reads, requests, or stores API keys. The data layer constructs CCXT
exchange objects with no credentials and calls public endpoints only.
`verify_env.py` asserts this explicitly.

---

## Status: verified working

`verify_env.py` passes end to end on Windows 11 / Python 3.13.14, with real data
fetched from a public exchange and a real costed backtest executed.

Resolved environment:

```
vectorbt   1.1.0      ccxt    4.5.76     numba   0.67.0
pandas     3.0.5      numpy   2.5.2      plotly  6.9.0
pyarrow   25.0.1      matplotlib 3.11.1
```

The verification backtest — EMA(10, 30) on 745 hourly BTC/USDT bars from
Binance, January 2024, at 30 bps round-trip cost — returned -4.37%, max drawdown
7.30%, Sharpe -2.05, over 17 trades, flagged `NOT MEANINGFUL`. That is a
one-month window and the flag is correct: it proves the plumbing works, nothing
more. A losing result here is a good sign, not a bad one. See
`requirements.lock.txt` for the exact resolved set.

Two arguments were flagged as risky before the first run and both were accepted
by vectorbt 1.1.0 without complaint, so the full sizing path is live:
`size_type="percent"` and the per-bar `sl_stop` array in `backtest/harness.py`.
If a future upgrade breaks either, the harness degrades and emits a loud
`UserWarning` naming which fallback tier it used. **Degraded results are not
comparable to a full run** — fix the argument rather than living with it.

Three pins exist to work around real incompatibilities, each commented in
`requirements.txt`:

- `pandas>=3.0.3` — required by vectorbt 1.1.0; do not cap below 3.0.
- `plotly<7.0` — vectorbt's chart template references the `scattermapbox`
  trace, which plotly removed in 7.0. With plotly 7 installed, `import vectorbt`
  raises before any project code runs. Remove this pin once vectorbt updates.
- Python `>=3.11,<3.15` — vectorbt's own constraint.

`setup.ps1` is deliberately pure ASCII. Windows PowerShell 5.1 reads `.ps1`
files as ANSI, and a mis-decoded em dash becomes a character the parser treats
as a string delimiter, which breaks the script in a way the error message points
at the wrong line for. Keep it ASCII.

---

## Setup

Requires Python 3.11–3.14 (vectorbt 1.1.0's own constraint; 3.12 or 3.13 is the
safest choice, since compiled dependencies get wheels for those first).

```powershell
cd C:\Users\uni_2\Documents\Claude\trading-bot
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

That finds a suitable Python, creates `.venv`, installs `requirements.txt`,
writes `requirements.lock.txt` with the exact resolved versions, and runs
`verify_env.py`.

Add `-WithOptional` for the Polymarket public-data client and JupyterLab.

Manually, if you prefer:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python verify_env.py
```

## Running it

```powershell
.\.venv\Scripts\Activate.ps1

python verify_env.py                 # environment self-check — run this first
python scripts\run_example.py        # end-to-end: fetch, backtest, walk-forward, report

python scripts\run_example.py --strategy rsi_mean_reversion --timeframe 1d
python scripts\run_example.py --symbol ETH/USDT --since 2020-01-01 --splits 6
python scripts\run_example.py --fee-bps 15 --slippage-bps 25   # pessimistic costs
```

`run_example.py` prints four sections: the data it fetched, a plain train/test
split, a walk-forward validation with in-sample and out-of-sample side by side,
and a cost-sensitivity scan. It writes CSVs and an equity-curve plot to
`reports/`.

Run scripts from the project root — the packages are imported by name.

## Paper trading (simulated broker, no credentials)

```powershell
.\.venv\Scripts\python.exe tests\test_risk.py         # 13 tests
.\.venv\Scripts\python.exe tests\test_execution.py    # 15 tests

.\.venv\Scripts\python.exe scripts\run_paper.py --reset
.\.venv\Scripts\python.exe scripts\run_paper.py --mode live --poll-seconds 60
```

Every bar goes through the full path: strategy signal → ATR allocation → risk
governor → broker → persisted state. The only simulated part is the broker.

**The kill switch is a file.** Create one named `HALT` in the state directory
(printed at startup) and everything stops on the next check — from another
terminal, while it's running, without touching the process.

**Paper numbers will not match `run_example.py`, and that is expected.** The
backtester is vectorised, fills at the bar close, and applies intrabar ATR
stops. The paper runner walks bar by bar, sizes from live equity, and has no
intrabar stop. Two fill models, two answers, neither of them the truth. Judge
strategies with `run_example.py`; judge the plumbing with `run_paper.py`.

### The execution stack

| Module | What it does |
|---|---|
| `execution/risk.py` | Circuit breakers. Knows nothing about strategies — sees equity, exposure, trade count, proposed order. Fails closed. Daily loss resets at UTC midnight; drawdown and consecutive-loss breaches **latch** until `reset("I have investigated")`. State written atomically so peak equity survives a crash. |
| `execution/broker.py` | `Broker` interface plus `SimulatedBroker`. Long-only, market orders. Rejects what it can't afford rather than going silently negative. `AlpacaPaperBroker` is a documented stub. |
| `execution/engine.py` | Intent → risk gate → broker. No bypass path. Deterministic `client_order_id` per bar, so a crash-and-replay can't double a position. Reconciles against the broker every sync and halts on divergence. Exits are never blocked — a breaker that traps you in a position is a bug, not safety. |

Still no live execution. `AlpacaPaperBroker` refuses to construct, and
`live/executor.py` raises `NotImplementedError`. Wiring a real broker means
first modelling asynchronous and partial fills, which the engine does not yet
express — it assumes immediate complete fills, true of the simulator and false
of every real venue.

## Layout

```
trading-bot/
├── data/
│   ├── fetch.py             CCXT public OHLCV: pagination, rate limits, Parquet cache
│   ├── polymarket_data.py   optional, public Polymarket data, read-only
│   └── cache/               generated Parquet files (gitignored)
├── strategies/
│   ├── base.py              Strategy interface — signals only, no sizing or costs
│   ├── indicators.py        EMA, RSI, ATR, Donchian, in plain pandas
│   ├── ema_crossover.py         EXAMPLE — trend following
│   ├── rsi_mean_reversion.py    EXAMPLE — mean reversion
│   └── donchian_breakout.py     EXAMPLE — breakout
├── backtest/
│   ├── costs.py             fee + slippage model, with presets
│   ├── sizing.py            ATR fixed-fractional position sizing
│   ├── metrics.py           metrics, significance flags, IS/OOS comparison tables
│   └── harness.py           vectorbt simulation, train/test split, walk-forward
├── live/
│   └── executor.py          STUB. Raises NotImplementedError. Read the docstring.
├── scripts/
│   └── run_example.py       runnable end-to-end example
├── reports/                 generated output (gitignored)
├── verify_env.py            environment self-check
├── setup.ps1                venv + install + lock + verify
├── requirements.txt         core, with an explicit pinning policy
└── requirements-optional.txt
```

### The example strategies are examples

EMA crossover, RSI mean reversion, and Donchian breakout are **reference
implementations, not recommendations**. Each file says so at the top and lists
its own specific failure modes.

They are there because they are short enough to read in full, and because they
have genuinely different trade profiles — trend following, mean reversion, and
breakout fail in opposite market regimes. Running all three through the same
harness on the same data is the fastest way to internalise that a backtest
measures a strategy *and a period*, not a strategy alone.

None of them is expected to be profitable after costs. Dual moving-average
crossover on liquid crypto has been public knowledge for decades.

---

## What was deliberately left out, and why

You asked to combine ten repositories. Combining four competing backtesting
engines is not more power, it is four incompatible data models, four sets of
assumptions about fills, and no single source of truth for a result. This stack
is CCXT for data, VectorBT for backtesting, and nothing else that overlaps.

Each exclusion is reversible. If one turns out to matter for what you actually
want to do, it can be added on its own terms.

| Excluded | Why |
|---|---|
| **Backtrader** | Effectively unmaintained; the original author stepped away years ago and upstream activity never recovered. Event-driven and slow relative to vectorised alternatives. Direct competitor to VectorBT — including both means every strategy needs two implementations that will drift apart. |
| **NautilusTrader** | Genuinely good, and heavier than this problem: a Rust-core, event-driven, production trading platform aimed at live deployment with nanosecond-resolution order books. Overlaps VectorBT for backtesting while adding a much steeper learning curve. Worth revisiting *if* you later want live execution with serious infrastructure — at which point it replaces this stack rather than joining it. |
| **Lumibot** | Another full backtest-plus-live framework, competing directly with VectorBT and oriented toward broker integrations you explicitly do not want yet. |
| **Freqtrade** | Not a library — a standalone crypto trading bot application with its own config format, CLI, strategy interface, database and web UI. You do not import it, you run it. Bolting it onto a Python research repo means maintaining two disconnected worlds. It is a reasonable *alternative* to this whole project, not a component of it. |
| **Hummingbot** | Same category: a standalone market-making and arbitrage application. Its problem domain — inventory management, spread quoting, exchange connectivity — barely intersects strategy research. |
| **FinRL** | A research artifact: reinforcement learning for trading, published alongside papers. Deep RL on financial data is notoriously prone to overfitting the training regime, and the library's value is in reproducing its papers, not in production research. Adding it invites you to trust a black box before you can evaluate a moving-average crossover. |
| **TradingAgents** | An LLM-multi-agent trading research project. Interesting, entirely unproven, and not infrastructure. Nothing in it makes your backtests more trustworthy. |
| **py-clob-client** | Archived and superseded. Replaced here by `polymarket-client` (the official `Polymarket/py-sdk`), and included only as optional, public, read-only data. |

**Included:** CCXT (market data), VectorBT 1.1.0 (backtesting engine),
pandas/numpy/pyarrow (data), matplotlib (plots), `polymarket-client` (optional,
public data only).

A note on the one that is closest to a real trade-off: NautilusTrader is a
serious piece of engineering and the right answer for a different question. The
question this repo answers is "does this idea survive honest costs and
out-of-sample data", and for that, a fast vectorised sweep beats an accurate
event-driven simulator, because you will run thousands of them.

---

## Design decisions worth knowing about

**Signals are lagged one bar.** A signal computed from bar *t*'s close is acted
on at bar *t+1*. Backtests that enter at the close of the bar that generated the
signal are trading on information they did not have at the time. The harness
applies this centrally (`BacktestConfig.signal_lag`), so no strategy can forget.

**Costs are never optional.** Fees and slippage are passed to every simulation.
Defaults are 10 bps commission and 5 bps slippage *per side* — 30 bps round trip.
Check your own venue: this varies by an order of magnitude. `CostModel` has
presets, and `run_example.py` runs a sensitivity scan from zero to 4x, because
"at what cost level does the edge disappear" is a better question than "what does
it earn at my favourite assumption".

**Position size and stop come from the same risk model.** Size is
`risk_per_trade * price / (atr_mult * ATR)` as a fraction of equity, capped at
100% (no leverage), and vectorbt receives a matching `sl_stop`. ATR sizing
without the corresponding stop is not a risk model. Because size is a fraction
of equity rather than a fixed unit count, the risk budget compounds with the
account.

**The last, still-forming bar is dropped.** Including a partial bar is a quiet
source of lookahead in intraday backtests.

**Walk-forward is built in, not bolted on.** `walk_forward()` selects parameters
inside each training window and scores them on the next, untouched window, then
stitches the out-of-sample periods into one curve. In-sample and out-of-sample
are printed side by side because the gap between them is the actual finding.

**Metrics are computed in plain pandas** from the equity curve and trade list,
not read off vectorbt's stats table — so they are version-independent and you can
see exactly what each number means. `Portfolio.stats()` is still there if you
want a second opinion; comparing the two is a decent sanity check.

---

## Why a good backtest is not a good strategy

This is the part that matters. A backtest is a measurement of the past under a
set of assumptions. Treating it as a prediction is the single most expensive
mistake in this field, and the tooling above is designed to make it harder — not
impossible.

**Overfitting.** Search enough parameter combinations and one of them will look
excellent by chance alone. Test 100 combinations at random and you expect the
best one to look good even if none has any edge; you have not found a strategy,
you have found the luckiest sample. This is why parameter selection here only
ever sees training data, and why the out-of-sample column is the only one worth
reading. It is also why the parameter grids are deliberately small. A related and
subtler version: *you* are the overfitter. Every time you look at a result,
change something, and look again, you leak information from the test set into
your choices. The test set is only clean the first time you use it.

**Lookahead bias.** Using information that was not available at decision time.
The obvious form — entering on the bar that generated the signal — is handled by
the one-bar lag. The subtle forms are not automatically caught: survivorship bias
in your symbol list (you only backtest coins that still exist), restated or
back-adjusted data, indicators that peek via centred windows, and using an
exchange's current fee schedule for a period when it was different. If a simple
strategy produces a Sharpe above 3, look for the leak before celebrating. The
metrics module flags this for you.

**Costs.** Fees, slippage, spread, and market impact. Beginners omit them and get
fantasy equity curves. Worse, costs interact with strategy design: a strategy
that trades 10x more often needs 10x the gross edge to break even, so cost
assumptions silently determine which parameters "win" a sweep. Slippage is the
one people get most wrong, and breakout strategies suffer most — by definition
you are buying when price is moving fastest against you. Model costs
pessimistically; if the edge only exists at optimistic assumptions, it is not an
edge.

**Regime change.** Markets are not stationary. A strategy fitted on 2020–2021
crypto learned the behaviour of a zero-rate bull market with retail inflows. That
regime ended. Trend following works until it whipsaws; mean reversion works until
it does not revert. Walk-forward helps because it forces the strategy to trade
periods it was not fitted on, but every fold is still drawn from the same handful
of years and the same asset. Out-of-sample in a backtest is not the same as
out-of-sample in the future.

**Sample size.** Ten trades tell you nothing. With N trades, the standard error
of the win rate is roughly `0.5/sqrt(N)` — at 25 trades that is ±10 percentage
points, wide enough that a 55% win rate and a 45% win rate are indistinguishable.
The metrics report flags anything under 30 trades as not meaningful and anything
under 100 as weak. Believe the flag. A beautiful equity curve built from 12
trades is a beautiful coincidence.

**Execution reality.** Even a genuinely predictive signal has to survive partial
fills, outages, latency, funding rates, borrow costs, minimum order sizes, and
your own behaviour during a drawdown. A backtest models none of this. That
distance is exactly why live execution is a stub in this repository rather than
a convenience function.

The useful posture is not "my backtest says this works" but "my backtest failed
to kill this idea, on data it had not seen, at costs I did not flatter". That is
a much weaker claim, and it is the strongest one a backtest can support.

---

## Disclaimer

Research and educational software. Not financial advice. No part of this is a
recommendation to trade anything. Historical results, including out-of-sample
ones, do not predict future returns. If you eventually risk money, risk only what
you can afford to lose entirely.
