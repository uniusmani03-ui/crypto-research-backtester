"""
Environment self-check. Run this first, and read what it prints.

    python verify_env.py

It proves four things, in order, and stops at the first failure:

  1. Python version is inside vectorbt's supported range.
  2. Every required package imports, and reports its actual version.
  3. Public market data can be fetched over the network with no credentials.
  4. A full costed, ATR-sized backtest runs end to end and produces metrics.

If step 4 prints numbers, the stack works. If it does not, the error it prints
is the real one — do not skip ahead to run_example.py.
"""

from __future__ import annotations

import platform
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = "  [ok]  "
FAIL = "  [FAIL]"


def header(n: int, text: str) -> None:
    print(f"\n{n}. {text}\n{'-' * 70}")


def check_python() -> bool:
    header(1, "PYTHON VERSION")
    v = sys.version_info
    print(f"  {platform.python_version()} ({platform.python_implementation()}) on {platform.system()}")
    print(f"  executable: {sys.executable}")
    ok = (3, 11) <= (v.major, v.minor) < (3, 15)
    if ok:
        print(f"{PASS} inside vectorbt's supported range (>=3.11, <3.15)")
    else:
        print(f"{FAIL} vectorbt 1.1.0 requires Python >=3.11,<3.15. Recreate the venv.")
    if sys.prefix == sys.base_prefix:
        print("  note: this does not look like a virtual environment.")
    return ok


def check_imports() -> bool:
    header(2, "PACKAGE IMPORTS")
    required = ["numpy", "pandas", "pyarrow", "matplotlib", "ccxt", "vectorbt"]
    optional = ["polymarket", "numba", "plotly"]
    ok = True

    for name in required:
        try:
            mod = __import__(name)
            print(f"{PASS} {name:<14} {getattr(mod, '__version__', 'unknown')}")
        except Exception as exc:
            ok = False
            print(f"{FAIL} {name:<14} {type(exc).__name__}: {exc}")

    for name in optional:
        try:
            mod = __import__(name)
            print(f"  [opt] {name:<14} {getattr(mod, '__version__', 'unknown')}")
        except Exception:
            print(f"  [opt] {name:<14} not installed (fine)")

    for name in ["data", "strategies", "backtest", "live"]:
        try:
            __import__(name)
            print(f"{PASS} local package '{name}'")
        except Exception as exc:
            ok = False
            print(f"{FAIL} local package '{name}': {type(exc).__name__}: {exc}")
            print("        (are you running this from the project root?)")
    return ok


def check_public_data():
    header(3, "PUBLIC MARKET DATA (no API keys)")
    from data import OHLCVFetcher

    last_error = None
    for exchange_id in ("binance", "kraken", "coinbase", "bybit"):
        try:
            f = OHLCVFetcher(exchange_id)
            creds = [k for k in ("apiKey", "secret", "password") if getattr(f.exchange, k, None)]
            if creds:
                print(f"{FAIL} {exchange_id}: credentials are set on the exchange object ({creds}).")
                return None
            df = f.fetch("BTC/USDT", "1h", since="2024-01-01", until="2024-02-01", use_cache=False)
            print(f"{PASS} {exchange_id}: {len(df)} bars, no credentials attached")
            print(f"        {df.index[0]} -> {df.index[-1]}")
            print(f"        last close {df['close'].iloc[-1]:,.2f}")
            return df
        except Exception as exc:
            last_error = exc
            print(f"  [skip] {exchange_id}: {type(exc).__name__}: {exc}")

    print(f"{FAIL} no exchange reachable. Last error: {last_error}")
    print("        Check your network/proxy. Some exchanges are geo-blocked.")
    return None


def check_backtest(df) -> bool:
    header(4, "END-TO-END BACKTEST")
    if df is None:
        print(f"{FAIL} skipped — no data from step 3.")
        return False
    try:
        from backtest import BacktestConfig, CostModel, RiskModel, run_strategy
        from strategies import EmaCrossover

        cfg = BacktestConfig(
            cost=CostModel(fee_pct=0.001, slippage_pct=0.0005),
            risk=RiskModel(risk_per_trade=0.01, atr_mult=2.0),
            init_cash=10_000.0,
            timeframe="1h",
        )
        strategy = EmaCrossover(fast=10, slow=30)
        _, m = run_strategy(df, strategy, cfg, label="verify")

        print(f"{PASS} simulated {m.bars} bars with {cfg.cost.describe()}")
        print(f"        strategy      {strategy.describe()}")
        print(f"        total return  {m.total_return:+.2%}")
        print(f"        max drawdown  {m.max_drawdown:.2%}")
        print(f"        sharpe        {m.sharpe:.2f}")
        print(f"        trades        {m.trade_count}")
        print(f"        significance  {m.significance}")
        print("        (a one-month window is far too short to mean anything — "
              "this only proves the plumbing works)")
        return True
    except Exception:
        print(f"{FAIL} backtest raised:")
        traceback.print_exc()
        return False


def check_live_stub() -> bool:
    header(5, "LIVE EXECUTION IS STUBBED")
    try:
        from live import LiveExecutor

        LiveExecutor()
        print(f"{FAIL} LiveExecutor constructed. It should have raised NotImplementedError.")
        return False
    except NotImplementedError:
        print(f"{PASS} LiveExecutor raises NotImplementedError, as intended.")
        return True
    except Exception as exc:
        print(f"{FAIL} unexpected error: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    print("=" * 70)
    print("trading-bot environment check")
    print("=" * 70)

    if not check_python():
        return 1
    if not check_imports():
        print("\nFAILED at imports. Run setup.ps1, or activate the venv first.")
        return 1

    df = check_public_data()
    ok_bt = check_backtest(df)
    ok_stub = check_live_stub()

    print("\n" + "=" * 70)
    if df is not None and ok_bt and ok_stub:
        print("ALL CHECKS PASSED. Next:  python scripts/run_example.py")
        return 0
    print("SOME CHECKS FAILED — see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
