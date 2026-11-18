# Portfolio note

> **Personal project**: built in my own time
> **Dates:** late August to early September 2026
> **Author:** Unais Usmani

This is a research and backtesting environment for systematic crypto strategies:
- CCXT data with a Parquet cache
- VectorBT backtests with fees and slippage
- ATR-based position sizing
- train/test split and walk-forward validation
- a paper-trading execution engine with a separate risk module

**27 pytest tests pass** (`python -m pytest tests`). Live trading is deliberately not implemented, and no API keys are read or stored. See `README.md` for full details.
