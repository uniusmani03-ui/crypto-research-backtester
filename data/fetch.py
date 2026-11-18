"""
Market data layer: public OHLCV via CCXT, cached to Parquet.

PUBLIC DATA ONLY.
=================
This module deliberately constructs CCXT exchange objects with NO credentials.
It calls only public endpoints (`fetch_ohlcv`, `load_markets`). It does not read
API keys from the environment, does not accept them as arguments, and does not
store them. If you ever need authenticated endpoints, that is a separate concern
you should implement and review yourself.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    import ccxt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ccxt is not installed. Run setup.ps1, or: pip install -r requirements.txt"
    ) from exc

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "cache"

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Rough bars-per-year, used for annualising Sharpe etc. Crypto trades 24/7/365.
BARS_PER_YEAR = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "1h": 365 * 24,
    "2h": 365 * 12,
    "4h": 365 * 6,
    "6h": 365 * 4,
    "12h": 365 * 2,
    "1d": 365,
    "1w": 52,
}

# pandas offset alias per timeframe, handed to vectorbt as `freq`.
PANDAS_FREQ = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1D",
    "1w": "1W",
}


@dataclass(frozen=True)
class FetchSpec:
    """Everything needed to identify one OHLCV series."""

    exchange_id: str
    symbol: str
    timeframe: str

    @property
    def cache_key(self) -> str:
        safe_symbol = self.symbol.replace("/", "-").replace(":", "_")
        return f"{self.exchange_id}__{safe_symbol}__{self.timeframe}"


class OHLCVFetcher:
    """Paginating, rate-limit-aware, cached OHLCV fetcher over CCXT public data."""

    def __init__(
        self,
        exchange_id: str = "binance",
        cache_dir: Path | str = CACHE_DIR,
        max_retries: int = 5,
        request_limit: int = 1000,
    ) -> None:
        if not hasattr(ccxt, exchange_id):
            raise ValueError(
                f"Unknown exchange id {exchange_id!r}. "
                f"Try one of: binance, kraken, coinbase, bybit, okx, bitstamp."
            )
        # No apiKey / secret / password. Public endpoints only, on purpose.
        self.exchange = getattr(ccxt, exchange_id)(
            {
                "enableRateLimit": True,  # CCXT sleeps between calls for us
                "timeout": 30_000,
            }
        )
        self.exchange_id = exchange_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.request_limit = request_limit

    # ------------------------------------------------------------------ utils

    def timeframe_ms(self, timeframe: str) -> int:
        return int(self.exchange.parse_timeframe(timeframe) * 1000)

    def _cache_path(self, spec: FetchSpec) -> Path:
        return self.cache_dir / f"{spec.cache_key}.parquet"

    @staticmethod
    def _to_frame(rows: Iterable[list]) -> pd.DataFrame:
        df = pd.DataFrame(list(rows), columns=["timestamp", *OHLCV_COLUMNS])
        if df.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.astype("float64")

    # ------------------------------------------------------------- networking

    def _fetch_page(self, symbol: str, timeframe: str, since: Optional[int]) -> list:
        """One `fetch_ohlcv` call with retry/backoff on transient errors."""
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=since, limit=self.request_limit
                )
            except (
                ccxt.DDoSProtection,
                ccxt.RateLimitExceeded,
                ccxt.RequestTimeout,
                ccxt.ExchangeNotAvailable,
                ccxt.NetworkError,
            ) as exc:
                last_exc = exc
                log.warning(
                    "%s: transient error on attempt %d/%d (%s). Backing off %.1fs.",
                    self.exchange_id, attempt, self.max_retries, type(exc).__name__, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            except ccxt.BadSymbol as exc:
                raise ValueError(
                    f"{self.exchange_id} does not list {symbol!r}. "
                    f"Call OHLCVFetcher.list_symbols() to see what it does list."
                ) from exc
        raise RuntimeError(
            f"Gave up fetching {symbol} {timeframe} from {self.exchange_id} "
            f"after {self.max_retries} attempts"
        ) from last_exc

    def list_symbols(self, contains: str = "") -> list[str]:
        """Public market list. Handy when an exchange names a pair unexpectedly."""
        markets = self.exchange.load_markets()
        return sorted(s for s in markets if contains.upper() in s.upper())

    # ------------------------------------------------------------------- main

    def fetch(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: str | pd.Timestamp | None = None,
        until: str | pd.Timestamp | None = None,
        use_cache: bool = True,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV as a UTC-indexed DataFrame [open, high, low, close, volume].

        Pages forward from `since` until `until` (or now), respecting the
        exchange rate limit. Results are merged into a Parquet cache so repeat
        runs only fetch the missing tail.

        The final, possibly still-forming bar is dropped: including it is a
        subtle source of lookahead bias in intraday backtests.
        """
        spec = FetchSpec(self.exchange_id, symbol, timeframe)
        tf_ms = self.timeframe_ms(timeframe)

        start_ts = pd.Timestamp(since or "2019-01-01", tz="UTC")
        end_ts = pd.Timestamp(until, tz="UTC") if until is not None else pd.Timestamp.now(tz="UTC")

        cached = self._load_cache(spec) if (use_cache and not refresh) else None

        cursor = int(start_ts.timestamp() * 1000)
        if cached is not None and not cached.empty:
            covered_start = cached.index.min()
            covered_end = cached.index.max()
            if covered_start <= start_ts and covered_end >= end_ts - pd.Timedelta(milliseconds=tf_ms):
                log.info("Cache hit for %s (%s rows)", spec.cache_key, len(cached))
                return self._slice(cached, start_ts, end_ts)
            # Only fetch the tail we are missing.
            if covered_start <= start_ts:
                cursor = int(covered_end.timestamp() * 1000) + tf_ms

        end_ms = int(end_ts.timestamp() * 1000)
        pages: list[list] = []
        n_pages = 0

        while cursor < end_ms:
            batch = self._fetch_page(symbol, timeframe, cursor)
            if not batch:
                break
            pages.extend(batch)
            n_pages += 1
            last_ts = batch[-1][0]
            next_cursor = last_ts + tf_ms
            if next_cursor <= cursor:  # exchange refused to advance; stop rather than spin
                break
            cursor = next_cursor
            if len(batch) < self.request_limit:
                break
            if n_pages % 10 == 0:
                log.info(
                    "  ... %s %s: %d bars so far (through %s)",
                    symbol, timeframe, len(pages),
                    pd.to_datetime(last_ts, unit="ms", utc=True).date(),
                )

        fresh = self._to_frame(pages)
        combined = fresh if cached is None else self._merge(cached, fresh)

        if use_cache and not combined.empty:
            self._save_cache(spec, combined)

        out = self._slice(combined, start_ts, end_ts)
        out = self._drop_forming_bar(out, tf_ms)
        if out.empty:
            raise RuntimeError(
                f"No data returned for {symbol} {timeframe} on {self.exchange_id} "
                f"between {start_ts.date()} and {end_ts.date()}."
            )
        return out

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        if new.empty:
            return old
        if old.empty:
            return new
        combined = pd.concat([old, new])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    @staticmethod
    def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if df.empty:
            return df
        return df.loc[(df.index >= start) & (df.index <= end)].copy()

    @staticmethod
    def _drop_forming_bar(df: pd.DataFrame, tf_ms: int) -> pd.DataFrame:
        """Drop the most recent bar if its period has not closed yet."""
        if df.empty:
            return df
        now = pd.Timestamp.now(tz="UTC")
        last_close_time = df.index[-1] + pd.Timedelta(milliseconds=tf_ms)
        if last_close_time > now:
            return df.iloc[:-1].copy()
        return df

    def _load_cache(self, spec: FetchSpec) -> Optional[pd.DataFrame]:
        path = self._cache_path(spec)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                return None
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return df.sort_index()
        except Exception as exc:  # corrupt cache should never be fatal
            log.warning("Ignoring unreadable cache %s (%s)", path.name, exc)
            return None

    def _save_cache(self, spec: FetchSpec, df: pd.DataFrame) -> None:
        path = self._cache_path(spec)
        try:
            df.to_parquet(path, engine="pyarrow", compression="snappy")
            log.info("Cached %d rows -> %s", len(df), path.name)
        except Exception as exc:  # pragma: no cover
            log.warning("Could not write cache %s (%s)", path.name, exc)


def load_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since: str = "2021-01-01",
    until: str | None = None,
    exchanges: Iterable[str] = ("binance", "kraken", "coinbase", "bybit"),
    **kwargs,
) -> tuple[pd.DataFrame, str]:
    """
    Convenience wrapper: try each exchange in turn, return the first that works.

    Exchanges get geo-blocked, rename pairs, and go down for maintenance. Falling
    through a short list makes the example script robust without hiding failures
    — the exchange actually used is returned alongside the data.
    """
    errors: dict[str, str] = {}
    for exchange_id in exchanges:
        try:
            fetcher = OHLCVFetcher(exchange_id)
            df = fetcher.fetch(symbol, timeframe, since=since, until=until, **kwargs)
            return df, exchange_id
        except Exception as exc:
            errors[exchange_id] = f"{type(exc).__name__}: {exc}"
            log.warning("%s failed for %s: %s", exchange_id, symbol, exc)
    detail = "\n".join(f"  {k}: {v}" for k, v in errors.items())
    raise RuntimeError(f"Could not fetch {symbol} {timeframe} from any exchange:\n{detail}")
