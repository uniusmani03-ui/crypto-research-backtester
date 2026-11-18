"""
Optional: Polymarket public market data.

Uses `polymarket-client` (the official SDK, https://github.com/Polymarket/py-sdk),
NOT the archived `py-clob-client`.

PUBLIC DATA ONLY. This module uses `PublicClient`, which requires no credentials.
The SDK also has authenticated trading clients — they are deliberately not
imported, wrapped, or referenced here.

This is a read-only convenience layer for research. Prediction-market prices are
probabilities in [0, 1] with discrete resolution events; they do not behave like
continuous asset prices, so do not naively feed them into the OHLCV backtest
harness and expect the numbers to mean what they usually mean.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_IMPORT_HINT = (
    "polymarket-client is not installed (it is optional). Install with:\n"
    "    pip install polymarket-client"
)


def _client():
    try:
        from polymarket import PublicClient  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_IMPORT_HINT) from exc
    return PublicClient()


def get_market(url: str) -> Any:
    """
    Fetch one public market by its polymarket.com URL.

    Example:
        get_market("https://polymarket.com/event/some-event-slug")

    Returns the SDK's `Market` model. The SDK is on a 0.x line and its models
    change between minor releases, so this returns the raw object rather than
    a normalised DataFrame that would silently rot.
    """
    with _client() as client:
        return client.get_market(url=url)


def is_available() -> bool:
    """True if the optional SDK is importable."""
    try:
        import polymarket  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not is_available():
        print(_IMPORT_HINT)
    else:
        import polymarket

        print(f"polymarket-client {getattr(polymarket, '__version__', 'unknown')} importable.")
        print("Pass a polymarket.com event URL to get_market() to pull public market data.")
