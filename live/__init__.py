"""
Live execution boundary. Stub only — see executor.py.

This package intentionally contains no working order-placement code and no
credential handling.
"""

from .executor import LiveExecutor, OrderIntent

__all__ = ["LiveExecutor", "OrderIntent"]
