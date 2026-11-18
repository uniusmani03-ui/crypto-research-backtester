"""
Execution stack.

Build order is trust order: the risk governor exists before anything that can
place an order, and the engine cannot reach a broker without passing through
it.

  risk.py     Circuit breakers. Model-independent, fails closed, latches.
  broker.py   Broker interface + an in-process simulator. No credentials.
  engine.py   Intent -> risk gate -> broker, with idempotency and reconciliation.

There is still no live order execution in this project. `AlpacaPaperBroker` is
a documented stub, and `live/executor.py` raises NotImplementedError.
"""

from .broker import (
    AlpacaPaperBroker,
    Broker,
    Order,
    OrderStatus,
    Position,
    Side,
    SimulatedBroker,
)
from .engine import ExecutionEngine, make_client_order_id
from .risk import (
    RiskDecision,
    RiskGovernor,
    RiskHalted,
    RiskLimits,
    RiskState,
    utc_now,
)

__all__ = [
    # risk
    "RiskGovernor",
    "RiskLimits",
    "RiskState",
    "RiskDecision",
    "RiskHalted",
    "utc_now",
    # broker
    "Broker",
    "SimulatedBroker",
    "AlpacaPaperBroker",
    "Order",
    "OrderStatus",
    "Position",
    "Side",
    # engine
    "ExecutionEngine",
    "make_client_order_id",
]
