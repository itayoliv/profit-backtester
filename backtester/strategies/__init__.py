"""Strategies: one signal definition, consumed by both engines."""

from .base import BaseStrategy, Signals, StrategyParams
from .registry import get_strategy, list_strategies, register

__all__ = [
    "BaseStrategy",
    "Signals",
    "StrategyParams",
    "get_strategy",
    "list_strategies",
    "register",
]
