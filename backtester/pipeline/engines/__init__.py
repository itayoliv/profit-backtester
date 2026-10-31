"""Engine adapters. Both consume the same signals and emit the same ``EngineResult``."""

from __future__ import annotations

from .base import ENGINE_NAMES, EngineResult, TRADE_COLUMNS


def get_engine(name: str):
    if name == "vectorbt":
        from .vectorbt_engine import VectorbtEngine

        return VectorbtEngine()
    if name == "backtesting":
        from .backtesting_engine import BacktestingEngine

        return BacktestingEngine()
    raise KeyError(f"unknown engine {name!r}; expected one of {', '.join(ENGINE_NAMES)}")


__all__ = ["ENGINE_NAMES", "EngineResult", "TRADE_COLUMNS", "get_engine"]
