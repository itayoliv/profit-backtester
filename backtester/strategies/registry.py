"""Name to strategy lookup."""

from __future__ import annotations

from .base import BaseStrategy

_REGISTRY: dict[str, BaseStrategy] = {}


def register(strategy: BaseStrategy) -> BaseStrategy:
    if not strategy.name:
        raise ValueError("strategy must define a name")
    _REGISTRY[strategy.name] = strategy
    return strategy


def _load_builtins() -> None:
    if _REGISTRY:
        return
    from . import sma_cross  # noqa: F401  (import registers)


def get_strategy(name: str) -> BaseStrategy:
    _load_builtins()
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown strategy {name!r}; registered: {known}") from None


def list_strategies() -> dict[str, BaseStrategy]:
    _load_builtins()
    return dict(sorted(_REGISTRY.items()))
