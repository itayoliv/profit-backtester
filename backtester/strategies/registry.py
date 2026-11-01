"""Name to strategy lookup, covering the built-ins and those added with ``bt add-strategy``."""

from __future__ import annotations

import threading
from typing import Any, Callable

from .base import BaseStrategy

_REGISTRY: dict[str, BaseStrategy] = {}
_capture: list[BaseStrategy] | None = None
_capture_lock = threading.RLock()


def register(strategy: BaseStrategy) -> BaseStrategy:
    if not strategy.name:
        raise ValueError("strategy must define a name")
    if _capture is not None:
        _capture.append(strategy)
    else:
        _REGISTRY[strategy.name] = strategy
    return strategy


def collect(load: Callable[[], Any]) -> list[BaseStrategy]:
    """Run ``load`` and return what it registered, without adding it to the built-ins."""
    global _capture
    with _capture_lock:
        previous, _capture = _capture, []
        try:
            load()
            return _capture
        finally:
            _capture = previous


def _load_builtins() -> None:
    from . import macd_cross, sma_cross  # noqa: F401  (import registers)


def builtin_names() -> set[str]:
    _load_builtins()
    return set(_REGISTRY)


def get_strategy(name: str) -> BaseStrategy:
    _load_builtins()
    if name in _REGISTRY:
        return _REGISTRY[name]
    from .custom import load_custom

    custom = load_custom()
    if name in custom.strategies:
        return custom.strategies[name]
    if name in custom.errors:
        raise KeyError(f"strategy {name!r} is not loadable: {custom.errors[name]}")
    known = ", ".join(sorted({*_REGISTRY, *custom.strategies})) or "none"
    raise KeyError(f"unknown strategy {name!r}; registered: {known}")


def list_strategies() -> dict[str, BaseStrategy]:
    _load_builtins()
    from .custom import load_custom

    return dict(sorted({**load_custom().strategies, **_REGISTRY}.items()))


def strategy_origin(name: str) -> dict[str, Any] | None:
    """Who wrote a strategy: ``{"origin": "builtin"}``, or the ``bt add-strategy`` record."""
    if name in builtin_names():
        return {"origin": "builtin", "author": None}
    from .custom import load_custom

    entry = load_custom().entries.get(name)
    if entry is None:
        return None
    return {
        "origin": "custom",
        "author": entry.get("author"),
        "added_at": entry.get("added_at"),
        "file": entry.get("file"),
        "sha256": entry.get("sha256"),
    }
