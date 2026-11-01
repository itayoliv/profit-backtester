"""Strategies an agent writes and adds with ``bt add-strategy``.

Each lives in its own ``<name>.py`` in the custom strategies directory, next to a
``manifest.json`` that records who added it, when, and the SHA-256 of the exact file
that passed the checks. A file only loads while its hash still matches, so the author
on record is always the author of the code that runs. Python files that are not in
the manifest are never imported.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from ..config import custom_strategies_dir
from ..data.timeframes import get_timeframe
from ..jsonio import iso, read_json, utcnow, write_json
from ..pipeline.session import add_session_columns
from .base import BaseStrategy, Signals, StrategyParams
from .registry import builtin_names, collect

MANIFEST = "manifest.json"
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
AUTHOR_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\- ]{0,39}$")
_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


class StrategyCheckError(Exception):
    def __init__(self, details: list[str]) -> None:
        super().__init__("; ".join(details))
        self.details = details


@dataclass
class CustomStrategies:
    strategies: dict[str, BaseStrategy] = field(default_factory=dict)
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    unregistered_files: list[str] = field(default_factory=list)


_loaded: dict[tuple[str, str], BaseStrategy] = {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(directory: Path | None = None) -> dict[str, dict[str, Any]]:
    directory = directory or custom_strategies_dir()
    return (read_json(directory / MANIFEST, default={}) or {}).get("strategies", {})


def import_strategy(path: Path) -> BaseStrategy:
    """Import one strategy file and return the single strategy it registers."""
    module_name = f"bt_custom_{path.stem.strip('.')}_{_sha256(path)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise StrategyCheckError([f"{path.name} is not an importable Python file"])
    module = importlib.util.module_from_spec(spec)
    # Pydantic resolves the Params annotations through sys.modules.
    sys.modules[module_name] = module
    try:
        found = collect(lambda: spec.loader.exec_module(module))
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise StrategyCheckError([f"importing {path.name} failed: {_last_line(exc)}"]) from exc
    if len(found) != 1:
        raise StrategyCheckError(
            [f"{path.name} must call register() exactly once; it registered {len(found)}"]
        )
    return found[0]


def load_custom() -> CustomStrategies:
    directory = custom_strategies_dir()
    out = CustomStrategies(entries=read_manifest(directory))
    for name, entry in out.entries.items():
        path = directory / str(entry.get("file"))
        if not path.is_file():
            out.errors[name] = f"{entry.get('file')} is missing"
            continue
        digest = _sha256(path)
        if digest != entry.get("sha256"):
            out.errors[name] = (
                f"{entry.get('file')} changed after it was added; "
                "run bt add-strategy on it again to re-check and re-attribute it"
            )
            continue
        key = (str(path), digest)
        if key not in _loaded:
            try:
                _loaded[key] = import_strategy(path)
            except StrategyCheckError as exc:
                out.errors[name] = str(exc)
                continue
        out.strategies[name] = _loaded[key]
    if directory.is_dir():
        known = {str(entry.get("file")) for entry in out.entries.values()}
        out.unregistered_files = sorted(
            p.name for p in directory.glob("*.py") if p.name not in known and not p.name.startswith(".")
        )
    return out


# --- checks ---------------------------------------------------------------


def _sample_frame(timeframe: str, seed: int) -> pd.DataFrame:
    """Synthetic bars shaped like the pipeline's, with the session columns added."""
    spec = get_timeframe(timeframe)
    if spec.intraday:
        index = pd.date_range("2024-01-02", "2024-02-29", freq="15min", tz="UTC", name="timestamp")
        index = index[index.tz_convert("America/New_York").weekday < 5]
    else:
        index = pd.bdate_range("2022-01-03", periods=500, tz="UTC", name="timestamp")
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 1.0, len(index)) + 2.0 * np.sin(np.arange(len(index)) / 15.0)
    close = np.maximum(100.0 + np.cumsum(steps) * 0.5, 1.0)
    frame = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, len(index))),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(100_000, 500_000, len(index)).astype("float64"),
            "trade_count": 1000.0,
            "vwap": close,
        },
        index=index,
    )
    frame = add_session_columns(frame, spec)
    frame["tradable"] = frame["in_session"]
    return frame


def _signal_series(signals: Signals, index: pd.Index, label: str) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for name in ("entries", "exits", "short_entries", "short_exits"):
        series = getattr(signals, name)
        if series is None:
            if name in ("entries", "exits"):
                raise StrategyCheckError([f"{label}: Signals.{name} is None"])
            continue
        if not isinstance(series, pd.Series):
            raise StrategyCheckError([f"{label}: Signals.{name} must be a pandas Series"])
        if not series.index.equals(index):
            raise StrategyCheckError([f"{label}: Signals.{name} must use the input frame's index"])
        out[name] = series.fillna(False).astype(bool)
    return out


def check_strategy(strategy: Any) -> list[str]:
    """Fail on anything that would break or mislead a backtest; return soft warnings."""
    if not isinstance(strategy, BaseStrategy):
        raise StrategyCheckError(["the registered object must be a BaseStrategy instance"])
    problems: list[str] = []
    if not NAME_PATTERN.match(strategy.name or ""):
        problems.append("name must be lowercase snake_case, 2 to 48 characters, starting with a letter")
    if not (strategy.description or "").strip():
        problems.append("description must say when it enters and when it exits")
    if not (isinstance(strategy.Params, type) and issubclass(strategy.Params, StrategyParams)):
        problems.append("Params must subclass StrategyParams")
    else:
        for name, info in strategy.Params.model_fields.items():
            if not (info.description or "").strip():
                problems.append(f"parameter {name!r} needs a docstring saying what it does to trading")
    if problems:
        raise StrategyCheckError(problems)

    try:
        params = strategy.parse_params({})
    except Exception as exc:
        raise StrategyCheckError([f"the default params do not validate: {_last_line(exc)}"]) from exc

    warnings: list[str] = []
    entered = False
    for seed, timeframe in enumerate(("1Day", "15Min")):
        frame = _sample_frame(timeframe, seed)
        prices = frame[_PRICE_COLUMNS].copy()
        full = _run(strategy, frame, params, timeframe)
        if not frame[_PRICE_COLUMNS].equals(prices):
            raise StrategyCheckError([f"{timeframe}: generate_signals must not modify the input bars"])
        entered = entered or bool(full["entries"].any())

        for last in _cut_points(full, len(frame)):
            prefix = frame.iloc[: last + 1].copy()
            partial = _run(strategy, prefix, params, f"{timeframe} prefix")
            for name, series in partial.items():
                if not series.equals(full[name].iloc[: last + 1]):
                    raise StrategyCheckError(
                        [
                            f"{timeframe}: {name} up to {frame.index[last]} change when later bars "
                            "are added, so the strategy looks into the future"
                        ]
                    )
    if not entered:
        warnings.append("default params produced no entries on synthetic daily or 15Min bars")
    return warnings


def _cut_points(full: dict[str, pd.Series], length: int, sample: int = 25) -> list[int]:
    """Where to truncate the bars: a spread across the frame, plus bars that signal.

    A strategy that peeks one bar ahead only gives itself away on a bar whose signal
    the peek decides, so cutting just after signalling bars is what catches it.
    """
    points = set(np.linspace(length // 4, length - 2, 6).astype(int).tolist())
    signalled = np.flatnonzero(np.logical_or.reduce([s.to_numpy() for s in full.values()]))
    if len(signalled):
        picks = np.linspace(0, len(signalled) - 1, min(sample, len(signalled))).astype(int)
        points.update(int(i) for i in signalled[picks])
    return sorted(p for p in points if 0 < p < length - 1)


def _run(strategy: BaseStrategy, frame: pd.DataFrame, params: StrategyParams, label: str) -> dict[str, pd.Series]:
    try:
        signals = strategy.generate_signals(frame, params)
    except Exception as exc:
        raise StrategyCheckError([f"{label}: generate_signals raised {_last_line(exc)}"]) from exc
    if not isinstance(signals, Signals):
        raise StrategyCheckError([f"{label}: generate_signals must return Signals"])
    return _signal_series(signals, frame.index, label)


def _last_line(exc: BaseException) -> str:
    return traceback.format_exception_only(type(exc), exc)[-1].strip()


# --- add and remove -------------------------------------------------------


def add_strategy(source: Path, author: str, replace: bool = False) -> dict[str, Any]:
    """Check ``source`` and install it under its strategy name, attributed to ``author``."""
    if not AUTHOR_PATTERN.match(author or ""):
        raise StrategyCheckError(["--author must be 1 to 40 letters, digits, spaces, '.', '_' or '-'"])
    if not source.is_file():
        raise StrategyCheckError([f"no such file: {source}"])
    code = source.read_bytes()

    directory = custom_strategies_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".manifest.lock")):
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".incoming_", suffix=".py")
        staged = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(code)
            strategy = import_strategy(staged)
            warnings = check_strategy(strategy)

            name = strategy.name
            if name in builtin_names():
                raise StrategyCheckError([f"{name!r} is a built-in strategy; pick another name"])
            manifest = read_manifest(directory)
            previous = manifest.get(name)
            if previous is not None and not replace:
                raise StrategyCheckError(
                    [f"{name!r} already exists (author {previous.get('author')}); pass --replace to overwrite it"]
                )

            target = directory / f"{name}.py"
            os.replace(staged, target)
            entry = {
                "file": target.name,
                "author": author,
                "added_at": iso(utcnow()),
                "sha256": hashlib.sha256(code).hexdigest(),
                "description": strategy.description,
            }
            manifest[name] = entry
            write_json(directory / MANIFEST, {"strategies": dict(sorted(manifest.items()))})
        finally:
            staged.unlink(missing_ok=True)

    return {
        "name": name,
        **entry,
        "path": str(target.resolve()),
        "replaced": previous,
        "warnings": warnings,
    }


def remove_strategy(name: str) -> dict[str, Any]:
    directory = custom_strategies_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".manifest.lock")):
        manifest = read_manifest(directory)
        entry = manifest.pop(name, None)
        if entry is None:
            reason = "is built in and cannot be removed" if name in builtin_names() else "was never added"
            raise StrategyCheckError([f"{name!r} {reason}"])
        write_json(directory / MANIFEST, {"strategies": manifest})
        (directory / str(entry.get("file"))).unlink(missing_ok=True)
    return {"removed": name, **entry}
