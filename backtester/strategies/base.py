"""The strategy interface shared by the vectorbt and backtesting.py engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, ConfigDict


class StrategyParams(BaseModel):
    """Base for per-strategy parameter models. Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")


@dataclass
class Signals:
    """Boolean series aligned to the bar index.

    Short series are only honoured when ``capital.allow_short`` is set.
    """

    entries: pd.Series
    exits: pd.Series
    short_entries: pd.Series | None = None
    short_exits: pd.Series | None = None

    def aligned_to(self, index: pd.Index) -> "Signals":
        def fix(series: pd.Series | None) -> pd.Series | None:
            if series is None:
                return None
            return series.reindex(index).fillna(False).astype(bool)

        return Signals(
            entries=fix(self.entries),
            exits=fix(self.exits),
            short_entries=fix(self.short_entries),
            short_exits=fix(self.short_exits),
        )


@runtime_checkable
class Strategy(Protocol):
    name: str
    Params: type[StrategyParams]

    def generate_signals(self, df: pd.DataFrame, params: StrategyParams) -> Signals: ...


class BaseStrategy:
    """Convenience base: subclasses set ``name``/``Params`` and implement signals.

    ``df`` carries OHLCV plus the pipeline columns ``tradable``, ``session_date`` and
    ``is_last_bar_of_day``. Signals must not peek at future bars.
    """

    name: ClassVar[str] = ""
    Params: ClassVar[type[StrategyParams]] = StrategyParams

    def generate_signals(self, df: pd.DataFrame, params: StrategyParams) -> Signals:
        raise NotImplementedError

    def parse_params(self, raw: dict | None) -> StrategyParams:
        return self.Params(**(raw or {}))

    def params_schema(self) -> dict:
        return self.Params.model_json_schema()
