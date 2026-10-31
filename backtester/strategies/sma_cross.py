"""Moving-average crossover: the reference strategy."""

from __future__ import annotations

import pandas as pd
from pydantic import Field, model_validator

from .base import BaseStrategy, Signals, StrategyParams
from .registry import register


class SmaCrossParams(StrategyParams):
    fast: int = Field(default=10, ge=1, le=1000)
    slow: int = Field(default=50, ge=2, le=2000)

    @model_validator(mode="after")
    def _fast_below_slow(self) -> "SmaCrossParams":
        if self.fast >= self.slow:
            raise ValueError("fast must be strictly less than slow")
        return self


class SmaCross(BaseStrategy):
    name = "sma_cross"
    Params = SmaCrossParams

    def generate_signals(self, df: pd.DataFrame, params: SmaCrossParams) -> Signals:
        close = df["close"]
        fast = close.rolling(params.fast, min_periods=params.fast).mean()
        slow = close.rolling(params.slow, min_periods=params.slow).mean()
        above = fast > slow
        crossed_up = above & ~above.shift(1, fill_value=False)
        crossed_down = ~above & above.shift(1, fill_value=False)
        warm = slow.notna() & fast.notna()
        return Signals(entries=crossed_up & warm, exits=crossed_down & warm)


register(SmaCross())
