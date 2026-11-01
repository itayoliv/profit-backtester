"""RSI dip buyer: a complete strategy to copy when writing a new one.

Add it with:  bt add-strategy examples/strategy_template.py --author <your name>
"""

from __future__ import annotations

import pandas as pd
from pydantic import Field, model_validator

from backtester.strategies import BaseStrategy, Signals, StrategyParams, register


class RsiDipParams(StrategyParams):
    length: int = Field(default=14, ge=2, le=500)
    """Bars in the RSI average. Smaller reacts faster and gives more, noisier signals."""
    buy_below: float = Field(default=30.0, ge=1, le=99)
    """Buy when RSI closes below this level. Lower waits for deeper dips and trades less."""
    sell_above: float = Field(default=55.0, ge=1, le=99)
    """Sell when RSI closes above this level. Higher holds longer. Must exceed buy_below."""

    @model_validator(mode="after")
    def _levels(self) -> "RsiDipParams":
        if self.buy_below >= self.sell_above:
            raise ValueError("buy_below must be strictly less than sell_above")
        return self


class RsiDip(BaseStrategy):
    name = "rsi_dip"
    description = (
        "Long only. Buys at the close of the bar where RSI first drops below buy_below, "
        "and sells at the close of the bar where RSI first rises above sell_above."
    )
    Params = RsiDipParams

    def generate_signals(self, df: pd.DataFrame, params: RsiDipParams) -> Signals:
        change = df["close"].diff()
        gain = change.clip(lower=0).ewm(alpha=1 / params.length, min_periods=params.length).mean()
        loss = (-change.clip(upper=0)).ewm(alpha=1 / params.length, min_periods=params.length).mean()
        rsi = 100 - 100 / (1 + gain / loss)
        below = rsi < params.buy_below
        above = rsi > params.sell_above
        return Signals(
            entries=below & ~below.shift(1, fill_value=False),
            exits=above & ~above.shift(1, fill_value=False),
        )


register(RsiDip())
