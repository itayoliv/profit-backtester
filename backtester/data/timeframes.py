"""Supported timeframes and their download chunk sizes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

DAY = timedelta(days=1)


@dataclass(frozen=True)
class TimeframeSpec:
    name: str
    amount: int
    unit: str
    chunk: timedelta
    intraday: bool
    bars_per_year: float
    """Used to annualise Sharpe identically across engines."""

    def to_alpaca(self) -> Any:
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        unit = {"Min": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour, "Day": TimeFrameUnit.Day}[
            self.unit
        ]
        return TimeFrame(amount=self.amount, unit=unit)

    @property
    def pandas_freq(self) -> str:
        suffix = {"Min": "min", "Hour": "h", "Day": "D"}[self.unit]
        return f"{self.amount}{suffix}"


# Regular-session bar counts: 390 one-minute bars per day, 6.5 hours, 252 trading days.
# 1Hour counts 7: hourly bars are built from 15Min bars aligned to the 09:30 open, so
# they open at 09:30 through 15:30 and the last one is a half hour (see AGENTS.md).
TIMEFRAMES: dict[str, TimeframeSpec] = {
    "1Min": TimeframeSpec("1Min", 1, "Min", 30 * DAY, True, 252 * 390),
    "5Min": TimeframeSpec("5Min", 5, "Min", 90 * DAY, True, 252 * 78),
    "15Min": TimeframeSpec("15Min", 15, "Min", 180 * DAY, True, 252 * 26),
    "1Hour": TimeframeSpec("1Hour", 1, "Hour", 730 * DAY, True, 252 * 7),
    "1Day": TimeframeSpec("1Day", 1, "Day", 3650 * DAY, False, 252),
}

TIMEFRAME_NAMES = tuple(TIMEFRAMES)


def get_timeframe(name: str) -> TimeframeSpec:
    try:
        return TIMEFRAMES[name]
    except KeyError:
        raise KeyError(
            f"unknown timeframe {name!r}; expected one of {', '.join(TIMEFRAME_NAMES)}"
        ) from None
