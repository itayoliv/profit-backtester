"""Filters produce a boolean ``tradable`` mask; they never drop rows.

Dropping rows would corrupt indicator warm-up, so filters only block *entries*.
"""

from __future__ import annotations

import pandas as pd

from ..data.manager import DataManager
from .request import FilterConfig, RegimeConfig
from .session import EXCHANGE_TZ


def weekday_mask(index: pd.DatetimeIndex, weekdays: list[int] | None) -> pd.Series:
    if not weekdays:
        return pd.Series(True, index=index)
    local = index.tz_convert(EXCHANGE_TZ)
    allowed = set(weekdays)
    return pd.Series([d in allowed for d in local.dayofweek], index=index)


def regime_mask(
    index: pd.DatetimeIndex,
    config: RegimeConfig,
    data_manager: DataManager,
    start,
    end,
) -> pd.Series:
    """True where the reference symbol sits on the requested side of its moving average.

    The reference series is fetched through the same cache and forward-filled onto the
    traded symbol's index, so a daily regime can gate intraday bars.
    """
    # Pull extra history so the moving average is warm at the start of the window.
    lookback = pd.Timedelta(days=config.period * 2 + 10)
    reference = data_manager.get_bars(
        config.symbol, config.timeframe, pd.Timestamp(start) - lookback, end
    ).frame["close"]

    if config.indicator == "sma":
        line = reference.rolling(config.period, min_periods=config.period).mean()
    else:
        line = reference.ewm(span=config.period, min_periods=config.period).mean()

    ok = reference > line if config.condition == "above" else reference < line
    # Shift by one bar: the regime is only known once the reference bar has closed.
    aligned = ok.shift(1).reindex(reference.index.union(index)).ffill().reindex(index)
    return aligned.fillna(False).astype(bool)


def build_tradable_mask(
    frame: pd.DataFrame,
    config: FilterConfig,
    data_manager: DataManager | None,
    start,
    end,
) -> tuple[pd.Series, dict[str, int]]:
    """Combine session, weekday and regime masks. Returns the mask and bar counts."""
    index = frame.index
    mask = frame["in_session"].astype(bool)
    counts = {"bars": int(len(index)), "in_session": int(mask.sum())}

    weekdays = weekday_mask(index, config.weekdays)
    mask &= weekdays
    counts["after_weekdays"] = int(mask.sum())

    if config.regime is not None:
        if data_manager is None:
            raise ValueError("a regime filter needs a DataManager to fetch the reference symbol")
        mask &= regime_mask(index, config.regime, data_manager, start, end)
    counts["tradable"] = int(mask.sum())
    return mask, counts
