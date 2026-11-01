"""MACD crossover: long when the MACD line crosses above its signal, flat on the reverse.

Mirrors the trading rules of the TradingView script "AI MACD Crossover". Entries and
exits are decided on the bar close, which is also where both engines fill. Optional
stop and target percentages exit on a later bar when that bar's low or high touches
the level; the fill is still that bar's close, the same as every other exit here.

Chart boxes, the ATR drawing fallback, and the Pine share-count formula are not
part of the signal. Position size stays on the request's ``capital`` block.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from .base import BaseStrategy, Signals, StrategyParams
from .registry import register


class MacdCrossParams(StrategyParams):
    fast: int = Field(default=12, ge=1, le=500)
    """Bars in the fast EMA of the close. MACD line = fast EMA minus slow EMA. Smaller
    reacts sooner and trades more often. Must be strictly less than slow."""
    slow: int = Field(default=26, ge=2, le=1000)
    """Bars in the slow EMA of the close. Larger smooths the MACD line, so crosses come
    later and less often."""
    signal_len: int = Field(default=9, ge=1, le=500)
    """Bars in the EMA of the MACD line itself (the signal line). Entries and exits are
    MACD crossing this line; smaller means earlier crosses and more whipsaws."""
    stop_pct: float = Field(default=0.0, ge=0, le=100)
    """Stop loss, in percent below the entry close (2 means 2%). Checked from the bar
    after entry against the bar's low; the exit fills at that bar's close, so the loss
    can exceed this. 0 disables the stop."""
    target_pct: float = Field(default=0.0, ge=0, le=1000)
    """Profit target, in percent above the entry close (5 means 5%). Checked from the
    bar after entry against the bar's high; the exit fills at that bar's close. 0
    disables the target."""
    use_window: bool = False
    """When true, new entries are only allowed on session dates between window_start
    and window_end. Exits are never blocked. When false both dates are ignored."""
    window_start: date = date(2021, 1, 1)
    """First session date (YYYY-MM-DD, New York) that may open a trade. Only used when
    use_window is true."""
    window_end: date = date(2099, 12, 31)
    """Last session date (YYYY-MM-DD, New York) that may open a trade. Only used when
    use_window is true."""

    @model_validator(mode="after")
    def _ranges(self) -> "MacdCrossParams":
        if self.fast >= self.slow:
            raise ValueError("fast must be strictly less than slow")
        if self.window_start > self.window_end:
            raise ValueError("window_start must be on or before window_end")
        return self


def _ema(values: np.ndarray, length: int) -> np.ndarray:
    """TradingView ``ta.ema``: seed with an SMA, then ``alpha = 2 / (length + 1)``."""
    out = np.full(len(values), np.nan)
    if length <= 0 or len(values) < length:
        return out
    start = 0
    while start < len(values) and not np.isfinite(values[start]):
        start += 1
    seed_at = start + length - 1
    if seed_at >= len(values):
        return out
    window = values[start : seed_at + 1]
    if not np.isfinite(window).all():
        return out
    alpha = 2.0 / (length + 1.0)
    out[seed_at] = float(window.mean())
    for i in range(seed_at + 1, len(values)):
        price = values[i]
        prev = out[i - 1]
        if np.isfinite(price) and np.isfinite(prev):
            out[i] = alpha * price + (1.0 - alpha) * prev
    return out


def _bar_dates(df: pd.DataFrame) -> np.ndarray:
    if "session_date" in df.columns:
        return pd.to_datetime(pd.Series(df["session_date"])).dt.date.to_numpy()
    index = df.index
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert("America/New_York")
    return np.array([ts.date() for ts in index])


class MacdCross(BaseStrategy):
    name = "macd_cross"
    description = (
        "Long only. Buys at the close of the bar where the MACD line crosses above its "
        "signal line, and sells when it crosses back below, or earlier if the optional "
        "stop_pct or target_pct is touched."
    )
    Params = MacdCrossParams

    def generate_signals(self, df: pd.DataFrame, params: MacdCrossParams) -> Signals:
        close = df["close"].to_numpy(dtype="float64")
        high = df["high"].to_numpy(dtype="float64")
        low = df["low"].to_numpy(dtype="float64")
        macd = _ema(close, params.fast) - _ema(close, params.slow)
        signal = _ema(macd, params.signal_len)

        prev_macd = np.roll(macd, 1)
        prev_signal = np.roll(signal, 1)
        prev_macd[0] = np.nan
        prev_signal[0] = np.nan
        ready = np.isfinite(macd) & np.isfinite(signal) & np.isfinite(prev_macd) & np.isfinite(prev_signal)
        cross_up = ready & (macd > signal) & (prev_macd <= prev_signal)
        cross_down = ready & (macd < signal) & (prev_macd >= prev_signal)

        dates = _bar_dates(df)
        in_window = np.ones(len(df), dtype=bool)
        if params.use_window:
            in_window = (dates >= params.window_start) & (dates <= params.window_end)

        entries = np.zeros(len(df), dtype=bool)
        exits = np.zeros(len(df), dtype=bool)
        in_position = False
        entry_price = 0.0
        stop_at = params.stop_pct / 100.0
        target_at = params.target_pct / 100.0

        for i in range(len(df)):
            if not in_position:
                if cross_up[i] and in_window[i]:
                    entries[i] = True
                    in_position = True
                    entry_price = float(close[i])
                continue
            # The entry fills at this bar's close, so its low and high are already past.
            hit_stop = stop_at > 0 and np.isfinite(low[i]) and low[i] <= entry_price * (1.0 - stop_at)
            hit_target = (
                target_at > 0 and np.isfinite(high[i]) and high[i] >= entry_price * (1.0 + target_at)
            )
            if cross_down[i] or hit_stop or hit_target:
                exits[i] = True
                in_position = False

        index = df.index
        return Signals(
            entries=pd.Series(entries, index=index),
            exits=pd.Series(exits, index=index),
        )


register(MacdCross())
