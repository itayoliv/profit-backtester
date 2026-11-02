"""Fill timing and protective-stop levels shared by both engines.

``fill="close"`` is the historical path: the order fills on the signal bar's close.
``fill="next_open"`` fills at the next bar's open. When ``no_overnight`` is set, an
exit that would be pushed onto the next session stays on the last regular-session
bar and fills at that bar's close, so the position is flat before the session ends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategies.base import Signals
from .indicators import average_true_range
from .request import Fill, RuleConfig, StopConfig


def stop_distance(frame: pd.DataFrame, stop: StopConfig) -> pd.Series:
    """Price distance of the protective stop, known on the decision bar."""
    if stop.type == "fixed":
        return pd.Series(float(stop.dollars), index=frame.index, dtype="float64")
    return (
        float(stop.atr_mult) * average_true_range(frame, int(stop.atr_period))
    ).astype("float64")


def distance_on_entry_bar(
    frame: pd.DataFrame, stop: StopConfig | None, fill: Fill
) -> pd.Series | None:
    """Distance aligned to the bar the trade's ``entry_time`` will carry.

    A next-open fill is stamped on the bar after the signal, so the distance
    decided on the signal bar is shifted onto that fill bar. A close fill is
    stamped on the signal bar and keeps the unshifted distance.
    """
    if stop is None:
        return None
    distance = stop_distance(frame, stop)
    if fill == "next_open":
        distance = distance.shift(1)
    return distance


def order_stop_prices(
    frame: pd.DataFrame, stop: StopConfig | None, fill: Fill
) -> tuple[pd.Series, pd.Series] | None:
    """Absolute long and short stop prices, indexed by the bar that places the order.

    backtesting.py attaches these to the entry order. A long stop has to sit
    below that bar's close or the order is rejected, so a next-open stop that
    would not is pulled back to the signal close minus the distance.
    """
    if stop is None:
        return None
    distance = stop_distance(frame, stop)
    close = frame["close"].astype("float64")
    if fill == "next_open":
        anchor = frame["open"].astype("float64").shift(-1)
    else:
        anchor = close

    long_level = anchor - distance
    short_level = anchor + distance
    if fill == "next_open":
        long_bad = ~np.isfinite(long_level) | (long_level >= close) | (long_level <= 0)
        long_level = long_level.where(~long_bad, close - distance)
        short_bad = ~np.isfinite(short_level) | (short_level <= close)
        short_level = short_level.where(~short_bad, close + distance)

    long_level = long_level.where((long_level > 0) & (long_level < close))
    short_level = short_level.where((short_level > close) & np.isfinite(short_level))
    return long_level.astype("float64"), short_level.astype("float64")


def resolve_fill(
    signals: Signals, frame: pd.DataFrame, fill: Fill, rules: RuleConfig
) -> tuple[Signals, pd.Series | None]:
    """Move next-open orders onto the fill bar. Close fills are returned unchanged.

    The returned price series is the fill price for vectorbt (``None`` means the
    close). backtesting.py does its own delay with ``trade_on_close`` and should
    keep the unshifted signals; it only needs ``flatten_at_close``.
    """
    if fill == "close":
        return signals, None
    if fill != "next_open":
        raise ValueError(f"unknown fill {fill!r}")

    last = frame["is_last_bar_of_day"].astype(bool)
    follows_last = last.shift(1, fill_value=False)

    def _shift(flags: pd.Series) -> pd.Series:
        return flags.shift(1, fill_value=False).astype(bool)

    entries = _shift(signals.entries)
    exits = _shift(signals.exits)
    if rules.no_overnight:
        # A last-bar exit must not be the first order of the next session.
        exits = (exits & ~follows_last) | (signals.exits.astype(bool) & last)
        entries = entries & ~last & ~follows_last

    price = frame["open"].astype("float64").copy()
    if rules.no_overnight:
        carried = _shift(signals.exits) & ~follows_last
        flatten_at_close = last & exits & ~carried
        price = price.where(~flatten_at_close, frame["close"].astype("float64"))

    short_entries = short_exits = None
    if signals.short_entries is not None:
        short_entries = _shift(signals.short_entries)
        short_source = (
            signals.short_exits
            if signals.short_exits is not None
            else pd.Series(False, index=frame.index)
        )
        short_exits = _shift(short_source)
        if rules.no_overnight:
            short_exits = (short_exits & ~follows_last) | (short_source.astype(bool) & last)
            short_entries = short_entries & ~last & ~follows_last

    return (
        Signals(
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            short_entries=short_entries.astype(bool) if short_entries is not None else None,
            short_exits=short_exits.astype(bool) if short_exits is not None else None,
        ),
        price,
    )


def flatten_at_close_mask(frame: pd.DataFrame, fill: Fill, rules: RuleConfig) -> pd.Series:
    """Bars where an open position must be closed at this bar's close.

    Used by backtesting.py, which otherwise fills a market exit at the next open.
    """
    if fill == "next_open" and rules.no_overnight:
        return frame["is_last_bar_of_day"].astype(bool)
    return pd.Series(False, index=frame.index)


def stop_fraction(frame: pd.DataFrame, distance: pd.Series | None, fill: Fill) -> pd.Series | None:
    """vectorbt ``sl_stop`` fraction of the pre-slippage fill price.

    ``distance`` is already aligned to the entry bar. A fraction of 0.05 is five
    percent of that fill, which is ``distance`` dollars when the fill is the
    price used here and slippage is zero.
    """
    if distance is None:
        return None
    exec_price = frame["close"] if fill == "close" else frame["open"]
    fraction = distance / exec_price.astype("float64").replace(0, np.nan)
    return fraction.where((fraction > 0) & (fraction < 1))


def stops_hit(
    entry_time: pd.Series,
    side: pd.Series,
    entry_price: pd.Series,
    exit_price: pd.Series,
    distance: pd.Series | None,
) -> list[bool]:
    """True when the raw fill is at or through the protective stop.

    A long is stopped when the exit is at or below ``entry - distance``. A gap
    through the stop (a worse price) still counts. Strategy exits on the right
    side of the stop do not.
    """
    count = len(entry_time)
    if distance is None or count == 0:
        return [False] * count

    levels = distance.copy()
    levels.index = pd.to_datetime(levels.index, utc=True)
    looked = levels.reindex(pd.to_datetime(entry_time, utc=True))
    flags: list[bool] = []
    for side_i, entry_px, exit_px, dist in zip(
        side.tolist(),
        pd.to_numeric(entry_price, errors="coerce").tolist(),
        pd.to_numeric(exit_price, errors="coerce").tolist(),
        looked.tolist(),
        strict=True,
    ):
        flags.append(_through_stop(side_i, entry_px, exit_px, dist))
    return flags


def _through_stop(side, entry_px, exit_px, dist) -> bool:
    if dist is None or entry_px is None or exit_px is None:
        return False
    try:
        dist_f = float(dist)
        entry_f = float(entry_px)
        exit_f = float(exit_px)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(dist_f) or dist_f <= 0 or not np.isfinite(entry_f) or not np.isfinite(exit_f):
        return False
    tolerance = max(1e-6, abs(entry_f) * 1e-5)
    if str(side).lower().startswith("short"):
        return exit_f >= entry_f + dist_f - tolerance
    return exit_f <= entry_f - dist_f + tolerance
