"""Indicator samples and the ATR used by protective stops.

Extended-hours bars are removed before either sample, which is what the pipeline
has always done, so they cannot move an average or fill a trade. The
``indicators_rth_only`` flag then decides whether the remaining regular-session
bars are one continuous series (the default, historical behavior) or one series
per session.
"""

from __future__ import annotations

import pandas as pd

from ..strategies.base import Signals, StrategyParams


def average_true_range(frame: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR. Uses only the current bar and earlier closes."""
    close = frame["close"].astype("float64")
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    prev = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def signals_by_session(frame: pd.DataFrame, strategy, params: StrategyParams) -> Signals:
    """Run ``strategy`` on each regular session separately and stitch the signals.

    A session is one New York trading day of the bars passed in. The strategy
    cannot see the previous session, so a crossover can print at a different bar
    than it does on the continuous series.
    """
    if frame.empty or "session_date" not in frame.columns:
        return strategy.generate_signals(frame, params)

    produced: list[Signals] = []
    groups = frame.groupby(frame["session_date"].to_numpy(), sort=False)
    for _, session in groups:
        produced.append(strategy.generate_signals(session, params).aligned_to(session.index))
    if not produced:
        return strategy.generate_signals(frame, params)

    saw_short = any(item.short_entries is not None for item in produced)

    def _take(session_signals: Signals, name: str) -> pd.Series:
        series = getattr(session_signals, name)
        if series is None:
            return pd.Series(False, index=session_signals.entries.index)
        return series

    return Signals(
        entries=pd.concat([item.entries for item in produced]),
        exits=pd.concat([item.exits for item in produced]),
        short_entries=pd.concat([_take(item, "short_entries") for item in produced])
        if saw_short
        else None,
        short_exits=pd.concat([_take(item, "short_exits") for item in produced])
        if saw_short
        else None,
    ).aligned_to(frame.index)
