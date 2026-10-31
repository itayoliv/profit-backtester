"""Global trading rules, enforced before simulation and re-checked afterwards."""

from __future__ import annotations

import pandas as pd

from ..strategies.base import Signals
from .request import RuleConfig
from .session import EXCHANGE_TZ


class RuleEngine:
    """Applies the tradable mask and the global rules to a strategy's signals."""

    def __init__(self, rules: RuleConfig) -> None:
        self.rules = rules

    def apply(self, frame: pd.DataFrame, signals: Signals) -> Signals:
        signals = signals.aligned_to(frame.index)
        tradable = frame["tradable"].astype(bool)
        last_bar = frame["is_last_bar_of_day"].astype(bool)

        entries = signals.entries & tradable
        exits = signals.exits
        short_entries = (
            signals.short_entries & tradable if signals.short_entries is not None else None
        )
        short_exits = signals.short_exits

        if self.rules.no_overnight:
            # Close on the final session bar, and never open a position there: the
            # entry would have to be unwound on the same bar.
            entries = entries & ~last_bar
            exits = exits | last_bar
            if short_entries is not None:
                short_entries = short_entries & ~last_bar
            if short_exits is not None:
                short_exits = short_exits | last_bar

        cap = self.rules.max_trades_per_day
        if cap is not None:
            entries = _cap_per_day(entries, frame["session_date"], cap)
            if short_entries is not None:
                short_entries = _cap_per_day(short_entries, frame["session_date"], cap)

        return Signals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )


def _cap_per_day(entries: pd.Series, session_date: pd.Series, cap: int) -> pd.Series:
    """Keep only the first ``cap`` entry signals of each session.

    Applied to signals rather than filled trades, so the realised trade count can come
    in under the cap but never over it.
    """
    flagged = entries[entries]
    if flagged.empty:
        return entries
    order = flagged.groupby(session_date.loc[flagged.index].to_numpy()).cumcount()
    blocked = flagged.index[order.to_numpy() >= cap]
    if len(blocked) == 0:
        return entries
    out = entries.copy()
    out.loc[blocked] = False
    return out


def validate_trades(trades: pd.DataFrame, rules: RuleConfig) -> int:
    """Independently re-check the realised trade log. Returns the violation count.

    Anything other than zero means an engine failed to honour a rule, which is
    reported in ``summary.json`` and in the report's Notes rather than failing the job.
    """
    if trades is None or trades.empty:
        return 0

    violations = 0
    entry = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(EXCHANGE_TZ)
    exit_ = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert(EXCHANGE_TZ)

    if rules.max_trades_per_day is not None:
        per_day = entry.dt.date.value_counts()
        over = (per_day - rules.max_trades_per_day).clip(lower=0)
        violations += int(over.sum())

    if rules.no_overnight:
        held_over = exit_.dt.date.to_numpy() != entry.dt.date.to_numpy()
        violations += int(held_over.sum())

    return violations
