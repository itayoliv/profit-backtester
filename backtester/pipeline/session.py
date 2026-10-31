"""Exchange-session columns derived from UTC bars."""

from __future__ import annotations

from datetime import time

import pandas as pd

from ..data.timeframes import TimeframeSpec

EXCHANGE_TZ = "America/New_York"
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)


def add_session_columns(frame: pd.DataFrame, spec: TimeframeSpec) -> pd.DataFrame:
    """Add ``session_date``, ``in_session`` and ``is_last_bar_of_day``.

    Bars are labelled by their open time, so an intraday bar counts as in-session when
    its open falls in [09:30, 16:00). For 1Hour that excludes the 09:00 bar, which
    straddles the open, making 10:00 the first tradable bar and 15:00 the last.
    """
    out = frame.copy()
    local = out.index.tz_convert(EXCHANGE_TZ)
    out["session_date"] = pd.Index(local.date, name="session_date")

    if spec.intraday:
        local_time = pd.Index(local.time)
        in_session = pd.Series(
            [SESSION_OPEN <= t < SESSION_CLOSE for t in local_time], index=out.index
        )
    else:
        in_session = pd.Series(True, index=out.index)
    out["in_session"] = in_session

    last = pd.Series(False, index=out.index)
    session_bars = out.index[in_session.to_numpy()]
    if len(session_bars):
        closes = (
            pd.Series(session_bars, index=session_bars)
            .groupby(out.loc[session_bars, "session_date"].to_numpy())
            .max()
        )
        last.loc[pd.DatetimeIndex(closes.to_numpy())] = True
    out["is_last_bar_of_day"] = last
    return out
