"""Opt-in execution: session-reset indicators, protective stops, next-open fills."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from backtester.data.timeframes import get_timeframe
from backtester.pipeline.engines import get_engine
from backtester.pipeline.indicators import signals_by_session
from backtester.pipeline.request import CapitalConfig, RuleConfig, StopConfig, parse_request
from backtester.pipeline.rules import RuleEngine, validate_trades
from backtester.pipeline.session import add_session_columns
from backtester.strategies.base import Signals
from backtester.strategies.sma_cross import SmaCross
from helpers import base_request

SPEC = get_timeframe("1Hour")


def _sessions(closes: list[list[float]]) -> pd.DataFrame:
    """Hourly regular-session bars, one inner list per New York session.

    A wild pre-market bar is included each day so a strategy that accidentally
    read extended hours would not match the regular-session sample.
    """
    stamps: list[pd.Timestamp] = []
    prices: list[float] = []
    for day_i, session in enumerate(closes):
        day = pd.Timestamp("2023-03-06", tz="America/New_York") + pd.Timedelta(days=day_i)
        # 08:00 ET is outside the regular session.
        stamps.append(day + pd.Timedelta(hours=8))
        prices.append(1.0)
        for offset, price in enumerate(session):
            stamps.append(day + pd.Timedelta(hours=10 + offset))
            prices.append(price)
    index = pd.DatetimeIndex(stamps).tz_convert("UTC")
    index.name = "timestamp"
    close = pd.Series(prices, index=index, dtype="float64")
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1e5,
            "trade_count": 100.0,
            "vwap": close,
        },
        index=index,
    )
    frame = add_session_columns(frame, SPEC)
    frame["tradable"] = frame["in_session"]
    return frame


def _signals(frame: pd.DataFrame, *, rth_only: bool) -> Signals:
    """Mirror the pipeline: indicators never see extended hours; the flag restarts
    each regular session."""
    trade = frame.loc[frame["in_session"].astype(bool)]
    strategy = SmaCross()
    params = strategy.parse_params({"fast": 2, "slow": 4})
    if rth_only:
        return signals_by_session(trade, strategy, params)
    return strategy.generate_signals(trade, params)


def test_indicators_rth_only_changes_entry_time():
    # Day 1 is flat, so the continuous slow average is already warm on day 2.
    # Rebuilding it per session delays the first cross until day 2 has four bars.
    frame = _sessions(
        [
            [10, 10, 10, 10, 10, 10],
            [10, 11, 14, 14, 14, 14],
        ]
    )
    off = _signals(frame, rth_only=False)
    on = _signals(frame, rth_only=True)

    assert off.entries.any() and on.entries.any()
    assert off.entries.idxmax() != on.entries.idxmax()

    def _entry_time(signals: Signals) -> pd.Timestamp:
        rules = RuleConfig()
        result = get_engine("vectorbt").run(
            frame=frame.loc[frame["in_session"]],
            signals=RuleEngine(rules).apply(frame.loc[frame["in_session"]], signals),
            capital=CapitalConfig(initial_cash=10_000, slippage_bps=0),
            rules=rules,
            bars_per_year=SPEC.bars_per_year,
            symbol="TEST",
        )
        assert len(result.trades) >= 1
        return pd.Timestamp(result.trades["entry_time"].iloc[0])

    assert _entry_time(off) != _entry_time(on)


def _stop_frame() -> tuple[pd.DataFrame, Signals]:
    index = pd.date_range("2024-01-02", periods=8, freq="D", tz="UTC", name="timestamp")
    close = pd.Series([100, 100, 101, 102, 103, 104, 105, 106], index=index, dtype="float64")
    low = close - 1.0
    low.iloc[3] = 80.0  # trades through a stop 5 dollars under the 100 entry
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": low,
            "close": close,
            "volume": 1e5,
            "is_last_bar_of_day": False,
        },
        index=index,
    )
    signals = Signals(
        entries=pd.Series([False, True, False, False, False, False, False, False], index=index),
        exits=pd.Series([False, False, False, False, False, False, True, False], index=index),
    )
    return frame, signals


@pytest.mark.parametrize("engine_name", ["vectorbt", "backtesting"])
@pytest.mark.parametrize(
    "stop",
    [
        StopConfig(type="fixed", dollars=5),
        StopConfig(type="atr", atr_mult=3, atr_period=2),
    ],
)
def test_tight_stop_exits_more_trades_than_no_stop(engine_name, stop):
    frame, signals = _stop_frame()
    capital = CapitalConfig(
        initial_cash=100_000,
        sizing="fixed_shares",
        size_value=10,
        slippage_bps=0,
        commission=0,
    )

    def _run(stop):
        return get_engine(engine_name).run(
            frame=frame,
            signals=signals,
            capital=capital,
            rules=RuleConfig(),
            bars_per_year=252,
            symbol="TEST",
            stop=stop,
        )

    plain = _run(None)
    stopped = _run(stop)

    plain_stops = int(plain.trades["exit_reason"].eq("stop").sum()) if len(plain.trades) else 0
    stopped_stops = int(stopped.trades["exit_reason"].eq("stop").sum())
    assert stopped_stops > plain_stops
    assert stopped.trades["exit_time"].iloc[0] < plain.trades["exit_time"].iloc[0]


def _intraday() -> pd.DataFrame:
    stamps: list[pd.Timestamp] = []
    for day in pd.bdate_range("2023-03-06", periods=4, tz="America/New_York"):
        stamps += [day + pd.Timedelta(hours=h) for h in range(8, 17)]
    index = pd.DatetimeIndex(stamps).tz_convert("UTC")
    index.name = "timestamp"
    close = pd.Series(np.linspace(100, 120, len(index)), index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1e5,
        },
        index=index,
    )
    frame = add_session_columns(frame, SPEC)
    frame["tradable"] = frame["in_session"]
    return frame


@pytest.mark.parametrize("engine_name", ["vectorbt", "backtesting"])
def test_next_open_with_no_overnight_stays_inside_the_session(engine_name):
    frame = _intraday()
    rules = RuleConfig(no_overnight=True)
    signals = RuleEngine(rules).apply(
        frame,
        Signals(
            entries=pd.Series(True, index=frame.index),
            exits=pd.Series(False, index=frame.index),
        ),
    )
    result = get_engine(engine_name).run(
        frame=frame,
        signals=signals,
        capital=CapitalConfig(initial_cash=100_000, slippage_bps=0, commission=0),
        rules=rules,
        bars_per_year=SPEC.bars_per_year,
        symbol="TEST",
        fill="next_open",
    )

    assert len(result.trades) > 0
    assert validate_trades(result.trades, rules) == 0
    entry = pd.to_datetime(result.trades["entry_time"], utc=True).dt.tz_convert("America/New_York")
    exit_ = pd.to_datetime(result.trades["exit_time"], utc=True).dt.tz_convert("America/New_York")
    assert (entry.dt.date.to_numpy() == exit_.dt.date.to_numpy()).all()
    # 16:00 ET is the first out-of-session bar. A next-open exit of the 15:00
    # signal would land there, or on the next morning, and neither is allowed.
    assert all(ts.time() < time(16, 0) for ts in exit_)
    assert set(result.trades["exit_reason"]) <= {"signal", "eod_close", "end_of_data", "stop"}
    assert result.trades["exit_reason"].eq("eod_close").any()


def test_omitted_execution_fields_keep_the_historical_defaults():
    request = parse_request(base_request())
    assert request.fill == "close"
    assert request.stop is None
    assert request.rules.indicators_rth_only is False
