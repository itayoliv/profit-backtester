"""Rule enforcement and the independent trade-log validator, on synthetic bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.data.timeframes import get_timeframe
from backtester.pipeline.engines import get_engine
from backtester.pipeline.request import CapitalConfig, RuleConfig
from backtester.pipeline.rules import RuleEngine, validate_trades
from backtester.pipeline.session import add_session_columns
from backtester.strategies.base import Signals

SPEC = get_timeframe("1Hour")


@pytest.fixture
def intraday_frame() -> pd.DataFrame:
    """Five sessions of hourly bars, 08:00 to 16:00 ET, so pre/post market is present."""
    stamps: list[pd.Timestamp] = []
    for day in pd.bdate_range("2023-03-06", periods=5, tz="America/New_York"):
        stamps += [day + pd.Timedelta(hours=h) for h in range(8, 17)]
    index = pd.DatetimeIndex(stamps).tz_convert("UTC")
    index.name = "timestamp"
    close = pd.Series(np.linspace(100, 130, len(index)), index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
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


def all_bars_signal(frame: pd.DataFrame) -> Signals:
    """Entry on every bar, exit never: the most aggressive thing a strategy can do."""
    return Signals(
        entries=pd.Series(True, index=frame.index),
        exits=pd.Series(False, index=frame.index),
    )


def test_session_columns_exclude_the_straddling_hourly_bar(intraday_frame):
    local = intraday_frame.index.tz_convert("America/New_York")
    tradable_hours = sorted({t.hour for t in local[intraday_frame["in_session"]]})
    assert tradable_hours == [10, 11, 12, 13, 14, 15]

    closing = local[intraday_frame["is_last_bar_of_day"]]
    assert len(closing) == 5
    assert {t.hour for t in closing} == {15}


def test_no_overnight_blocks_entries_on_the_closing_bar(intraday_frame):
    rules = RuleConfig(no_overnight=True)
    signals = RuleEngine(rules).apply(intraday_frame, all_bars_signal(intraday_frame))

    assert not (signals.entries & intraday_frame["is_last_bar_of_day"]).any()
    assert (signals.exits & intraday_frame["is_last_bar_of_day"]).sum() == 5
    # Out-of-session bars are never entry candidates.
    assert not (signals.entries & ~intraday_frame["in_session"]).any()


def test_daily_cap_keeps_only_the_first_signals(intraday_frame):
    rules = RuleConfig(max_trades_per_day=2)
    signals = RuleEngine(rules).apply(intraday_frame, all_bars_signal(intraday_frame))

    per_day = signals.entries.groupby(intraday_frame["session_date"].to_numpy()).sum()
    assert set(per_day.unique()) == {2}


@pytest.mark.parametrize("engine_name", ["vectorbt", "backtesting"])
def test_engines_produce_no_violations(intraday_frame, engine_name):
    rules = RuleConfig(max_trades_per_day=1, no_overnight=True)
    signals = RuleEngine(rules).apply(intraday_frame, all_bars_signal(intraday_frame))

    result = get_engine(engine_name).run(
        frame=intraday_frame,
        signals=signals,
        capital=CapitalConfig(),
        rules=rules,
        bars_per_year=SPEC.bars_per_year,
        symbol="TEST",
    )

    assert validate_trades(result.trades, rules) == 0
    assert len(result.trades) > 0
    assert set(result.trades["exit_reason"]) <= {"signal", "eod_close", "end_of_data"}
    assert result.trades["exit_reason"].eq("eod_close").any()
    assert set(result.metrics) >= {
        "total_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "trade_count",
    }


def test_validator_catches_an_overnight_hold():
    rules = RuleConfig(no_overnight=True)
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2023-03-06 15:00", "2023-03-07 15:00"], utc=True),
            "exit_time": pd.to_datetime(["2023-03-06 20:00", "2023-03-08 20:00"], utc=True),
            "pnl": [1.0, -1.0],
        }
    )
    # The first trade opens and closes inside one New York session; the second does not.
    assert validate_trades(trades, rules) == 1


def test_validator_counts_trades_over_the_daily_cap():
    rules = RuleConfig(max_trades_per_day=1)
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2023-03-06 15:00", "2023-03-06 16:00", "2023-03-06 17:00"], utc=True
            ),
            "exit_time": pd.to_datetime(
                ["2023-03-06 15:30", "2023-03-06 16:30", "2023-03-06 17:30"], utc=True
            ),
            "pnl": [1.0, 1.0, 1.0],
        }
    )
    assert validate_trades(trades, rules) == 2


def test_validator_is_quiet_without_rules():
    assert validate_trades(pd.DataFrame(), RuleConfig()) == 0
