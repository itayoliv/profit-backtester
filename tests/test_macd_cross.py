"""MACD crossover signals: reverse-cross exit, stop, and the date window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.errors import ValidationError
from backtester.pipeline.request import parse_request
from backtester.strategies.macd_cross import MacdCross
from backtester.strategies.registry import get_strategy
from helpers import base_request


def _frame(close: list[float], *, low_mult: float = 0.99, high_mult: float = 1.01) -> pd.DataFrame:
    index = pd.date_range("2023-01-02", periods=len(close), freq="B", tz="UTC")
    prices = np.asarray(close, dtype="float64")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * high_mult,
            "low": prices * low_mult,
            "close": prices,
            "volume": 1e5,
        },
        index=index,
    )


def test_request_accepts_macd_cross_defaults():
    request = parse_request(base_request(strategy="macd_cross", params={}))
    params = get_strategy("macd_cross").parse_params(request.params)
    assert request.strategy == "macd_cross"
    assert params.fast == 12
    assert params.slow == 26
    assert params.signal_len == 9
    assert params.stop_pct == 0.0
    assert params.target_pct == 0.0
    assert params.use_window is False


def test_fast_must_be_shorter_than_slow():
    with pytest.raises(ValidationError):
        parse_request(base_request(strategy="macd_cross", params={"fast": 26, "slow": 12}))


def test_cross_up_enters_and_cross_down_exits_later():
    # A long rise then a long fall is enough for one MACD cross each way.
    close = list(np.linspace(100, 160, 80)) + list(np.linspace(160, 90, 80))
    signals = MacdCross().generate_signals(_frame(close), MacdCross().parse_params({}))

    assert signals.entries.any()
    assert signals.exits.any()
    assert not (signals.entries & signals.exits).any()
    first_entry = int(signals.entries.to_numpy().argmax())
    exits_after = signals.exits.to_numpy().copy()
    exits_after[: first_entry + 1] = False
    assert exits_after.any()


def test_stop_exits_without_waiting_for_the_reverse_cross():
    close = list(np.linspace(100, 160, 80)) + [165] * 15
    frame = _frame(close)
    strategy = MacdCross()
    plain = strategy.generate_signals(frame, strategy.parse_params({}))
    entry_at = int(plain.entries.to_numpy().argmax())
    # Crash the low of the next bar only. The close stays up, so MACD does not cross down.
    frame.loc[frame.index[entry_at + 1], "low"] = frame.loc[frame.index[entry_at], "close"] * 0.5

    assert plain.entries.any()
    stopped = strategy.generate_signals(frame, strategy.parse_params({"stop_pct": 10}))
    assert bool(stopped.exits.iloc[entry_at + 1])
    assert not bool(plain.exits.iloc[entry_at + 1])


def test_date_window_blocks_entries_outside_it():
    close = list(np.linspace(100, 160, 80)) + list(np.linspace(160, 90, 80))
    strategy = MacdCross()
    signals = strategy.generate_signals(
        _frame(close),
        strategy.parse_params(
            {"use_window": True, "window_start": "2099-01-01", "window_end": "2099-12-31"}
        ),
    )
    assert not signals.entries.any()
