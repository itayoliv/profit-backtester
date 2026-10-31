"""Request defaults, normalisation and rejection."""

from __future__ import annotations

import pytest

from backtester.errors import ValidationError
from backtester.pipeline.request import parse_request
from helpers import base_request


def test_defaults_match_the_documented_contract():
    request = parse_request(base_request())

    assert request.source == "agent"
    assert request.engine == "vectorbt"
    assert request.parent_job_id is None
    assert request.job_id  # auto-generated
    assert request.capital.initial_cash == 100_000
    assert request.capital.sizing == "percent_equity"
    assert request.capital.size_value == 1.0
    assert request.capital.commission == 0.0
    assert request.capital.slippage_bps == 5
    assert request.capital.allow_short is False
    assert request.rules.max_trades_per_day is None
    assert request.rules.no_overnight is False
    assert request.filters.weekdays is None
    assert request.filters.regime is None


def test_tickers_are_upper_cased_and_deduplicated():
    request = parse_request(base_request(tickers=["aapl", " msft ", "AAPL"]))
    assert request.tickers == ["AAPL", "MSFT"]


def test_batch_fans_out_into_named_children():
    parent = parse_request(base_request(tickers=["AAPL", "MSFT"], job_id="batch"))
    children = parent.children()

    assert parent.is_batch
    assert [c.job_id for c in children] == ["batch-AAPL", "batch-MSFT"]
    assert all(c.parent_job_id == "batch" for c in children)
    assert all(len(c.tickers) == 1 for c in children)
    assert not children[0].is_batch


@pytest.mark.parametrize(
    "overrides, expected_field",
    [
        ({"tickers": []}, "tickers"),
        ({"timeframe": "3Min"}, "timeframe"),
        ({"start": "2024-01-01", "end": "2023-01-01"}, "start"),
        ({"strategy": "does_not_exist"}, "strategy"),
        ({"params": {"fast": 50, "slow": 10}}, "fast"),
        ({"params": {"nonsense": 1}}, "nonsense"),
        ({"engine": "zipline"}, "engine"),
        ({"capital": {"sizing": "martingale"}}, "sizing"),
        ({"capital": {"initial_cash": -5}}, "initial_cash"),
        ({"capital": {"sizing": "percent_equity", "size_value": 2.0}}, "size_value"),
        ({"rules": {"max_trades_per_day": 0}}, "max_trades_per_day"),
        ({"filters": {"weekdays": [9]}}, "weekdays"),
        ({"unexpected_key": 1}, "unexpected_key"),
    ],
)
def test_invalid_requests_are_rejected(overrides, expected_field):
    with pytest.raises(ValidationError) as caught:
        parse_request(base_request(**overrides))
    assert expected_field in " ".join(caught.value.details)


def test_non_object_payload_is_rejected():
    with pytest.raises(ValidationError):
        parse_request(["not", "an", "object"])
