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
    assert request.job_id.startswith("AAPL_2023-01-03_2024-01-01_sma_cross_")
    assert len(request.job_id.removeprefix("AAPL_2023-01-03_2024-01-01_sma_cross_")) == 8


def test_explicit_job_id_is_kept():
    request = parse_request(base_request(job_id="custom-name"))
    assert request.job_id == "custom-name"


def test_batch_job_id_lists_every_ticker():
    request = parse_request(base_request(tickers=["AAPL", "MSFT"]))
    prefix = "AAPL-MSFT_2023-01-03_2024-01-01_sma_cross_"
    assert request.job_id.startswith(prefix)
    assert request.capital.initial_cash == 500
    assert request.capital.allow_fractional is False
    assert request.capital.sizing == "percent_equity"
    assert request.capital.size_value == 1.0
    assert request.capital.commission == 0.0
    assert request.capital.slippage_bps == 1
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


def test_fractional_shares_are_rejected_on_the_backtesting_engine():
    with pytest.raises(ValidationError) as caught:
        parse_request(base_request(engine="backtesting", capital={"allow_fractional": True}))
    assert "allow_fractional" in " ".join(caught.value.details)


def test_non_object_payload_is_rejected():
    with pytest.raises(ValidationError):
        parse_request(["not", "an", "object"])


def test_params_grid_fans_out_one_child_per_valid_combination():
    parent = parse_request(
        base_request(
            job_id="sweep",
            params={},
            params_grid={"fast": [5, 10, 30], "slow": [20, 50]},
        )
    )
    children = parent.children()

    assert parent.is_batch
    assert [c.job_id for c in children] == [
        "sweep-fast5_slow20",
        "sweep-fast5_slow50",
        "sweep-fast10_slow20",
        "sweep-fast10_slow50",
        "sweep-fast30_slow50",
    ]
    assert children[0].params == {"fast": 5, "slow": 20}
    assert all(c.params_grid is None and not c.is_batch for c in children)
    assert parent.skipped_combinations() == [
        {"params": {"fast": 30, "slow": 20}, "reason": "Value error, fast must be strictly less than slow"}
    ]


def test_params_grid_with_several_tickers_names_ticker_then_combination():
    parent = parse_request(
        base_request(
            job_id="sweep",
            tickers=["AAPL", "MSFT"],
            params={"slow": 50},
            params_grid={"fast": [5, 10]},
        )
    )
    assert [c.job_id for c in parent.children()] == [
        "sweep-AAPL-fast5",
        "sweep-MSFT-fast5",
        "sweep-AAPL-fast10",
        "sweep-MSFT-fast10",
    ]
    assert parent.children()[0].params == {"slow": 50, "fast": 5}


@pytest.mark.parametrize(
    "grid, params, expected",
    [
        ({"nonsense": [1]}, {}, "unknown parameters"),
        ({"fast": []}, {}, "at least one value"),
        ({}, {}, "at least one parameter"),
        ({"fast": [5]}, {"fast": 5}, "both params and params_grid"),
        ({"fast": [50, 60], "slow": [10, 20]}, {}, "every params_grid combination is invalid"),
        ({"fast": list(range(1, 30)), "slow": list(range(31, 40))}, {}, "run limit"),
    ],
)
def test_bad_params_grid_is_rejected(grid, params, expected):
    with pytest.raises(ValidationError) as caught:
        parse_request(base_request(params=params, params_grid=grid))
    assert expected in " ".join(caught.value.details)
