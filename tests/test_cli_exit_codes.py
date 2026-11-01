"""The exit codes and stdout shape the agent depends on."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from backtester.cli import app
from backtester.jobs.queue import JobQueue
from backtester.jobs.worker import Worker
from helpers import base_request, seed_cache

runner = CliRunner()


def write_request(tmp_path, **overrides):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(base_request(**overrides)), encoding="utf-8")
    return path


def invoke(settings, *args):
    result = runner.invoke(app, [*args, "--workspace", str(settings.workspace)])
    return result


def payload_of(result):
    return json.loads(result.stdout)


def test_submit_returns_zero_and_a_job_id(settings, tmp_path):
    result = invoke(settings, "submit", str(write_request(tmp_path, job_id="ok-1")))

    assert result.exit_code == 0
    assert payload_of(result)["job_id"] == "ok-1"
    assert (settings.inbox_dir / "ok-1.json").exists()


def test_submit_rejects_an_invalid_request(settings, tmp_path):
    result = invoke(settings, "submit", str(write_request(tmp_path, timeframe="3Min")))

    assert result.exit_code == 2
    body = payload_of(result)
    assert body["error"] == "validation_error"
    assert any("timeframe" in detail for detail in body["details"])


def test_submit_rejects_malformed_json(settings, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    result = invoke(settings, "submit", str(path))

    assert result.exit_code == 2
    assert payload_of(result)["error"] == "validation_error"


def test_submit_rejects_a_missing_file(settings):
    result = invoke(settings, "submit", "nope.json")
    assert result.exit_code == 2


def test_submit_rejects_a_duplicate_job_id(settings, tmp_path):
    path = write_request(tmp_path, job_id="dupe")
    assert invoke(settings, "submit", str(path)).exit_code == 0

    result = invoke(settings, "submit", str(path))

    assert result.exit_code == 2
    assert "dupe" in " ".join(payload_of(result)["details"])


def test_results_returns_three_while_queued(settings, tmp_path):
    invoke(settings, "submit", str(write_request(tmp_path, job_id="pending")))

    result = invoke(settings, "results", "pending")

    assert result.exit_code == 3
    assert payload_of(result)["phase"] == "queued"


def test_results_returns_four_for_an_unknown_job(settings):
    result = invoke(settings, "results", "never-existed")

    assert result.exit_code == 4
    assert payload_of(result)["error"] == "unknown_job"


def test_results_returns_four_for_a_failed_job(settings, tmp_path):
    invoke(settings, "submit", str(write_request(tmp_path, job_id="doomed")))
    JobQueue(settings).state.update(
        "doomed", phase="failed", error={"code": "no_data", "message": "none", "traceback": None}
    )

    result = invoke(settings, "results", "doomed")

    assert result.exit_code == 4
    assert payload_of(result)["error"]["code"] == "no_data"


def test_status_always_returns_zero(settings):
    assert invoke(settings, "status").exit_code == 0
    unknown = invoke(settings, "status", "--job-id", "nope")
    assert unknown.exit_code == 0
    assert payload_of(unknown)["error"] == "unknown_job"


def test_strategies_lists_the_builtin(settings):
    result = invoke(settings, "strategies")

    assert result.exit_code == 0
    names = [s["name"] for s in payload_of(result)["strategies"]]
    assert "sma_cross" in names


def test_every_strategy_and_parameter_is_described(settings):
    """The agent only learns what a parameter does from these descriptions."""
    for strategy in payload_of(invoke(settings, "strategies"))["strategies"]:
        assert strategy["description"], strategy["name"]
        for name, field in strategy["params_schema"]["properties"].items():
            assert field.get("description"), f"{strategy['name']}.{name} has no description"


def test_submit_reports_sweep_size_and_skips(settings, tmp_path):
    path = write_request(
        tmp_path, job_id="grid", params={}, params_grid={"fast": [5, 30], "slow": [20, 50]}
    )
    body = payload_of(invoke(settings, "submit", str(path)))

    assert body["children"] == 3
    assert body["param_sets"] == 3
    assert body["skipped_combinations"][0]["params"] == {"fast": 30, "slow": 20}


def test_compare_exit_codes(settings, tmp_path):
    assert invoke(settings, "compare", "never-existed").exit_code == 4
    invoke(settings, "submit", str(write_request(tmp_path, job_id="waiting")))
    assert invoke(settings, "compare", "waiting").exit_code == 3
    assert invoke(settings, "compare", "waiting", "--sort-by", "luck").exit_code == 2


@pytest.mark.slow
def test_sweep_then_compare(settings, tmp_path):
    seed_cache(settings, "AAPL")
    path = write_request(
        tmp_path, job_id="sweep", params={}, params_grid={"fast": [3, 5], "slow": [20, 40]}
    )
    assert invoke(settings, "submit", str(path)).exit_code == 0

    Worker(settings).run(once=True)

    body = payload_of(invoke(settings, "results", "sweep"))
    assert "summary" in body, payload_of(invoke(settings, "status", "--job-id", "sweep"))
    summary = body["summary"]
    assert summary["params_grid"] == {"fast": [3, 5], "slow": [20, 40]}
    assert len(summary["comparison"]["rows"]) == 4
    report = (settings.result_dir("sweep") / "report.md").read_text(encoding="utf-8")
    assert "Per-run results" in report

    result = invoke(settings, "compare", "sweep", "--sort-by", "total_return")
    assert result.exit_code == 0
    body = payload_of(result)
    rows = body["rows"]
    assert [r["rank"] for r in rows] == [1, 2, 3, 4]
    returns = [r["total_return"] for r in rows]
    assert returns == sorted(returns, reverse=True)
    assert body["varied"] == ["params.fast", "params.slow"]
    assert {r["job_id"] for r in rows} == {
        "sweep-fast3_slow20",
        "sweep-fast3_slow40",
        "sweep-fast5_slow20",
        "sweep-fast5_slow40",
    }


@pytest.mark.slow
def test_full_loop_submit_run_results(settings, tmp_path):
    """End to end on cached bars: submit, drain the queue, read the results."""
    seed_cache(settings, "AAPL")
    seed_cache(settings, "MSFT")

    submitted = invoke(
        settings, "submit", str(write_request(tmp_path, job_id="loop", tickers=["AAPL", "MSFT"]))
    )
    assert submitted.exit_code == 0

    Worker(settings).run(once=True)

    status = invoke(settings, "status", "--job-id", "loop")
    assert status.exit_code == 0
    record = payload_of(status)
    assert record["phase"] == "done"
    assert len(record["children"]) == 2

    result = invoke(settings, "results", "loop")
    assert result.exit_code == 0
    body = payload_of(result)
    summary = body["summary"]
    for key in (
        "total_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "trade_count",
        "rule_violations",
        "params",
        "rules",
        "data_coverage",
        "engine",
        "strategy",
    ):
        assert key in summary
    assert summary["rule_violations"] == 0
    assert len(summary["children"]) == 2
    assert summary["failed_children"] == []

    report = (settings.result_dir("loop") / "report.md").read_text(encoding="utf-8")
    for section in ("## Setup", "## Metrics", "## Rules", "## Data coverage", "## Notes"):
        assert section in report
    assert len(report.splitlines()) <= 150


@pytest.mark.slow
def test_one_bad_ticker_does_not_sink_the_batch(settings, tmp_path):
    seed_cache(settings, "AAPL")  # MSFT is deliberately absent and has no credentials

    invoke(
        settings,
        "submit",
        str(write_request(tmp_path, job_id="partial", tickers=["AAPL", "MSFT"])),
    )
    Worker(settings).run(once=True)

    result = invoke(settings, "results", "partial")
    assert result.exit_code == 0
    summary = payload_of(result)["summary"]
    assert summary["tickers"] == ["AAPL", "MSFT"]
    assert len(summary["children"]) == 1
    assert len(summary["failed_children"]) == 1
    assert summary["failed_children"][0]["job_id"] == "partial-MSFT"
