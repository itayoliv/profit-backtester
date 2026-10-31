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
