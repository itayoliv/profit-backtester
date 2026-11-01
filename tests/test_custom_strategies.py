"""Agent-written strategies: the checks, the author record, and where it shows up."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from backtester.cli import app
from backtester.jobs.worker import Worker
from helpers import base_request, seed_cache

runner = CliRunner()
TEMPLATE = Path(__file__).resolve().parent.parent / "examples" / "strategy_template.py"


def invoke(*args):
    return runner.invoke(app, list(args))


def body(result):
    return json.loads(result.stdout)


def draft(tmp_path, text: str, name: str = "draft.py") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def strategies_by_name():
    return {s["name"]: s for s in body(invoke("strategies"))["strategies"]}


def test_add_records_the_author_and_lists_it(settings):
    result = invoke("add-strategy", str(TEMPLATE), "--author", "grok")

    assert result.exit_code == 0, result.stdout
    assert body(result)["author"] == "grok"
    listed = strategies_by_name()
    assert listed["rsi_dip"]["origin"] == "custom"
    assert listed["rsi_dip"]["author"] == "grok"
    assert listed["sma_cross"]["origin"] == "builtin"
    assert listed["sma_cross"]["author"] is None


def test_author_is_required(settings):
    assert invoke("add-strategy", str(TEMPLATE)).exit_code != 0


def test_same_name_needs_replace_and_takes_the_new_author(settings):
    assert invoke("add-strategy", str(TEMPLATE), "--author", "grok").exit_code == 0

    again = invoke("add-strategy", str(TEMPLATE), "--author", "itay")
    assert again.exit_code == 2
    assert "--replace" in " ".join(body(again)["details"])

    replaced = invoke("add-strategy", str(TEMPLATE), "--author", "itay", "--replace")
    assert replaced.exit_code == 0
    assert body(replaced)["replaced"]["author"] == "grok"
    assert strategies_by_name()["rsi_dip"]["author"] == "itay"


def test_cannot_shadow_a_builtin(settings, tmp_path):
    text = TEMPLATE.read_text(encoding="utf-8").replace('name = "rsi_dip"', 'name = "sma_cross"')
    result = invoke("add-strategy", draft(tmp_path, text), "--author", "grok")

    assert result.exit_code == 2
    assert "built-in" in " ".join(body(result)["details"])


def test_lookahead_is_rejected(settings, tmp_path):
    text = TEMPLATE.read_text(encoding="utf-8").replace(
        'change = df["close"].diff()', 'change = df["close"].shift(-1) - df["close"]'
    )
    result = invoke("add-strategy", draft(tmp_path, text), "--author", "grok")

    assert result.exit_code == 2
    assert "future" in " ".join(body(result)["details"])


@pytest.mark.parametrize("name", ["sma_cross", "macd_cross"])
def test_builtins_pass_the_same_checks(name):
    from backtester.strategies.custom import check_strategy
    from backtester.strategies.registry import get_strategy

    check_strategy(get_strategy(name))


def test_undocumented_parameter_is_rejected(settings, tmp_path):
    text = TEMPLATE.read_text(encoding="utf-8").replace(
        '    """Bars in the RSI average. Smaller reacts faster and gives more, noisier signals."""\n',
        "",
    )
    result = invoke("add-strategy", draft(tmp_path, text), "--author", "grok")

    assert result.exit_code == 2
    assert "length" in " ".join(body(result)["details"])


def test_a_crashing_strategy_is_rejected(settings, tmp_path):
    text = TEMPLATE.read_text(encoding="utf-8").replace(
        'change = df["close"].diff()', 'change = df["no_such_column"].diff()'
    )
    result = invoke("add-strategy", draft(tmp_path, text), "--author", "grok")

    assert result.exit_code == 2
    assert "generate_signals raised" in " ".join(body(result)["details"])


def test_editing_the_installed_file_stops_it_loading(settings, tmp_path):
    installed = Path(body(invoke("add-strategy", str(TEMPLATE), "--author", "grok"))["path"])
    installed.write_text(installed.read_text(encoding="utf-8") + "\n# tweak\n", encoding="utf-8")

    listing = body(invoke("strategies"))
    assert "rsi_dip" not in {s["name"] for s in listing["strategies"]}
    assert listing["load_errors"][0]["name"] == "rsi_dip"
    result = invoke("submit", draft(tmp_path, json.dumps(base_request(strategy="rsi_dip", params={})), "r.json"))
    assert result.exit_code == 2


def test_files_dropped_in_by_hand_are_not_loaded(settings, tmp_path):
    directory = Path(settings.workspace).parent / "custom_strategies"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sneaky.py").write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")

    listing = body(invoke("strategies"))
    assert "rsi_dip" not in {s["name"] for s in listing["strategies"]}
    assert listing["unregistered_files"] == ["sneaky.py"]


def test_remove(settings):
    invoke("add-strategy", str(TEMPLATE), "--author", "grok")

    assert invoke("remove-strategy", "rsi_dip").exit_code == 0
    assert "rsi_dip" not in strategies_by_name()
    assert invoke("remove-strategy", "rsi_dip").exit_code == 2
    assert invoke("remove-strategy", "sma_cross").exit_code == 2


@pytest.mark.slow
def test_author_reaches_results_report_and_compare(settings, tmp_path):
    seed_cache(settings, "AAPL")
    invoke("add-strategy", str(TEMPLATE), "--author", "grok")
    request = base_request(job_id="by-grok", strategy="rsi_dip", params={})
    workspace = ["--workspace", str(settings.workspace)]
    assert invoke("submit", draft(tmp_path, json.dumps(request), "r.json"), *workspace).exit_code == 0

    Worker(settings).run(once=True)

    result = body(invoke("results", "by-grok", *workspace))
    origin = result["summary"]["strategy_origin"]
    assert origin["origin"] == "custom" and origin["author"] == "grok"
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "| written by | grok |" in report
    assert "written by grok" in report
    rows = body(invoke("compare", "by-grok", *workspace))["rows"]
    assert rows[0]["strategy_author"] == "grok"
