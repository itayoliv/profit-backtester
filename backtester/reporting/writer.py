"""Writes summary.json, the trade log, the equity curve and report.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..config import Settings
from ..jsonio import jsonable, read_json, write_json
from ..pipeline.engines.base import empty_trades

MAX_REPORT_LINES = 150


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _table(rows: Iterable[tuple[str, Any]]) -> list[str]:
    lines = ["| field | value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return lines


class ResultWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def write(
        self,
        job_id: str,
        summary: dict[str, Any],
        trades: pd.DataFrame | None = None,
        equity: pd.Series | None = None,
        notes: list[str] | None = None,
    ) -> Path:
        directory = self.settings.result_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)

        write_json(directory / "summary.json", summary)

        # Always write the contract columns, even for a parent job with no trades of
        # its own, so readers never hit a headerless CSV.
        frame = empty_trades() if trades is None or trades.empty else trades
        frame.to_csv(directory / "trades.csv", index=False)
        write_json(
            directory / "trades.json",
            [
                {key: jsonable(value) for key, value in record.items()}
                for record in frame.to_dict(orient="records")
            ],
        )

        curve = equity if equity is not None else pd.Series(dtype="float64", name="equity")
        curve.to_frame(name="equity").to_parquet(
            directory / "equity.parquet", engine="pyarrow", index=True
        )

        (directory / "report.md").write_text(
            self.render_report(summary, notes or []), encoding="utf-8"
        )
        return directory

    def summary(self, job_id: str) -> dict[str, Any] | None:
        return read_json(self.settings.result_dir(job_id) / "summary.json")

    def report_path(self, job_id: str) -> Path:
        return self.settings.result_dir(job_id) / "report.md"

    # --- markdown ---------------------------------------------------------
    def render_report(self, summary: dict[str, Any], notes: list[str]) -> str:
        coverage = summary.get("data_coverage") or {}
        rules = summary.get("rules") or {}
        capital = summary.get("capital") or {}
        children = summary.get("children")

        lines = [f"# Backtest {summary.get('job_id', 'unknown')}", ""]

        lines += ["## Setup", ""]
        lines += _table(
            [
                ("tickers", ", ".join(summary.get("tickers") or []) or "n/a"),
                ("strategy", summary.get("strategy")),
                ("params", summary.get("params")),
                ("engine", summary.get("engine")),
                ("timeframe", coverage.get("timeframe", "n/a")),
                ("window", f"{coverage.get('requested_start', '?')} to {coverage.get('requested_end', '?')}"),
                ("initial cash", _num(capital.get("initial_cash"), 2)),
                ("sizing", f"{capital.get('sizing')} @ {capital.get('size_value')}"),
                ("commission", capital.get("commission")),
                ("slippage", f"{capital.get('slippage_bps')} bps"),
                ("shorting", "enabled" if capital.get("allow_short") else "disabled"),
            ]
        )
        lines.append("")

        lines += ["## Metrics", ""]
        lines += _table(
            [
                ("total_return", _pct(summary.get("total_return"))),
                ("sharpe (annualised)", _num(summary.get("sharpe"))),
                ("max_drawdown", _pct(summary.get("max_drawdown"))),
                ("win_rate", _pct(summary.get("win_rate"))),
                ("trade_count", summary.get("trade_count")),
            ]
        )
        lines.append("")

        if children:
            lines += ["Per-ticker results:", "", "| ticker | total_return | sharpe | trades |", "| --- | --- | --- | --- |"]
            for child in children[: MAX_REPORT_LINES // 4]:
                lines.append(
                    f"| {','.join(child.get('tickers') or [])} | {_pct(child.get('total_return'))} "
                    f"| {_num(child.get('sharpe'))} | {child.get('trade_count')} |"
                )
            lines.append("")

        lines += ["## Rules", ""]
        lines += _table(
            [
                ("max_trades_per_day", rules.get("max_trades_per_day") or "unlimited"),
                ("no_overnight", bool(rules.get("no_overnight"))),
                ("rule_violations", summary.get("rule_violations", 0)),
                ("filters", summary.get("filters")),
            ]
        )
        lines.append("")

        lines += ["## Data coverage", ""]
        lines += _table(
            [
                ("feed", coverage.get("feed", "n/a")),
                ("adjustment", coverage.get("adjustment", "n/a")),
                ("bars", coverage.get("bar_count", "n/a")),
                ("first bar", coverage.get("first_bar", "n/a")),
                ("last bar", coverage.get("last_bar", "n/a")),
                ("tradable bars", coverage.get("tradable", "n/a")),
                ("cache hit", coverage.get("cache_hit", "n/a")),
                ("downloaded ranges", len(coverage.get("downloaded_ranges") or [])),
            ]
        )
        lines.append("")

        lines += ["## Notes", ""]
        body = notes or ["Nothing unusual."]
        lines += [f"- {note}" for note in body]

        if len(lines) > MAX_REPORT_LINES:
            lines = lines[: MAX_REPORT_LINES - 1] + ["- (report truncated)"]
        return "\n".join(lines) + "\n"
