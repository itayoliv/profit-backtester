"""The staged backtest: fetch, filter, run, validate. One symbol per run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config import Settings
from ..data.manager import DataManager
from ..data.progress import ProgressReporter
from ..data.timeframes import get_timeframe
from ..strategies.registry import get_strategy
from .engines import get_engine
from .filters import build_tradable_mask
from .request import BacktestRequest
from .rules import RuleEngine, validate_trades
from .session import add_session_columns


@dataclass
class PipelineResult:
    summary: dict[str, Any]
    trades: pd.DataFrame
    equity: pd.Series
    notes: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(self, settings: Settings, data_manager: DataManager | None = None) -> None:
        self.settings = settings
        self.data = data_manager or DataManager(settings)

    def run(
        self,
        request: BacktestRequest,
        symbol: str | None = None,
        reporter: ProgressReporter | None = None,
    ) -> PipelineResult:
        symbol = (symbol or request.tickers[0]).upper()
        reporter = reporter or ProgressReporter()
        reporter.bind(symbol)
        spec = get_timeframe(request.timeframe)
        strategy = get_strategy(request.strategy)
        params = strategy.parse_params(request.params)
        notes: list[str] = []

        reporter.set_phase("fetching_data")
        bars = self.data.get_bars(
            symbol, spec, request.start_ts, request.end_ts, reporter=reporter
        )

        reporter.set_phase("filtering")
        frame = add_session_columns(bars.frame, spec)
        tradable, counts = build_tradable_mask(
            frame, request.filters, self.data, request.start_ts, request.end_ts
        )
        frame["tradable"] = tradable
        if counts["tradable"] == 0:
            notes.append("No bars passed the filters, so no entry could be taken.")

        reporter.set_phase("running_strategy")
        raw_signals = strategy.generate_signals(frame, params)
        signals = RuleEngine(request.rules).apply(frame, raw_signals)
        engine = get_engine(request.engine)
        result = engine.run(
            frame=frame,
            signals=signals,
            capital=request.capital,
            rules=request.rules,
            bars_per_year=spec.bars_per_year,
            symbol=symbol,
        )

        reporter.set_phase("applying_rules")
        violations = validate_trades(result.trades, request.rules)
        if violations:
            notes.append(
                f"{violations} rule violation(s) survived enforcement; treat metrics "
                "with suspicion and report this as an engine bug."
            )
        if spec.name == "1Hour":
            notes.append(
                "Hourly bars are labelled by open time, so the 09:00 ET bar is excluded; "
                "the tradable session runs 10:00 to 15:00 and EOD exits fill at the "
                "15:00 bar's close."
            )
        if request.rules.no_overnight and not spec.intraday:
            notes.append(
                "no_overnight on a daily timeframe closes every position on its entry bar."
            )

        summary = {
            "job_id": request.job_id,
            "parent_job_id": request.parent_job_id,
            "tickers": [symbol],
            "strategy": request.strategy,
            "engine": request.engine,
            "params": params.model_dump(),
            "rules": request.rules.model_dump(),
            "filters": request.filters.model_dump(),
            "capital": request.capital.model_dump(),
            "data_coverage": {**bars.coverage, **counts},
            "rule_violations": violations,
            **result.metrics,
        }
        return PipelineResult(
            summary=summary, trades=result.trades, equity=result.equity, notes=notes
        )
