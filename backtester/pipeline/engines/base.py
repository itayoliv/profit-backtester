"""Shared engine contract: identical metric definitions and trade-log shape."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from ...strategies.base import Signals
from ..request import CapitalConfig, RuleConfig

ENGINE_NAMES = ("vectorbt", "backtesting")

TRADE_COLUMNS = (
    "symbol",
    "entry_time",
    "exit_time",
    "side",
    "size",
    "entry_price",
    "exit_price",
    "pnl",
    "return_pct",
    "exit_reason",
)

METRIC_KEYS = ("total_return", "sharpe", "max_drawdown", "win_rate", "trade_count")


@dataclass
class EngineResult:
    metrics: dict[str, float] = field(default_factory=dict)
    trades: pd.DataFrame = field(default_factory=lambda: empty_trades())
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))


class Engine(Protocol):
    name: str

    def run(
        self,
        frame: pd.DataFrame,
        signals: Signals,
        capital: CapitalConfig,
        rules: RuleConfig,
        bars_per_year: float,
        symbol: str,
    ) -> EngineResult: ...


def empty_trades() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in TRADE_COLUMNS})


def compute_metrics(
    equity: pd.Series, trades: pd.DataFrame, bars_per_year: float
) -> dict[str, float]:
    """Metrics are derived from the equity curve here, not from each library's own
    statistics, so vectorbt and backtesting.py results are directly comparable."""
    equity = pd.Series(equity).astype("float64").dropna()
    if len(equity) < 2 or equity.iloc[0] == 0:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trade_count": int(len(trades)),
        }

    returns = equity.pct_change().dropna()
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (
        float(returns.mean()) / std * float(np.sqrt(bars_per_year)) if std > 0 else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0
    wins = (
        float((pd.to_numeric(trades["pnl"], errors="coerce") > 0).mean())
        if len(trades)
        else 0.0
    )
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "sharpe": round(sharpe, 6),
        "max_drawdown": float(drawdown.min()),
        "win_rate": round(wins, 6),
        "trade_count": int(len(trades)),
    }


def exit_reasons(
    exit_times: pd.Series,
    still_open: pd.Series,
    frame: pd.DataFrame,
    rules: RuleConfig,
) -> list[str]:
    last_bar = frame["is_last_bar_of_day"].astype(bool)
    reasons: list[str] = []
    for exit_time, was_open in zip(exit_times, still_open, strict=True):
        if bool(was_open) or pd.isna(exit_time):
            reasons.append("end_of_data")
        elif rules.no_overnight and bool(last_bar.get(pd.Timestamp(exit_time), False)):
            reasons.append("eod_close")
        else:
            reasons.append("signal")
    return reasons


def finalise_trades(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Coerce an engine's trade frame into the contract column order and dtypes."""
    if trades is None or trades.empty:
        return empty_trades()
    out = trades.copy()
    out["symbol"] = symbol
    for column in TRADE_COLUMNS:
        if column not in out.columns:
            out[column] = None
    for column in ("entry_time", "exit_time"):
        out[column] = pd.to_datetime(out[column], utc=True)
    for column in ("size", "entry_price", "exit_price", "pnl", "return_pct"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.loc[:, list(TRADE_COLUMNS)].sort_values("entry_time").reset_index(drop=True)
