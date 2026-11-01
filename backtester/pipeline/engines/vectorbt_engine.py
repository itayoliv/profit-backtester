"""vectorbt adapter: rules are already baked into the signal arrays."""

from __future__ import annotations

import pandas as pd

from ...strategies.base import Signals
from ..request import CapitalConfig, RuleConfig
from .base import EngineResult, compute_metrics, exit_reasons, finalise_trades


class VectorbtEngine:
    name = "vectorbt"

    def run(
        self,
        frame: pd.DataFrame,
        signals: Signals,
        capital: CapitalConfig,
        rules: RuleConfig,
        bars_per_year: float,
        symbol: str,
    ) -> EngineResult:
        import vectorbt as vbt
        from vectorbt.portfolio.enums import Direction, SizeType

        size_type = {
            "percent_equity": SizeType.Percent,
            "fixed_cash": SizeType.Value,
            "fixed_shares": SizeType.Amount,
        }[capital.sizing]

        kwargs = {}
        if not capital.allow_fractional:
            # Round the share count down to whole shares, on every sizing mode.
            kwargs["size_granularity"] = 1.0
        if capital.allow_short:
            kwargs["short_entries"] = signals.short_entries
            kwargs["short_exits"] = signals.short_exits

        portfolio = vbt.Portfolio.from_signals(
            close=frame["close"],
            entries=signals.entries,
            exits=signals.exits,
            init_cash=capital.initial_cash,
            size=capital.size_value,
            size_type=size_type,
            fees=capital.commission,
            slippage=capital.slippage_bps / 10_000.0,
            direction=Direction.Both if capital.allow_short else Direction.LongOnly,
            freq=pd.Timedelta(frame.index.freq or _infer_step(frame.index)),
            **kwargs,
        )

        trades = self._trades(portfolio, frame, rules, symbol)
        equity = portfolio.value()
        equity.name = "equity"
        return EngineResult(
            metrics=compute_metrics(equity, trades, bars_per_year),
            trades=trades,
            equity=equity,
        )

    @staticmethod
    def _trades(portfolio, frame: pd.DataFrame, rules: RuleConfig, symbol: str) -> pd.DataFrame:
        records = portfolio.trades.records_readable
        if records is None or len(records) == 0:
            return finalise_trades(None, symbol)

        still_open = records["Status"].astype(str).str.lower().eq("open")
        out = pd.DataFrame(
            {
                "entry_time": records["Entry Timestamp"],
                "exit_time": records["Exit Timestamp"],
                "side": records["Direction"].astype(str).str.lower(),
                "size": records["Size"],
                "entry_price": records["Avg Entry Price"],
                "exit_price": records["Avg Exit Price"],
                "pnl": records["PnL"],
                "return_pct": records["Return"],
            }
        )
        out["exit_reason"] = exit_reasons(out["exit_time"], still_open, frame, rules)
        return finalise_trades(out, symbol)


def _infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(days=1)
    return pd.Timedelta(pd.Series(index).diff().dropna().median())
