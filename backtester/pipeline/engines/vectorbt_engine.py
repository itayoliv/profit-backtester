"""vectorbt adapter: rules are already baked into the signal arrays."""

from __future__ import annotations

import pandas as pd

from ...strategies.base import Signals
from ..execution import distance_on_entry_bar, resolve_fill, stop_fraction, stops_hit
from ..request import CapitalConfig, Fill, RuleConfig, StopConfig
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
        stop: StopConfig | None = None,
        fill: Fill = "close",
    ) -> EngineResult:
        import vectorbt as vbt
        from vectorbt.portfolio.enums import Direction, SizeType, StopEntryPrice, StopExitPrice

        # Close fills keep the original signals so a default request is unchanged.
        # Next-open moves the order onto the following bar and, with no_overnight,
        # leaves a last-bar exit on that bar so it cannot spill into the next session.
        planned, price = resolve_fill(signals, frame, fill, rules)
        distance = distance_on_entry_bar(frame, stop, fill)

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
            kwargs["short_entries"] = planned.short_entries
            kwargs["short_exits"] = planned.short_exits
        if price is not None:
            kwargs["price"] = price
        fraction = stop_fraction(frame, distance, fill)
        if fraction is not None:
            # Stops read high/low. The fraction is of the fill (FillPrice), so a
            # fixed dollar stop is that many dollars from the entry when slippage
            # is zero. StopMarket then applies the same slippage as a signal exit.
            kwargs["open"] = frame["open"]
            kwargs["high"] = frame["high"]
            kwargs["low"] = frame["low"]
            kwargs["sl_stop"] = fraction
            kwargs["stop_entry_price"] = StopEntryPrice.FillPrice
            kwargs["stop_exit_price"] = StopExitPrice.StopMarket

        portfolio = vbt.Portfolio.from_signals(
            close=frame["close"],
            entries=planned.entries,
            exits=planned.exits,
            init_cash=capital.initial_cash,
            size=capital.size_value,
            size_type=size_type,
            fees=capital.commission,
            slippage=capital.slippage_bps / 10_000.0,
            direction=Direction.Both if capital.allow_short else Direction.LongOnly,
            freq=pd.Timedelta(frame.index.freq or _infer_step(frame.index)),
            **kwargs,
        )

        trades = self._trades(portfolio, frame, rules, symbol, distance)
        equity = portfolio.value()
        equity.name = "equity"
        return EngineResult(
            metrics=compute_metrics(equity, trades, bars_per_year),
            trades=trades,
            equity=equity,
        )

    @staticmethod
    def _trades(portfolio, frame, rules, symbol, distance) -> pd.DataFrame:
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
        out["exit_reason"] = exit_reasons(
            out["exit_time"],
            still_open,
            frame,
            rules,
            stopped=stops_hit(
                out["entry_time"], out["side"], out["entry_price"], out["exit_price"], distance
            ),
        )
        return finalise_trades(out, symbol)


def _infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(days=1)
    return pd.Timedelta(pd.Series(index).diff().dropna().median())
