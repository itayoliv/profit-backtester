"""backtesting.py adapter: a generic strategy that replays the precomputed signals.

``trade_on_close=True`` is deliberate. backtesting.py fills on the next bar's open by
default, which would push a forced end-of-day exit into the following session and
break the ``no_overnight`` rule. Filling at the signal bar's close also matches how
vectorbt prices orders, so the two engines stay comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...strategies.base import Signals
from ..request import CapitalConfig, RuleConfig
from .base import EngineResult, compute_metrics, exit_reasons, finalise_trades


def _build_strategy(capital: CapitalConfig):
    from backtesting import Strategy

    class SignalReplay(Strategy):
        entries: np.ndarray = np.array([])
        exits: np.ndarray = np.array([])
        short_entries: np.ndarray = np.array([])
        short_exits: np.ndarray = np.array([])

        def init(self) -> None:
            self._entries = self.I(lambda: self.entries, name="entries", plot=False)
            self._exits = self.I(lambda: self.exits, name="exits", plot=False)
            self._short_entries = self.I(
                lambda: self.short_entries, name="short_entries", plot=False
            )
            self._short_exits = self.I(lambda: self.short_exits, name="short_exits", plot=False)

        def _size(self):
            if capital.sizing == "percent_equity":
                # backtesting.py reads a float in (0, 1) as a fraction of equity.
                return min(capital.size_value, 0.9999)
            if capital.sizing == "fixed_cash":
                return max(1, int(capital.size_value // self.data.Close[-1]))
            return max(1, int(capital.size_value))

        def next(self) -> None:
            long_open = self.position.is_long
            short_open = self.position.is_short

            if long_open and self._exits[-1]:
                self.position.close()
                long_open = False
            if short_open and self._short_exits[-1]:
                self.position.close()
                short_open = False

            if not self.position:
                if self._entries[-1]:
                    self.buy(size=self._size())
                elif capital.allow_short and self._short_entries[-1]:
                    self.sell(size=self._size())

    return SignalReplay


class BacktestingEngine:
    name = "backtesting"

    def run(
        self,
        frame: pd.DataFrame,
        signals: Signals,
        capital: CapitalConfig,
        rules: RuleConfig,
        bars_per_year: float,
        symbol: str,
    ) -> EngineResult:
        from backtesting import Backtest

        data = pd.DataFrame(
            {
                "Open": frame["open"],
                "High": frame["high"],
                "Low": frame["low"],
                "Close": frame["close"],
                "Volume": frame["volume"].fillna(0.0),
            },
            index=frame.index,
        ).ffill().dropna()

        empty = np.zeros(len(data), dtype=bool)
        strategy = _build_strategy(capital)
        # backtesting.py's ``spread`` is charged on entry only, so slippage is charged as
        # a per-side fee instead, which matches vectorbt's slipped entry and exit prices.
        slippage = capital.slippage_bps / 10_000.0
        backtest = Backtest(
            data,
            strategy,
            cash=capital.initial_cash,
            commission=capital.commission + slippage,
            trade_on_close=True,
            finalize_trades=True,
        )
        stats = backtest.run(
            entries=signals.entries.reindex(data.index).fillna(False).to_numpy(),
            exits=signals.exits.reindex(data.index).fillna(False).to_numpy(),
            short_entries=(
                signals.short_entries.reindex(data.index).fillna(False).to_numpy()
                if signals.short_entries is not None
                else empty
            ),
            short_exits=(
                signals.short_exits.reindex(data.index).fillna(False).to_numpy()
                if signals.short_exits is not None
                else empty
            ),
        )

        trades = self._trades(stats, frame, signals, rules, symbol, slippage)
        equity = stats["_equity_curve"]["Equity"].astype("float64")
        equity.name = "equity"
        return EngineResult(
            metrics=compute_metrics(equity, trades, bars_per_year),
            trades=trades,
            equity=equity,
        )

    @staticmethod
    def _trades(
        stats,
        frame: pd.DataFrame,
        signals: Signals,
        rules: RuleConfig,
        symbol: str,
        slippage: float,
    ) -> pd.DataFrame:
        records = stats["_trades"]
        if records is None or len(records) == 0:
            return finalise_trades(None, symbol)

        final_bar = frame.index[-1]
        exit_time = pd.to_datetime(records["ExitTime"], utc=True)
        # finalize_trades closes anything still open on the final bar, but a real exit
        # signal can also land there; only the former counts as end_of_data.
        closed_by_signal = bool(
            signals.exits.get(final_bar, False) or (signals.short_exits is not None
                                                    and signals.short_exits.get(final_bar, False))
        )
        still_open = exit_time.eq(final_bar) & (not closed_by_signal)
        # Report the slipped fill prices; PnL already carries slippage as a fee.
        direction = np.sign(records["Size"].to_numpy())
        out = pd.DataFrame(
            {
                "entry_time": pd.to_datetime(records["EntryTime"], utc=True),
                "exit_time": exit_time,
                "side": np.where(records["Size"].to_numpy() >= 0, "long", "short"),
                "size": records["Size"].abs(),
                "entry_price": records["EntryPrice"] * (1 + direction * slippage),
                "exit_price": records["ExitPrice"] * (1 - direction * slippage),
                "pnl": records["PnL"],
                "return_pct": records["ReturnPct"],
            }
        )
        out["exit_reason"] = exit_reasons(out["exit_time"], still_open, frame, rules)
        return finalise_trades(out, symbol)
