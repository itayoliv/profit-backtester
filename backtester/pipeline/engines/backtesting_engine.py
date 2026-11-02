"""backtesting.py adapter: a generic strategy that replays the precomputed signals.

``trade_on_close=True`` is the default and matches vectorbt's close fill. A
next-open request turns it off, so a market order placed on the signal bar fills
at the next bar's open. That would push a last-bar exit into the next session, so
when ``no_overnight`` is set the position is closed at the last bar's close inside
``next`` instead of waiting for that open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...strategies.base import Signals
from ..execution import distance_on_entry_bar, flatten_at_close_mask, order_stop_prices, stops_hit
from ..request import CapitalConfig, Fill, RuleConfig, StopConfig
from .base import EngineResult, compute_metrics, exit_reasons, finalise_trades


def _optional_stop(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def _flatten_at_close(strategy) -> None:
    """Close every open trade at this bar's close and restamp the equity point.

    A market ``position.close()`` would wait for the next bar's open. Calling the
    broker here, after that bar's open has already been processed, stamps the
    exit on the last regular-session bar.
    """
    broker = strategy._broker
    price = float(strategy.data.Close[-1])
    for trade in list(broker.trades):
        broker._close_trade(trade, price, broker._i)
    broker._equity[broker._i] = broker.equity


def _build_strategy(capital: CapitalConfig):
    from backtesting import Strategy

    class SignalReplay(Strategy):
        entries: np.ndarray = np.array([])
        exits: np.ndarray = np.array([])
        short_entries: np.ndarray = np.array([])
        short_exits: np.ndarray = np.array([])
        long_stop: np.ndarray = np.array([])
        short_stop: np.ndarray = np.array([])
        flatten: np.ndarray = np.array([])

        def init(self) -> None:
            self._entries = self.I(lambda: self.entries, name="entries", plot=False)
            self._exits = self.I(lambda: self.exits, name="exits", plot=False)
            self._short_entries = self.I(
                lambda: self.short_entries, name="short_entries", plot=False
            )
            self._short_exits = self.I(lambda: self.short_exits, name="short_exits", plot=False)
            # Stops and the flatten flag are not indicators. Wrapping them in
            # ``I`` would treat a leading NaN (ATR still warming up) as a
            # lookback and delay the first tradable bar.

        def _at(self, values):
            return values[len(self.data) - 1]

        def _size(self):
            if capital.sizing == "percent_equity":
                # backtesting.py reads a float in (0, 1) as a fraction of equity.
                return min(capital.size_value, 0.9999)
            if capital.sizing == "fixed_cash":
                return max(1, int(capital.size_value // self.data.Close[-1]))
            return max(1, int(capital.size_value))

        def next(self) -> None:
            # Last regular-session bar under next-open + no_overnight: flatten at
            # this close. Doing it before a new entry means we cannot open a hold
            # that would only be flat after the next session's open.
            if bool(self._at(self.flatten)) and self.position:
                _flatten_at_close(self)
                return

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
                    sl = _optional_stop(self._at(self.long_stop))
                    if sl is None:
                        self.buy(size=self._size())
                    else:
                        self.buy(size=self._size(), sl=sl)
                elif capital.allow_short and self._short_entries[-1]:
                    sl = _optional_stop(self._at(self.short_stop))
                    if sl is None:
                        self.sell(size=self._size())
                    else:
                        self.sell(size=self._size(), sl=sl)

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
        stop: StopConfig | None = None,
        fill: Fill = "close",
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
        nan = np.full(len(data), np.nan)
        levels = order_stop_prices(frame, stop, fill)
        if levels is None:
            long_stop = short_stop = nan
        else:
            long_stop = levels[0].reindex(data.index).to_numpy(dtype="float64")
            short_stop = levels[1].reindex(data.index).to_numpy(dtype="float64")
        flatten = flatten_at_close_mask(frame, fill, rules).reindex(data.index).fillna(False)

        strategy = _build_strategy(capital)
        # backtesting.py's ``spread`` is charged on entry only, so slippage is charged as
        # a per-side fee instead, which matches vectorbt's slipped entry and exit prices.
        slippage = capital.slippage_bps / 10_000.0
        backtest = Backtest(
            data,
            strategy,
            cash=capital.initial_cash,
            commission=capital.commission + slippage,
            trade_on_close=(fill == "close"),
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
            long_stop=long_stop,
            short_stop=short_stop,
            flatten=flatten.to_numpy(dtype=bool),
        )

        distance = distance_on_entry_bar(frame, stop, fill)
        trades = self._trades(stats, frame, signals, rules, symbol, slippage, distance)
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
        distance: pd.Series | None,
    ) -> pd.DataFrame:
        records = stats["_trades"]
        if records is None or len(records) == 0:
            return finalise_trades(None, symbol)

        final_bar = frame.index[-1]
        exit_time = pd.to_datetime(records["ExitTime"], utc=True)
        entry_time = pd.to_datetime(records["EntryTime"], utc=True)
        # finalize_trades closes anything still open on the final bar, but a real exit
        # signal can also land there; only the former counts as end_of_data.
        closed_by_signal = bool(
            signals.exits.get(final_bar, False)
            or (signals.short_exits is not None and signals.short_exits.get(final_bar, False))
        )
        direction = np.sign(records["Size"].to_numpy())
        raw_entry = records["EntryPrice"].astype("float64")
        raw_exit = records["ExitPrice"].astype("float64")
        side = np.where(records["Size"].to_numpy() >= 0, "long", "short")
        # Classify on the broker's fill, before the cosmetic slippage adjustment.
        stopped = stops_hit(
            entry_time,
            pd.Series(side),
            raw_entry,
            raw_exit,
            distance,
        )
        # A stop on the final bar is a real exit. finalize_trades is the only
        # path that should read as end_of_data.
        still_open = (
            exit_time.eq(final_bar).to_numpy()
            & (not closed_by_signal)
            & ~np.asarray(stopped, dtype=bool)
        )
        out = pd.DataFrame(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": side,
                "size": records["Size"].abs(),
                "entry_price": raw_entry * (1 + direction * slippage),
                "exit_price": raw_exit * (1 - direction * slippage),
                "pnl": records["PnL"],
                "return_pct": records["ReturnPct"],
            }
        )
        out["exit_reason"] = exit_reasons(
            out["exit_time"], still_open, frame, rules, stopped=stopped
        )
        return finalise_trades(out, symbol)
