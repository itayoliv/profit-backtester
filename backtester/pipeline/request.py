"""The agent-facing request schema. Unknown fields are rejected."""

from __future__ import annotations

import itertools
from datetime import date, datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from ..data.timeframes import TIMEFRAME_NAMES

Engine = Literal["vectorbt", "backtesting"]
Sizing = Literal["percent_equity", "fixed_cash", "fixed_shares"]
Fill = Literal["close", "next_open"]
StopType = Literal["fixed", "atr"]

MAX_GRID_RUNS = 200
"""Upper bound on tickers x parameter combinations in one request."""


class CapitalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_cash: float = Field(default=500.0, gt=0)
    sizing: Sizing = "percent_equity"
    size_value: float = Field(default=1.0, gt=0)
    """Fraction of equity, cash amount, or share count depending on ``sizing``."""
    commission: float = Field(default=0.0, ge=0, lt=1)
    """Per-side commission as a fraction of notional."""
    slippage_bps: float = Field(default=1.0, ge=0)
    allow_short: bool = False
    allow_fractional: bool = False
    """When false, every order is rounded down to a whole number of shares. When true,
    vectorbt may buy fractions of a share. The backtesting engine always buys whole
    shares, so a request that sets this with that engine is rejected."""

    @model_validator(mode="after")
    def _check_percent(self) -> "CapitalConfig":
        if self.sizing == "percent_equity" and self.size_value > 1:
            raise ValueError("size_value must be <= 1 when sizing is percent_equity")
        return self


class RuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_trades_per_day: int | None = Field(default=None, ge=1)
    no_overnight: bool = False
    indicators_rth_only: bool = False
    """When false (the default), indicators see one continuous regular-session series,
    which is the historical sample: extended-hours bars are removed first, so a
    moving average can use the previous session. When true, indicators are rebuilt
    on each regular session alone (09:30–16:00 ET), so signals can move. Flatten
    rules are unchanged either way."""


class StopConfig(BaseModel):
    """Protective stop attached by the engines after an entry. Omitted means no stop.

    ``fixed`` sits ``dollars`` under a long entry and the same distance above a
    short. ``atr`` uses ``atr_mult`` times the Wilder ATR of ``atr_period`` bars.
    The stop is live once the entry has filled; a hit is labeled ``stop`` on the
    trade, including when ``no_overnight`` would otherwise flatten at the close.
    """

    model_config = ConfigDict(extra="forbid")

    type: StopType
    atr_mult: float | None = Field(default=None, gt=0)
    atr_period: int | None = Field(default=None, ge=1, le=2000)
    dollars: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _matches_type(self) -> "StopConfig":
        if self.type == "fixed":
            if self.dollars is None:
                raise ValueError("fixed stop requires dollars")
            if self.atr_mult is not None or self.atr_period is not None:
                raise ValueError("fixed stop does not take atr_mult or atr_period")
        else:
            if self.atr_mult is None or self.atr_period is None:
                raise ValueError("atr stop requires atr_mult and atr_period")
            if self.dollars is not None:
                raise ValueError("atr stop does not take dollars")
        return self


class RegimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = "SPY"
    timeframe: str = "1Day"
    indicator: Literal["sma", "ema"] = "sma"
    period: int = Field(default=200, ge=2, le=2000)
    condition: Literal["above", "below"] = "above"

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("timeframe")
    @classmethod
    def _known_timeframe(cls, value: str) -> str:
        if value not in TIMEFRAME_NAMES:
            raise ValueError(f"timeframe must be one of {', '.join(TIMEFRAME_NAMES)}")
        return value


class FilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weekdays: list[int] | None = None
    """Monday=0 .. Sunday=6, in exchange-local time. None means every weekday."""
    regime: RegimeConfig | None = None

    @field_validator("weekdays")
    @classmethod
    def _valid_weekdays(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("weekdays must be non-empty when provided")
        bad = [d for d in value if d < 0 or d > 6]
        if bad:
            raise ValueError(f"weekdays must be 0-6 (Monday=0), got {bad}")
        return sorted(set(value))


def describe_job(tickers: list[str], start: date, end: date, strategy: str) -> str:
    """A job id an agent can read: ticker, window, strategy, and a short unique suffix.

    Example: ``AAPL_2023-01-01_2024-01-01_sma_cross_a1b2c3d4``. Several tickers are
    joined with ``-``. The suffix keeps a repeat of the same run from colliding.
    """
    symbols = "-".join(tickers)
    return f"{symbols}_{start.isoformat()}_{end.isoformat()}_{strategy}_{uuid4().hex[:8]}"


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = ""
    """Empty means :func:`describe_job` fills this in. An explicit id is kept as given."""
    parent_job_id: str | None = None
    """Set by the worker when fanning a multi-ticker request out; agents leave it unset."""
    source: str = "agent"
    tickers: list[str]
    timeframe: str
    start: date
    end: date
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    params_grid: dict[str, list[Any]] | None = None
    """A parameter sweep: one child job per combination, each merged over ``params``.
    Combinations the strategy rejects (say fast >= slow) are skipped, not fatal."""
    engine: Engine = "vectorbt"
    fill: Fill = "close"
    """``close`` fills on the signal bar (historical). ``next_open`` fills the order
    at the next bar's open. With ``rules.no_overnight``, an exit that would land in
    the next session fills at the last regular-session bar's close instead."""
    stop: StopConfig | None = None
    """Optional protective stop. ``null`` leaves exits entirely to the strategy."""
    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    rules: RuleConfig = Field(default_factory=RuleConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)

    @field_validator("tickers")
    @classmethod
    def _clean_tickers(cls, value: list[str]) -> list[str]:
        seen: list[str] = []
        for raw in value:
            symbol = raw.strip().upper()
            if symbol and symbol not in seen:
                seen.append(symbol)
        if not seen:
            raise ValueError("tickers must contain at least one symbol")
        return seen

    @field_validator("timeframe")
    @classmethod
    def _known_timeframe(cls, value: str) -> str:
        if value not in TIMEFRAME_NAMES:
            raise ValueError(f"timeframe must be one of {', '.join(TIMEFRAME_NAMES)}")
        return value

    @field_validator("job_id")
    @classmethod
    def _safe_job_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return ""
        if not all(c.isalnum() or c in "-_." for c in value):
            raise ValueError("job_id may only contain letters, digits, '-', '_' and '.'")
        return value

    @field_validator("params_grid")
    @classmethod
    def _clean_grid(cls, value: dict[str, list[Any]] | None) -> dict[str, list[Any]] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("params_grid must name at least one parameter")
        cleaned: dict[str, list[Any]] = {}
        for name, values in value.items():
            if not values:
                raise ValueError(f"params_grid.{name} must list at least one value")
            unique: list[Any] = []
            for item in values:
                if item not in unique:
                    unique.append(item)
            cleaned[name] = unique
        return cleaned

    @model_validator(mode="after")
    def _check_window_and_strategy(self) -> "BacktestRequest":
        if self.start >= self.end:
            raise ValueError("start must be strictly before end")
        if self.capital.allow_fractional and self.engine == "backtesting":
            raise ValueError(
                "allow_fractional is only supported by the vectorbt engine; "
                "the backtesting engine buys whole shares"
            )
        from ..strategies.registry import get_strategy

        try:
            strategy = get_strategy(self.strategy)
        except KeyError as exc:
            raise ValueError(str(exc)) from None
        if self.params_grid is None:
            strategy.parse_params(self.params)
        else:
            self._check_grid(strategy)
        if not self.job_id:
            self.job_id = describe_job(self.tickers, self.start, self.end, self.strategy)
        return self

    def _check_grid(self, strategy: Any) -> None:
        grid = self.params_grid or {}
        known = strategy.Params.model_fields
        unknown = [name for name in grid if name not in known]
        if unknown:
            raise ValueError(
                f"params_grid has unknown parameters {unknown}; "
                f"{self.strategy} accepts {sorted(known)}"
            )
        overlap = [name for name in grid if name in self.params]
        if overlap:
            raise ValueError(f"{overlap} appear in both params and params_grid; keep each in one")
        combos = 1
        for values in grid.values():
            combos *= len(values)
        if combos * len(self.tickers) > MAX_GRID_RUNS:
            raise ValueError(
                f"{combos} combinations x {len(self.tickers)} tickers exceeds the "
                f"{MAX_GRID_RUNS} run limit; split the sweep"
            )
        if not self.param_sets():
            first = self.skipped_combinations()[0]
            raise ValueError(
                f"every params_grid combination is invalid, e.g. {first['params']}: {first['reason']}"
            )

    # --- parameter sweeps --------------------------------------------------
    def _grid_overrides(self) -> list[tuple[dict[str, Any], str | None]]:
        """Each grid combination and why the strategy rejects it (None when valid)."""
        if not self.params_grid:
            return [({}, None)]
        from ..strategies.registry import get_strategy

        strategy = get_strategy(self.strategy)
        names = list(self.params_grid)
        out: list[tuple[dict[str, Any], str | None]] = []
        for values in itertools.product(*(self.params_grid[n] for n in names)):
            overrides = dict(zip(names, values))
            try:
                strategy.parse_params({**self.params, **overrides})
            except (PydanticValidationError, ValueError, TypeError) as exc:
                errors = exc.errors() if isinstance(exc, PydanticValidationError) else None
                reason = "; ".join(e["msg"] for e in errors) if errors else str(exc)
                out.append((overrides, reason))
            else:
                out.append((overrides, None))
        return out

    def param_sets(self) -> list[dict[str, Any]]:
        """The full ``params`` of every run this request makes, one per valid combination."""
        return [
            {**self.params, **overrides}
            for overrides, reason in self._grid_overrides()
            if reason is None
        ]

    def skipped_combinations(self) -> list[dict[str, Any]]:
        return [
            {"params": overrides, "reason": reason}
            for overrides, reason in self._grid_overrides()
            if reason is not None
        ]

    def _grid_label(self, overrides: dict[str, Any]) -> str:
        grid = self.params_grid or {}
        varied = [name for name in grid if len(grid[name]) > 1] or list(grid)
        parts = [f"{name}{_label_value(overrides[name])}" for name in varied]
        return "".join(c for c in "_".join(parts) if c.isalnum() or c in "-_.")

    # --- derived -----------------------------------------------------------
    @property
    def start_ts(self) -> datetime:
        return datetime(self.start.year, self.start.month, self.start.day, tzinfo=timezone.utc)

    @property
    def end_ts(self) -> datetime:
        return datetime(self.end.year, self.end.month, self.end.day, tzinfo=timezone.utc)

    @property
    def is_batch(self) -> bool:
        return len(self.tickers) > 1 or self.params_grid is not None

    def child_for(self, ticker: str, overrides: dict[str, Any] | None = None) -> "BacktestRequest":
        """One single-ticker, single-parameter-set request, carrying this job as its parent.

        Named ``<job_id>-<TICKER>`` for a ticker batch, ``<job_id>-<combo>`` for a sweep
        on one ticker, and ``<job_id>-<TICKER>-<combo>`` for both.
        """
        parts = [self.job_id]
        if self.params_grid is None or len(self.tickers) > 1:
            parts.append(ticker)
        if self.params_grid is not None:
            parts.append(self._grid_label(overrides or {}))
        return self.model_copy(
            update={
                "job_id": "-".join(parts),
                "parent_job_id": self.job_id,
                "tickers": [ticker],
                "params": {**self.params, **(overrides or {})},
                "params_grid": None,
            }
        )

    def children(self) -> list["BacktestRequest"]:
        return [
            self.child_for(ticker, overrides)
            for overrides, reason in self._grid_overrides()
            if reason is None
            for ticker in self.tickers
        ]


def _label_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def parse_request(payload: dict[str, Any]) -> BacktestRequest:
    """Validate a raw dict, raising our ``ValidationError`` with flat details."""
    from pydantic import ValidationError as PydanticValidationError

    from ..errors import ValidationError

    try:
        return BacktestRequest(**payload)
    except PydanticValidationError as exc:
        details = [
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ValidationError("request failed validation", details) from None
    except TypeError as exc:
        raise ValidationError("request must be a JSON object", [str(exc)]) from None
