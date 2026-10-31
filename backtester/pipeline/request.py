"""The agent-facing request schema. Unknown fields are rejected."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..data.timeframes import TIMEFRAME_NAMES

Engine = Literal["vectorbt", "backtesting"]
Sizing = Literal["percent_equity", "fixed_cash", "fixed_shares"]


class CapitalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_cash: float = Field(default=100_000.0, gt=0)
    sizing: Sizing = "percent_equity"
    size_value: float = Field(default=1.0, gt=0)
    """Fraction of equity, cash amount, or share count depending on ``sizing``."""
    commission: float = Field(default=0.0, ge=0, lt=1)
    """Per-side commission as a fraction of notional."""
    slippage_bps: float = Field(default=5.0, ge=0)
    allow_short: bool = False

    @model_validator(mode="after")
    def _check_percent(self) -> "CapitalConfig":
        if self.sizing == "percent_equity" and self.size_value > 1:
            raise ValueError("size_value must be <= 1 when sizing is percent_equity")
        return self


class RuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_trades_per_day: int | None = Field(default=None, ge=1)
    no_overnight: bool = False


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


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_job_id: str | None = None
    """Set by the worker when fanning a multi-ticker request out; agents leave it unset."""
    source: str = "agent"
    tickers: list[str]
    timeframe: str
    start: date
    end: date
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    engine: Engine = "vectorbt"
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
            raise ValueError("job_id must not be empty")
        if not all(c.isalnum() or c in "-_." for c in value):
            raise ValueError("job_id may only contain letters, digits, '-', '_' and '.'")
        return value

    @model_validator(mode="after")
    def _check_window_and_strategy(self) -> "BacktestRequest":
        if self.start >= self.end:
            raise ValueError("start must be strictly before end")
        from ..strategies.registry import get_strategy

        try:
            strategy = get_strategy(self.strategy)
        except KeyError as exc:
            raise ValueError(str(exc)) from None
        strategy.parse_params(self.params)
        return self

    # --- derived -----------------------------------------------------------
    @property
    def start_ts(self) -> datetime:
        return datetime(self.start.year, self.start.month, self.start.day, tzinfo=timezone.utc)

    @property
    def end_ts(self) -> datetime:
        return datetime(self.end.year, self.end.month, self.end.day, tzinfo=timezone.utc)

    @property
    def is_batch(self) -> bool:
        return len(self.tickers) > 1

    def child_for(self, ticker: str) -> "BacktestRequest":
        """One single-ticker request, carrying this job as its parent."""
        return self.model_copy(
            update={
                "job_id": f"{self.job_id}-{ticker}",
                "parent_job_id": self.job_id,
                "tickers": [ticker],
            }
        )

    def children(self) -> list["BacktestRequest"]:
        return [self.child_for(ticker) for ticker in self.tickers]


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
