"""Alpaca historical-bar access: one chunk per call, with retry and backoff."""

from __future__ import annotations

import random
import time
from datetime import datetime
from typing import Any

import pandas as pd

from ..config import Settings
from ..errors import FeedNotPermittedError, MissingCredentialsError
from ..logging import get_logger

log = get_logger("data.alpaca")

BAR_COLUMNS = ("open", "high", "low", "close", "volume", "trade_count", "vwap")

MAX_ATTEMPTS = 5
BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0


def empty_bars() -> pd.DataFrame:
    index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame(
        {name: pd.Series([], dtype="float64") for name in BAR_COLUMNS}, index=index
    )


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(getattr(exc, "_http_error", None), "response", None)
    return getattr(response, "status_code", None)


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(getattr(exc, "_http_error", None), "response", None)
    raw = getattr(response, "headers", {}).get("Retry-After") if response is not None else None
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _is_subscription_error(exc: BaseException, status: int | None) -> bool:
    if status not in (401, 403):
        return False
    text = str(exc).lower()
    return "subscription" in text or "not permitted" in text or "forbidden" in text


class AlpacaBarClient:
    """Thin wrapper over ``StockHistoricalDataClient`` returning tidy UTC frames."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None

    @property
    def feed(self) -> str:
        return self._settings.feed

    @property
    def adjustment(self) -> str:
        return self._settings.adjustment

    def _ensure_client(self) -> Any:
        if self._client is None:
            if not self._settings.has_credentials:
                raise MissingCredentialsError(
                    "ALPACA_API_KEY and ALPACA_SECRET_KEY are not set; copy .env.example to .env"
                )
            from alpaca.data.historical import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(
                api_key=self._settings.alpaca_api_key,
                secret_key=self._settings.alpaca_secret_key,
            )
        return self._client

    def fetch_chunk(
        self,
        symbol: str,
        timeframe: Any,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch one time slice. Retries rate limits and transient server errors."""
        from alpaca.common.exceptions import APIError
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest

        client = self._ensure_client()
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe.to_alpaca(),
            start=start.to_pydatetime() if hasattr(start, "to_pydatetime") else start,
            end=end.to_pydatetime() if hasattr(end, "to_pydatetime") else end,
            adjustment=Adjustment(self.adjustment),
            feed=DataFeed(self.feed),
        )

        last_error: BaseException | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self._normalise(client.get_stock_bars(request))
            except APIError as exc:
                status = _status_code(exc)
                if _is_subscription_error(exc, status):
                    raise FeedNotPermittedError(
                        f"feed {self.feed!r} is not permitted for this account"
                    ) from exc
                if status is not None and status != 429 and status < 500:
                    raise
                last_error = exc
                wait = _retry_after(exc) or self._backoff(attempt)
            except (ConnectionError, TimeoutError, OSError) as exc:
                last_error = exc
                wait = self._backoff(attempt)

            if attempt == MAX_ATTEMPTS:
                break
            log.warning(
                "alpaca request for %s failed (attempt %d/%d), retrying in %.1fs: %s",
                symbol,
                attempt,
                MAX_ATTEMPTS,
                wait,
                last_error,
            )
            time.sleep(wait)

        assert last_error is not None
        raise last_error

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(MAX_BACKOFF_S, BASE_BACKOFF_S * 2 ** (attempt - 1)) * (
            0.75 + random.random() * 0.5
        )

    @staticmethod
    def _normalise(barset: Any) -> pd.DataFrame:
        frame = getattr(barset, "df", None)
        if frame is None or len(frame) == 0:
            return empty_bars()
        frame = frame.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.droplevel("symbol")
        frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
        frame.index.name = "timestamp"
        for column in BAR_COLUMNS:
            if column not in frame.columns:
                frame[column] = float("nan")
        frame = frame.loc[:, list(BAR_COLUMNS)].astype("float64")
        return frame[~frame.index.duplicated(keep="last")].sort_index()
