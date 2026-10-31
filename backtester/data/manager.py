"""Cache-first bar access: download only the missing deltas, then merge."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from ..config import Settings
from ..errors import NoDataError
from ..jsonio import iso
from ..logging import get_logger
from .alpaca_client import AlpacaBarClient
from .cache import CacheKey, ParquetCache, Range, chunk_range, missing_ranges
from .progress import ProgressReporter
from .timeframes import TimeframeSpec, get_timeframe

log = get_logger("data.manager")

# The newest slice of a cached window is re-fetched because bars near the present can
# still be filling in. Everything older is trusted and never re-downloaded.
TAIL_REFRESH = timedelta(hours=24)


@dataclass
class BarsResult:
    frame: pd.DataFrame
    coverage: dict[str, Any] = field(default_factory=dict)


def to_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


class DataManager:
    def __init__(self, settings: Settings, client: AlpacaBarClient | None = None) -> None:
        self.settings = settings
        self.cache = ParquetCache(settings.cache_dir)
        self._client = client

    @property
    def client(self) -> AlpacaBarClient:
        if self._client is None:
            self._client = AlpacaBarClient(self.settings)
        return self._client

    def get_bars(
        self,
        symbol: str,
        timeframe: str | TimeframeSpec,
        start: Any,
        end: Any,
        reporter: ProgressReporter | None = None,
    ) -> BarsResult:
        spec = timeframe if isinstance(timeframe, TimeframeSpec) else get_timeframe(timeframe)
        symbol = symbol.strip().upper()
        req_start = to_utc(start)
        req_end = min(to_utc(end), pd.Timestamp.now(tz="UTC"))
        if req_end <= req_start:
            raise NoDataError(f"{symbol}: requested window is empty after clamping to now")

        key = CacheKey(self.client.feed, self.client.adjustment, spec.name, symbol)
        downloaded: list[Range] = []

        with self.cache.lock(key):
            gaps = missing_ranges(req_start, req_end, self._effective_coverage(key))
            chunks = [c for gap in gaps for c in chunk_range(gap[0], gap[1], spec.chunk)]

            if chunks:
                if reporter is not None:
                    reporter.begin_chunks(len(chunks))
                frames: list[pd.DataFrame] = []
                for chunk_start, chunk_end in chunks:
                    frame = self.client.fetch_chunk(symbol, spec, chunk_start, chunk_end)
                    frames.append(frame)
                    if reporter is not None:
                        reporter.complete_chunk(len(frame))
                downloaded = gaps
                fetched = pd.concat(frames) if frames else None
                has_cached = self.cache.path(key).exists()
                if (fetched is not None and not fetched.empty) or has_cached:
                    self.cache.store(key, fetched, (req_start, req_end))

            frame = self.cache.read(key, req_start, req_end)

        if frame.empty:
            raise NoDataError(
                f"{symbol}: no {spec.name} bars between {req_start.date()} and {req_end.date()} "
                f"on the {key.feed} feed"
            )

        return BarsResult(
            frame=frame,
            coverage={
                "symbol": symbol,
                "timeframe": spec.name,
                "feed": key.feed,
                "adjustment": key.adjustment,
                "requested_start": iso(req_start.to_pydatetime()),
                "requested_end": iso(req_end.to_pydatetime()),
                "first_bar": iso(frame.index[0].to_pydatetime()),
                "last_bar": iso(frame.index[-1].to_pydatetime()),
                "bar_count": int(len(frame)),
                "cache_hit": not downloaded,
                "downloaded_ranges": [
                    [iso(a.to_pydatetime()), iso(b.to_pydatetime())] for a, b in downloaded
                ],
            },
        )

    def _effective_coverage(self, key: CacheKey) -> Range | None:
        covered = self.cache.coverage(key)
        if covered is None:
            return None
        start, end = covered
        cutoff = pd.Timestamp.now(tz="UTC") - TAIL_REFRESH
        if end > cutoff:
            end = max(start, cutoff)
        return (start, end)
