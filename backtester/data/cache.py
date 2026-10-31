"""Local Parquet cache keyed by feed, adjustment, timeframe and symbol."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
from filelock import FileLock

from ..jsonio import iso, read_json, utcnow, write_json
from .alpaca_client import BAR_COLUMNS, empty_bars

Range = tuple[pd.Timestamp, pd.Timestamp]


def missing_ranges(
    req_start: pd.Timestamp,
    req_end: pd.Timestamp,
    covered: Range | None,
) -> list[Range]:
    """Return the parts of ``[req_start, req_end)`` not inside ``covered``.

    At most two ranges come back, one before and one after the covered window, because
    coverage is stored as a single contiguous interval. Coverage tracks the *requested*
    window rather than observed bars, so market holidays never look like gaps.
    """
    if req_end <= req_start:
        return []
    if covered is None:
        return [(req_start, req_end)]
    covered_start, covered_end = covered
    if covered_end <= covered_start:
        return [(req_start, req_end)]
    gaps: list[Range] = []
    if req_start < covered_start:
        gaps.append((req_start, min(req_end, covered_start)))
    if req_end > covered_end:
        gaps.append((max(req_start, covered_end), req_end))
    return gaps


def chunk_range(start: pd.Timestamp, end: pd.Timestamp, chunk: timedelta) -> list[Range]:
    """Split a range into request-sized slices."""
    chunks: list[Range] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + chunk, end)
        chunks.append((cursor, stop))
        cursor = stop
    return chunks


@dataclass(frozen=True)
class CacheKey:
    feed: str
    adjustment: str
    timeframe: str
    symbol: str

    def relative_path(self) -> Path:
        return Path(self.feed) / self.adjustment / self.timeframe / f"{self.symbol}.parquet"


class ParquetCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # --- paths ------------------------------------------------------------
    def path(self, key: CacheKey) -> Path:
        return self.root / key.relative_path()

    def meta_path(self, key: CacheKey) -> Path:
        return self.path(key).with_suffix(".meta.json")

    def lock(self, key: CacheKey) -> FileLock:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(path) + ".lock")

    # --- reads ------------------------------------------------------------
    def coverage(self, key: CacheKey) -> Range | None:
        meta = read_json(self.meta_path(key))
        if not meta or not meta.get("covered_start") or not meta.get("covered_end"):
            return None
        return (
            pd.Timestamp(meta["covered_start"]).tz_convert("UTC"),
            pd.Timestamp(meta["covered_end"]).tz_convert("UTC"),
        )

    def load(self, key: CacheKey) -> pd.DataFrame:
        path = self.path(key)
        if not path.exists():
            return empty_bars()
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
        frame.index.name = "timestamp"
        return frame.sort_index()

    def read(self, key: CacheKey, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frame = self.load(key)
        if frame.empty:
            return frame
        return frame.loc[(frame.index >= start) & (frame.index < end)]

    # --- writes -----------------------------------------------------------
    def store(
        self,
        key: CacheKey,
        new_bars: pd.DataFrame,
        requested: Range,
    ) -> pd.DataFrame:
        """Merge ``new_bars`` into the cached file and widen the covered window.

        Returns the full merged frame. Caller is expected to hold ``lock(key)``.
        """
        existing = self.load(key)
        frames = [f for f in (existing, new_bars) if f is not None and not f.empty]
        merged = (
            pd.concat(frames) if frames else empty_bars()
        )
        if not merged.empty:
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged = merged.reindex(columns=list(BAR_COLUMNS))

        previous = self.coverage(key)
        req_start, req_end = requested
        covered_start = min(req_start, previous[0]) if previous else req_start
        covered_end = max(req_end, previous[1]) if previous else req_end

        self._write_parquet(self.path(key), merged)
        write_json(
            self.meta_path(key),
            {
                "symbol": key.symbol,
                "timeframe": key.timeframe,
                "feed": key.feed,
                "adjustment": key.adjustment,
                "covered_start": iso(covered_start.to_pydatetime()),
                "covered_end": iso(covered_end.to_pydatetime()),
                "bar_count": int(len(merged)),
                "first_bar": iso(merged.index[0].to_pydatetime()) if len(merged) else None,
                "last_bar": iso(merged.index[-1].to_pydatetime()) if len(merged) else None,
                "updated_at": iso(utcnow()),
            },
        )
        return merged

    @staticmethod
    def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            frame.to_parquet(tmp, engine="pyarrow", index=True)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
