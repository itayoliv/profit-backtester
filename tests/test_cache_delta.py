"""Delta maths and cache merging. No network, no credentials."""

from __future__ import annotations

import pandas as pd
import pytest

from backtester.data.cache import CacheKey, ParquetCache, chunk_range, missing_ranges
from backtester.data.manager import DataManager
from backtester.errors import NoDataError
from helpers import seed_cache, synthetic_bars


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def test_no_coverage_downloads_everything():
    assert missing_ranges(ts("2023-01-01"), ts("2023-02-01"), None) == [
        (ts("2023-01-01"), ts("2023-02-01"))
    ]


def test_fully_covered_downloads_nothing():
    covered = (ts("2022-01-01"), ts("2024-01-01"))
    assert missing_ranges(ts("2023-01-01"), ts("2023-02-01"), covered) == []


def test_delta_before_and_after():
    covered = (ts("2023-02-01"), ts("2023-03-01"))
    gaps = missing_ranges(ts("2023-01-01"), ts("2023-04-01"), covered)
    assert gaps == [
        (ts("2023-01-01"), ts("2023-02-01")),
        (ts("2023-03-01"), ts("2023-04-01")),
    ]


def test_delta_only_after():
    covered = (ts("2023-01-01"), ts("2023-02-01"))
    assert missing_ranges(ts("2023-01-15"), ts("2023-03-01"), covered) == [
        (ts("2023-02-01"), ts("2023-03-01"))
    ]


def test_empty_window_is_not_a_gap():
    assert missing_ranges(ts("2023-01-01"), ts("2023-01-01"), None) == []


def test_chunking_splits_on_the_chunk_size():
    chunks = chunk_range(ts("2023-01-01"), ts("2023-04-01"), pd.Timedelta(days=30))
    assert len(chunks) == 3
    assert chunks[0][0] == ts("2023-01-01")
    assert chunks[-1][1] == ts("2023-04-01")
    # Contiguous with no overlap.
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))


def test_merge_dedupes_and_sorts(settings):
    cache = ParquetCache(settings.cache_dir)
    key = CacheKey("iex", "all", "1Day", "AAPL")
    first = synthetic_bars(start="2023-01-03", periods=10)
    second = synthetic_bars(start="2023-01-08", periods=10, seed=99)

    cache.store(key, first, (ts("2023-01-03"), ts("2023-01-13")))
    merged = cache.store(key, second, (ts("2023-01-08"), ts("2023-01-18")))

    assert merged.index.is_monotonic_increasing
    assert not merged.index.has_duplicates
    assert len(merged) == 15
    # The later write wins on the overlapping days.
    assert merged.loc[second.index[0], "close"] == pytest.approx(second.iloc[0]["close"])
    assert cache.coverage(key) == (ts("2023-01-03"), ts("2023-01-18"))


def test_cached_window_needs_no_client(settings):
    """A full cache hit must not construct an Alpaca client, so no credentials."""
    seed_cache(settings, "AAPL")
    result = DataManager(settings).get_bars("aapl", "1Day", "2023-02-01", "2023-06-01")

    assert result.coverage["cache_hit"] is True
    assert result.coverage["downloaded_ranges"] == []
    assert result.coverage["symbol"] == "AAPL"
    assert len(result.frame) == result.coverage["bar_count"] > 0


def test_covered_window_with_no_bars_raises_no_data(settings):
    """Inside the covered range but before the first bar: nothing to download, no data."""
    seed_cache(settings, "AAPL")
    with pytest.raises(NoDataError):
        DataManager(settings).get_bars("AAPL", "1Day", "2022-06-01", "2022-07-01")
