"""Shared test fixtures: synthetic bars written straight into the Parquet cache.

Seeding the cache means the whole pipeline, including worker subprocesses, runs with
no Alpaca credentials and no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.config import Settings
from backtester.data.cache import CacheKey, ParquetCache


def synthetic_bars(
    start: str = "2023-01-03",
    periods: int = 400,
    freq: str = "1D",
    seed: int = 7,
    start_price: float = 100.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC", name="timestamp")
    close = start_price + np.cumsum(rng.normal(0.05, 1.0, periods))
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e5, 5e5, periods).astype("float64"),
            "trade_count": 1000.0,
            "vwap": close,
        },
        index=index,
    )


def seed_cache(
    settings: Settings,
    symbol: str,
    timeframe: str = "1Day",
    bars: pd.DataFrame | None = None,
    covered: tuple[str, str] = ("2022-01-01", "2030-01-01"),
) -> pd.DataFrame:
    """Write bars into the cache with a wide covered window so reads never download."""
    frame = bars if bars is not None else synthetic_bars()
    cache = ParquetCache(settings.cache_dir)
    key = CacheKey(settings.feed, settings.adjustment, timeframe, symbol)
    cache.store(
        key,
        frame,
        (pd.Timestamp(covered[0], tz="UTC"), pd.Timestamp(covered[1], tz="UTC")),
    )
    return frame


def base_request(**overrides) -> dict:
    payload = {
        "tickers": ["AAPL"],
        "timeframe": "1Day",
        "start": "2023-01-03",
        "end": "2024-01-01",
        "strategy": "sma_cross",
        "params": {"fast": 5, "slow": 20},
        "engine": "vectorbt",
    }
    payload.update(overrides)
    return payload
