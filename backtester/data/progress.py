"""Phase and download progress with a rolling-average ETA."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

# Each phase occupies a slice of the overall 0-100 progress bar. Fetching dominates
# because it is the only phase that can take minutes.
PHASE_SPAN: dict[str, tuple[float, float]] = {
    "queued": (0.0, 0.0),
    "fetching_data": (0.0, 60.0),
    "filtering": (60.0, 70.0),
    "running_strategy": (70.0, 90.0),
    "applying_rules": (90.0, 95.0),
    "reporting": (95.0, 100.0),
    "done": (100.0, 100.0),
    "failed": (100.0, 100.0),
}


@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    progress_pct: float
    eta_seconds: float | None
    current_symbol: str | None
    message: str | None


Callback = Callable[[ProgressEvent], None]


class ProgressReporter:
    """Collects phase transitions and chunk completions into ``ProgressEvent``s.

    ETA is estimated from a rolling mean of the last few chunk durations, which keeps
    it responsive when Alpaca throttles mid-download.
    """

    def __init__(self, callback: Callback | None = None, symbol: str | None = None) -> None:
        self._callback = callback
        self.symbol = symbol
        self.phase = "queued"
        self._chunks_total = 0
        self._chunks_done = 0
        self._durations: deque[float] = deque(maxlen=5)
        self._chunk_started: float | None = None
        self.rows = 0

    def bind(self, symbol: str) -> "ProgressReporter":
        self.symbol = symbol
        return self

    # --- phases -----------------------------------------------------------
    def set_phase(self, phase: str, message: str | None = None) -> None:
        self.phase = phase
        if phase != "fetching_data":
            self._chunks_total = 0
            self._chunks_done = 0
        self._emit(message)

    # --- download chunks --------------------------------------------------
    def begin_chunks(self, total: int) -> None:
        self._chunks_total = total
        self._chunks_done = 0
        self._durations.clear()
        self._chunk_started = time.monotonic()
        self._emit(f"0/{total} chunks")

    def complete_chunk(self, rows: int = 0) -> None:
        now = time.monotonic()
        if self._chunk_started is not None:
            self._durations.append(now - self._chunk_started)
        self._chunk_started = now
        self._chunks_done += 1
        self.rows += rows
        self._emit(f"{self._chunks_done}/{self._chunks_total} chunks, {self.rows} rows")

    # --- derived values ---------------------------------------------------
    @property
    def eta_seconds(self) -> float | None:
        remaining = self._chunks_total - self._chunks_done
        if remaining <= 0 or not self._durations:
            return None
        mean = sum(self._durations) / len(self._durations)
        return round(mean * remaining, 1)

    @property
    def progress_pct(self) -> float:
        low, high = PHASE_SPAN.get(self.phase, (0.0, 100.0))
        if self._chunks_total > 0:
            fraction = min(1.0, self._chunks_done / self._chunks_total)
        else:
            fraction = 0.0
        return round(low + (high - low) * fraction, 1)

    def _emit(self, message: str | None = None) -> None:
        if self._callback is None:
            return
        self._callback(
            ProgressEvent(
                phase=self.phase,
                progress_pct=self.progress_pct,
                eta_seconds=self.eta_seconds,
                current_symbol=self.symbol,
                message=message,
            )
        )
