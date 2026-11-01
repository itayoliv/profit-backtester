"""Directory-based job queue. A job is claimed by renaming its file, which is atomic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config import Settings
from ..errors import ValidationError
from ..jsonio import read_json, write_json
from ..logging import get_logger
from ..pipeline.request import BacktestRequest
from .state import StateStore

log = get_logger("jobs.queue")


class JobQueue:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = StateStore(settings)
        settings.ensure_dirs()

    # --- submission -------------------------------------------------------
    def submit(self, request: BacktestRequest) -> str:
        if self.locate(request.job_id) is not None or self.state.read(request.job_id):
            raise ValidationError(
                f"job_id {request.job_id!r} already exists",
                [f"job_id: {request.job_id!r} is already in use"],
            )
        self._enqueue(request)
        return request.job_id

    def _enqueue(self, request: BacktestRequest) -> None:
        write_json(self.settings.inbox_dir / f"{request.job_id}.json", request.model_dump(mode="json"))
        self.state.update(
            request.job_id,
            phase="queued",
            progress_pct=0.0,
            eta_seconds=None,
            current_symbol=None,
            parent_job_id=request.parent_job_id,
            source=request.source,
            tickers=request.tickers,
            strategy=request.strategy,
            engine=request.engine,
            timeframe=request.timeframe,
            error=None,
        )

    def fan_out(self, parent: BacktestRequest) -> list[str]:
        """Queue one child per ticker. The parent itself is not executed."""
        children = parent.children()
        for child in children:
            self._enqueue(child)
        return [child.job_id for child in children]

    # --- lifecycle --------------------------------------------------------
    def claim_next(self) -> BacktestRequest | None:
        """Move the oldest inbox job into ``running/`` and return it.

        The rename is the claim: if two workers race, only one rename succeeds.
        """
        for path in sorted(self.settings.inbox_dir.glob("*.json"), key=_mtime):
            target = self.settings.running_dir / path.name
            try:
                os.rename(path, target)
            except OSError:
                continue
            payload = read_json(target)
            if not isinstance(payload, dict):
                log.error("discarding unreadable job file %s", target.name)
                os.replace(target, self.settings.failed_dir / target.name)
                continue
            return BacktestRequest(**payload)
        return None

    def finish(self, job_id: str, failed: bool = False) -> None:
        source = self.settings.running_dir / f"{job_id}.json"
        destination = (self.settings.failed_dir if failed else self.settings.done_dir) / source.name
        if source.exists():
            os.replace(source, destination)

    def requeue(self, job_id: str, attempt: int) -> None:
        source = self.settings.running_dir / f"{job_id}.json"
        if source.exists():
            os.replace(source, self.settings.inbox_dir / source.name)
        self.state.update(job_id, phase="queued", attempt=attempt, progress_pct=0.0, error=None)

    # --- lookups ----------------------------------------------------------
    def running_ids(self) -> list[str]:
        return [p.stem for p in self.settings.running_dir.glob("*.json")]

    def has_work(self) -> bool:
        """True while any job is waiting or not yet finished, parents included."""
        return any(self.settings.inbox_dir.glob("*.json")) or any(
            self.settings.running_dir.glob("*.json")
        )

    def locate(self, job_id: str) -> Path | None:
        for directory in (
            self.settings.inbox_dir,
            self.settings.running_dir,
            self.settings.done_dir,
            self.settings.failed_dir,
        ):
            candidate = directory / f"{job_id}.json"
            if candidate.exists():
                return candidate
        return None

    def load_request(self, job_id: str) -> BacktestRequest | None:
        path = self.locate(job_id)
        if path is None:
            return None
        payload: Any = read_json(path)
        return BacktestRequest(**payload) if isinstance(payload, dict) else None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
