"""Per-job state files and the aggregated status.json the agent polls."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from ..config import TERMINAL_PHASES, Settings
from ..jsonio import iso, parse_iso, read_json, utcnow, write_json

SUMMARY_FIELDS = (
    "job_id",
    "parent_job_id",
    "phase",
    "progress_pct",
    "eta_seconds",
    "current_symbol",
    "heartbeat_at",
    "updated_at",
    "attempt",
    "source",
    "tickers",
    "strategy",
    "engine",
    "timeframe",
    "error",
)


class StateStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # --- per job ----------------------------------------------------------
    def read(self, job_id: str) -> dict[str, Any] | None:
        return read_json(self.settings.job_state_file(job_id))

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        """Merge ``fields`` into the job's state file and stamp the heartbeat."""
        now = iso(utcnow())
        record = self.read(job_id) or {
            "job_id": job_id,
            "parent_job_id": None,
            "phase": "queued",
            "progress_pct": 0.0,
            "eta_seconds": None,
            "current_symbol": None,
            "attempt": 1,
            "error": None,
            "created_at": now,
        }
        record.update(fields)
        record["job_id"] = job_id
        record["heartbeat_at"] = fields.get("heartbeat_at", now)
        record["updated_at"] = now
        write_json(self.settings.job_state_file(job_id), record)
        return record

    def heartbeat(self, job_id: str) -> None:
        record = self.read(job_id)
        if record is None or record.get("phase") in TERMINAL_PHASES:
            return
        record["heartbeat_at"] = iso(utcnow())
        write_json(self.settings.job_state_file(job_id), record)

    def is_stale(self, record: dict[str, Any]) -> bool:
        if record.get("phase") in TERMINAL_PHASES:
            return False
        beat = parse_iso(record.get("heartbeat_at"))
        if beat is None:
            return True
        return utcnow() - beat > timedelta(seconds=self.settings.stale_after_s)

    # --- aggregate --------------------------------------------------------
    def all_records(self) -> list[dict[str, Any]]:
        directory = self.settings.job_state_dir
        if not directory.exists():
            return []
        records = []
        for path in directory.glob("*.json"):
            record = read_json(path)
            if isinstance(record, dict) and record.get("job_id"):
                records.append(record)
        return records

    def children_of(self, job_id: str) -> list[dict[str, Any]]:
        return [r for r in self.all_records() if r.get("parent_job_id") == job_id]

    def write_status(self, worker_alive: bool = True) -> dict[str, Any]:
        running, queued, done, failed = [], [], [], []
        for record in self.all_records():
            entry = {k: record.get(k) for k in SUMMARY_FIELDS}
            phase = record.get("phase")
            if phase == "done":
                done.append(entry)
            elif phase == "failed":
                failed.append(entry)
            elif phase == "queued":
                queued.append(entry)
            else:
                running.append(entry)

        def newest(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(items, key=lambda r: r.get("updated_at") or "", reverse=True)

        limit = self.settings.recent_limit
        status = {
            "generated_at": iso(utcnow()),
            "worker_heartbeat_at": iso(utcnow()) if worker_alive else None,
            "running": newest(running),
            "queued": newest(queued),
            "recent_done": newest(done)[:limit],
            "recent_failed": newest(failed)[:limit],
        }
        if not worker_alive:
            existing = read_json(self.settings.status_file) or {}
            status["worker_heartbeat_at"] = existing.get("worker_heartbeat_at")
        write_json(self.settings.status_file, status)
        return status

    def read_status(self) -> dict[str, Any]:
        return read_json(self.settings.status_file) or {
            "generated_at": None,
            "worker_heartbeat_at": None,
            "running": [],
            "queued": [],
            "recent_done": [],
            "recent_failed": [],
        }
