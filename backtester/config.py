"""Runtime settings, resolved from the environment and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

PHASES = (
    "queued",
    "fetching_data",
    "filtering",
    "running_strategy",
    "applying_rules",
    "reporting",
    "done",
    "failed",
)
TERMINAL_PHASES = frozenset({"done", "failed"})


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Everything the harness needs to locate data, credentials and state.

    The repr deliberately omits credentials so settings can be logged freely.
    """

    workspace: Path
    feed: str = "iex"
    adjustment: str = "all"
    max_workers: int = 4
    poll_interval_s: float = 1.0
    heartbeat_interval_s: float = 5.0
    stale_after_s: float = 120.0
    recent_limit: int = 20
    alpaca_api_key: str | None = field(default=None, repr=False)
    alpaca_secret_key: str | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"Settings(workspace={str(self.workspace)!r}, feed={self.feed!r}, "
            f"adjustment={self.adjustment!r}, max_workers={self.max_workers}, "
            f"stale_after_s={self.stale_after_s}, credentials="
            f"{'set' if self.has_credentials else 'missing'})"
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    @property
    def secret_values(self) -> tuple[str, ...]:
        return tuple(v for v in (self.alpaca_api_key, self.alpaca_secret_key) if v)

    # --- workspace layout -------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        return self.workspace / "cache"

    @property
    def jobs_dir(self) -> Path:
        return self.workspace / "jobs"

    @property
    def inbox_dir(self) -> Path:
        return self.jobs_dir / "inbox"

    @property
    def running_dir(self) -> Path:
        return self.jobs_dir / "running"

    @property
    def done_dir(self) -> Path:
        return self.jobs_dir / "done"

    @property
    def failed_dir(self) -> Path:
        return self.jobs_dir / "failed"

    @property
    def state_dir(self) -> Path:
        return self.workspace / "state"

    @property
    def job_state_dir(self) -> Path:
        return self.state_dir / "jobs"

    @property
    def status_file(self) -> Path:
        return self.state_dir / "status.json"

    @property
    def results_dir(self) -> Path:
        return self.workspace / "results"

    def result_dir(self, job_id: str) -> Path:
        return self.results_dir / job_id

    def job_state_file(self, job_id: str) -> Path:
        return self.job_state_dir / f"{job_id}.json"

    def ensure_dirs(self) -> None:
        for path in (
            self.cache_dir,
            self.inbox_dir,
            self.running_dir,
            self.done_dir,
            self.failed_dir,
            self.job_state_dir,
            self.results_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_settings(workspace: Path | str | None = None) -> Settings:
    """Build settings from the environment. ``workspace`` overrides ``BT_WORKSPACE``."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    root = Path(workspace or os.environ.get("BT_WORKSPACE") or REPO_ROOT / "workspace")
    return Settings(
        workspace=root.resolve(),
        feed=(os.environ.get("ALPACA_FEED") or "iex").strip().lower(),
        adjustment=(os.environ.get("ALPACA_ADJUSTMENT") or "all").strip().lower(),
        max_workers=_env_int("BT_MAX_WORKERS", min(4, (os.cpu_count() or 2))),
        poll_interval_s=_env_float("BT_POLL_INTERVAL_S", 1.0),
        heartbeat_interval_s=_env_float("BT_HEARTBEAT_INTERVAL_S", 5.0),
        stale_after_s=_env_float("BT_STALE_AFTER_S", 120.0),
        recent_limit=_env_int("BT_RECENT_LIMIT", 20),
        alpaca_api_key=os.environ.get("ALPACA_API_KEY") or None,
        alpaca_secret_key=os.environ.get("ALPACA_SECRET_KEY") or None,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
