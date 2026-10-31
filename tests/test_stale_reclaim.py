"""A job whose worker vanished is requeued once, then failed with stale_worker."""

from __future__ import annotations

from datetime import timedelta

from backtester.jobs.queue import JobQueue
from backtester.jobs.worker import Worker
from backtester.jsonio import iso, utcnow
from backtester.pipeline.request import parse_request
from helpers import base_request


def claim_and_go_stale(settings, queue: JobQueue, job_id: str) -> None:
    """Simulate a worker that claimed the job and then died."""
    claimed = queue.claim_next()
    assert claimed is not None and claimed.job_id == job_id
    record = queue.state.read(job_id) or {}
    record["heartbeat_at"] = iso(utcnow() - timedelta(seconds=settings.stale_after_s + 60))
    record["phase"] = "fetching_data"
    from backtester.jsonio import write_json

    write_json(settings.job_state_file(job_id), record)


def test_stale_job_is_requeued_once_then_failed(settings):
    queue = JobQueue(settings)
    queue.submit(parse_request(base_request(job_id="ghost")))
    worker = Worker(settings)

    claim_and_go_stale(settings, queue, "ghost")
    worker.reclaim_stale()

    record = queue.state.read("ghost")
    assert record["phase"] == "queued"
    assert record["attempt"] == 2
    assert (settings.inbox_dir / "ghost.json").exists()
    assert not (settings.running_dir / "ghost.json").exists()

    claim_and_go_stale(settings, queue, "ghost")
    worker.reclaim_stale()

    record = queue.state.read("ghost")
    assert record["phase"] == "failed"
    assert record["error"]["code"] == "stale_worker"
    assert (settings.failed_dir / "ghost.json").exists()


def test_a_heartbeating_job_is_left_alone(settings):
    queue = JobQueue(settings)
    queue.submit(parse_request(base_request(job_id="alive")))
    queue.claim_next()
    queue.state.update("alive", phase="running_strategy")

    Worker(settings).reclaim_stale()

    record = queue.state.read("alive")
    assert record["phase"] == "running_strategy"
    assert record["attempt"] == 1


def test_status_file_buckets_jobs_by_phase(settings):
    queue = JobQueue(settings)
    queue.submit(parse_request(base_request(job_id="waiting")))
    queue.submit(parse_request(base_request(job_id="busy")))
    queue.submit(parse_request(base_request(job_id="finished")))
    queue.state.update("busy", phase="fetching_data")
    queue.state.update("finished", phase="done")

    status = queue.state.write_status(worker_alive=True)

    assert [j["job_id"] for j in status["queued"]] == ["waiting"]
    assert [j["job_id"] for j in status["running"]] == ["busy"]
    assert [j["job_id"] for j in status["recent_done"]] == ["finished"]
    assert status["recent_failed"] == []
    assert status["worker_heartbeat_at"] is not None


def test_recent_lists_are_capped(settings):
    queue = JobQueue(settings)
    for index in range(settings.recent_limit + 5):
        queue.submit(parse_request(base_request(job_id=f"job{index:03d}")))
        queue.state.update(f"job{index:03d}", phase="done")

    status = queue.state.write_status()
    assert len(status["recent_done"]) == settings.recent_limit
