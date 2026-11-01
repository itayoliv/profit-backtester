"""The worker: claims jobs, fans batches out, runs them in a process pool.

Backtests are CPU-bound, so processes rather than threads. Each child heartbeats so a
crashed worker leaves evidence the next worker can act on.
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

from ..config import PHASES, TERMINAL_PHASES, Settings, load_settings
from ..data.progress import ProgressEvent, ProgressReporter
from ..jsonio import iso, utcnow
from ..logging import get_logger, setup_logging
from ..pipeline.pipeline import Pipeline
from ..pipeline.request import BacktestRequest
from ..reporting.compare import compare
from ..reporting.writer import ResultWriter
from .queue import JobQueue
from .state import StateStore

log = get_logger("jobs.worker")

PHASE_ORDER = {name: i for i, name in enumerate(PHASES)}


# --------------------------------------------------------------------------
# Child process entry point
# --------------------------------------------------------------------------
def _heartbeat_loop(state: StateStore, job_id: str, stop: threading.Event, interval: float) -> None:
    while not stop.wait(interval):
        try:
            state.heartbeat(job_id)
        except OSError:  # a transient filesystem hiccup must not kill the run
            pass


def execute_job(payload: dict[str, Any], workspace: str) -> dict[str, Any]:
    """Run one single-ticker job to completion. Never raises: the outcome is returned."""
    settings = load_settings(workspace)
    setup_logging()
    state = StateStore(settings)
    job_id = str(payload.get("job_id"))

    stop = threading.Event()
    beat = threading.Thread(
        target=_heartbeat_loop,
        args=(state, job_id, stop, settings.heartbeat_interval_s),
        daemon=True,
    )
    beat.start()
    try:
        request = BacktestRequest(**payload)
        symbol = request.tickers[0]

        def on_progress(event: ProgressEvent) -> None:
            state.update(
                job_id,
                phase=event.phase,
                progress_pct=event.progress_pct,
                eta_seconds=event.eta_seconds,
                current_symbol=event.current_symbol,
            )

        reporter = ProgressReporter(on_progress, symbol=symbol)
        result = Pipeline(settings).run(request, symbol, reporter)

        state.update(job_id, phase="reporting", progress_pct=95.0, eta_seconds=None)
        ResultWriter(settings).write(
            job_id, result.summary, result.trades, result.equity, result.notes
        )
        state.update(
            job_id, phase="done", progress_pct=100.0, eta_seconds=None, error=None
        )
        return {"job_id": job_id, "ok": True}
    except BaseException as exc:  # noqa: BLE001 - the failure must reach the state file
        code = getattr(exc, "code", "internal_error")
        state.update(
            job_id,
            phase="failed",
            progress_pct=100.0,
            eta_seconds=None,
            error={"code": code, "message": str(exc), "traceback": traceback.format_exc()},
        )
        log.error("job %s failed (%s): %s", job_id, code, exc)
        return {"job_id": job_id, "ok": False, "error": code}
    finally:
        stop.set()


def run_inline(settings: Settings, request: BacktestRequest) -> dict[str, Any]:
    """Run a single-ticker request in this process. Used by the GUI's inline mode."""
    StateStore(settings).update(
        request.job_id,
        phase="queued",
        source=request.source,
        tickers=request.tickers,
        strategy=request.strategy,
        engine=request.engine,
        timeframe=request.timeframe,
        parent_job_id=request.parent_job_id,
    )
    return execute_job(request.model_dump(mode="json"), str(settings.workspace))


# --------------------------------------------------------------------------
# Worker loop
# --------------------------------------------------------------------------
class Worker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.queue = JobQueue(settings)
        self.state = self.queue.state
        self.writer = ResultWriter(settings)
        self._futures: dict[str, Future] = {}

    def run(self, once: bool = False, max_idle_polls: int | None = None) -> None:
        log.info(
            "worker started: workspace=%s max_workers=%d",
            self.settings.workspace,
            self.settings.max_workers,
        )
        idle = 0
        with ProcessPoolExecutor(max_workers=self.settings.max_workers) as pool:
            while True:
                self.reclaim_stale()
                claimed = self.dispatch(pool)
                self.harvest()
                self.resolve_parents()
                self.state.write_status(worker_alive=True)

                busy = claimed or self._futures
                idle = 0 if busy else idle + 1
                # A pass can claim nothing because the pool was full and then harvest it
                # empty, while children still wait in the inbox or a parent is still open.
                if once and not busy and not self.queue.has_work():
                    break
                if max_idle_polls is not None and idle >= max_idle_polls:
                    break
                time.sleep(self.settings.poll_interval_s)
        self.state.write_status(worker_alive=False)
        log.info("worker stopped")

    # --- steps ------------------------------------------------------------
    def dispatch(self, pool: ProcessPoolExecutor) -> int:
        claimed = 0
        while len(self._futures) < self.settings.max_workers:
            request = self.queue.claim_next()
            if request is None:
                break
            claimed += 1
            if request.is_batch and request.parent_job_id is None:
                self._open_parent(request)
                continue
            self.state.update(request.job_id, phase="queued", current_symbol=request.tickers[0])
            self._futures[request.job_id] = pool.submit(
                execute_job, request.model_dump(mode="json"), str(self.settings.workspace)
            )
        # Jobs waiting for a free slot inside the pool are not heartbeating yet.
        for job_id, future in self._futures.items():
            if not future.running() and not future.done():
                self.state.heartbeat(job_id)
        return claimed

    def harvest(self) -> None:
        for job_id in [j for j, f in self._futures.items() if f.done()]:
            future = self._futures.pop(job_id)
            try:
                outcome = future.result()
                failed = not outcome.get("ok", False)
            except BaseException as exc:  # noqa: BLE001 - pool-level crash
                failed = True
                self.state.update(
                    job_id,
                    phase="failed",
                    progress_pct=100.0,
                    error={
                        "code": "worker_crashed",
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            self.queue.finish(job_id, failed=failed)

    def reclaim_stale(self) -> None:
        for job_id in self.queue.running_ids():
            if job_id in self._futures:
                continue
            record = self.state.read(job_id)
            if record is None or record.get("is_parent") or not self.state.is_stale(record):
                continue
            attempt = int(record.get("attempt") or 1)
            if attempt < 2:
                log.warning("job %s went stale, requeueing (attempt %d)", job_id, attempt + 1)
                self.queue.requeue(job_id, attempt + 1)
            else:
                log.error("job %s went stale twice, failing", job_id)
                self.state.update(
                    job_id,
                    phase="failed",
                    progress_pct=100.0,
                    error={
                        "code": "stale_worker",
                        "message": (
                            "no heartbeat for over "
                            f"{self.settings.stale_after_s:.0f}s across two attempts"
                        ),
                        "traceback": None,
                    },
                )
                self.queue.finish(job_id, failed=True)

    # --- parents ----------------------------------------------------------
    def _open_parent(self, parent: BacktestRequest) -> None:
        child_ids = self.queue.fan_out(parent)
        self.state.update(
            parent.job_id,
            phase="queued",
            is_parent=True,
            child_ids=child_ids,
            source=parent.source,
            tickers=parent.tickers,
            strategy=parent.strategy,
            engine=parent.engine,
            timeframe=parent.timeframe,
        )
        log.info("job %s fanned out into %d children", parent.job_id, len(child_ids))

    def resolve_parents(self) -> None:
        for job_id in self.queue.running_ids():
            record = self.state.read(job_id)
            if record is None or not record.get("is_parent"):
                continue
            children = [
                self.state.read(child_id) or {} for child_id in record.get("child_ids") or []
            ]
            if not children:
                continue
            if any(child.get("phase") not in TERMINAL_PHASES for child in children):
                self._update_parent_progress(job_id, children)
                continue
            self._close_parent(job_id, children)

    def _update_parent_progress(self, job_id: str, children: list[dict[str, Any]]) -> None:
        pending = [c for c in children if c.get("phase") not in TERMINAL_PHASES]
        phase = min(
            (c.get("phase", "queued") for c in pending),
            key=lambda p: PHASE_ORDER.get(p, 0),
            default="queued",
        )
        progress = sum(float(c.get("progress_pct") or 0.0) for c in children) / len(children)
        self.state.update(
            job_id,
            phase=phase,
            progress_pct=round(progress, 1),
            current_symbol=pending[0].get("current_symbol") if pending else None,
        )

    def _close_parent(self, job_id: str, children: list[dict[str, Any]]) -> None:
        succeeded = [c for c in children if c.get("phase") == "done"]
        failed = [c for c in children if c.get("phase") != "done"]
        summaries = [s for s in (self.writer.summary(c["job_id"]) for c in succeeded) if s]

        request = self.queue.load_request(job_id)
        aggregate = _aggregate(job_id, request, summaries, failed)
        notes = [f"{len(succeeded)} of {len(children)} runs completed."]
        if request is not None and request.params_grid is not None:
            notes.append(
                "Parameter sweep: the headline metrics average runs with different "
                "parameters. Compare runs with the per-run table or `bt compare`."
            )
            notes += [
                f"skipped {s['params']}: {s['reason']}" for s in request.skipped_combinations()
            ]
        notes += [
            f"{(c.get('error') or {}).get('code', 'failed')}: {c.get('job_id')}" for c in failed
        ]
        notes += aggregate["comparison"]["warnings"]
        self.writer.write(job_id, aggregate, notes=notes)

        all_failed = not succeeded
        self.state.update(
            job_id,
            phase="failed" if all_failed else "done",
            progress_pct=100.0,
            eta_seconds=None,
            current_symbol=None,
            error=(
                {
                    "code": "all_children_failed",
                    "message": f"all {len(children)} ticker runs failed",
                    "traceback": None,
                }
                if all_failed
                else None
            ),
        )
        self.queue.finish(job_id, failed=all_failed)


def _aggregate(
    job_id: str,
    request: BacktestRequest | None,
    summaries: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> dict[str, Any]:
    def mean(key: str) -> float:
        values = [float(s.get(key) or 0.0) for s in summaries]
        return round(sum(values) / len(values), 6) if values else 0.0

    coverage = summaries[0].get("data_coverage", {}) if summaries else {}
    sweep = request is not None and request.params_grid is not None
    # A sweep's children each carry different params; the parent keeps the shared base.
    if request is not None and (sweep or not summaries):
        params = request.params
    else:
        params = summaries[0].get("params", {}) if summaries else {}
    return {
        "job_id": job_id,
        "parent_job_id": None,
        "tickers": request.tickers if request else [s["tickers"][0] for s in summaries],
        "strategy": request.strategy if request else (summaries[0].get("strategy") if summaries else None),
        "strategy_origin": summaries[0].get("strategy_origin") if summaries else None,
        "engine": request.engine if request else (summaries[0].get("engine") if summaries else None),
        "params": params,
        "params_grid": request.params_grid if request else None,
        "rules": request.rules.model_dump() if request else (summaries[0].get("rules") if summaries else {}),
        "filters": request.filters.model_dump() if request else {},
        "capital": request.capital.model_dump() if request else {},
        "data_coverage": {
            "feed": coverage.get("feed"),
            "adjustment": coverage.get("adjustment"),
            "timeframe": coverage.get("timeframe"),
            "requested_start": coverage.get("requested_start"),
            "requested_end": coverage.get("requested_end"),
            "bar_count": sum(int(s.get("data_coverage", {}).get("bar_count") or 0) for s in summaries),
            "cache_hit": all(s.get("data_coverage", {}).get("cache_hit") for s in summaries),
        },
        "total_return": mean("total_return"),
        "sharpe": mean("sharpe"),
        "max_drawdown": min((float(s.get("max_drawdown") or 0.0) for s in summaries), default=0.0),
        "win_rate": mean("win_rate"),
        "trade_count": sum(int(s.get("trade_count") or 0) for s in summaries),
        "rule_violations": sum(int(s.get("rule_violations") or 0) for s in summaries),
        "children": summaries,
        "comparison": compare(summaries),
        "failed_children": [
            {"job_id": c.get("job_id"), "error": c.get("error")} for c in failed
        ],
        "completed_at": iso(utcnow()),
    }
