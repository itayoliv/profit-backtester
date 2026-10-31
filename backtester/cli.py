"""The ``bt`` command line. This is the only supported surface for the agent.

Exit codes are part of the contract:
  0  success
  1  unexpected internal error
  2  request validation error
  3  results requested while the job is still queued or running
  4  job failed or is unknown
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

from .config import TERMINAL_PHASES, load_settings
from .errors import ValidationError
from .jsonio import jsonable
from .logging import get_logger, setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Agent-driven backtesting. Submit a job, poll its status, read its results.",
)
log = get_logger("cli")

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_VALIDATION = 2
EXIT_PENDING = 3
EXIT_FAILED = 4


def emit(payload: Any) -> None:
    """One JSON object on stdout. Everything human-readable goes to stderr."""
    json.dump(jsonable(payload), sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _settings(workspace: str | None = None):
    settings = load_settings(workspace)
    settings.ensure_dirs()
    return settings


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging on stderr."),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


@app.command()
def submit(
    request_file: str = typer.Argument(..., help="Path to a request JSON file, or - for stdin."),
    workspace: str = typer.Option(None, "--workspace", help="Override the workspace directory."),
) -> None:
    """Validate and queue a backtest. Prints {"job_id": ...}."""
    from .jobs.queue import JobQueue
    from .pipeline.request import parse_request

    raw = sys.stdin.read() if request_file == "-" else None
    if raw is None:
        path = Path(request_file)
        if not path.exists():
            emit({"error": "validation_error", "details": [f"no such file: {request_file}"]})
            raise typer.Exit(EXIT_VALIDATION)
        raw = path.read_text(encoding="utf-8")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        emit({"error": "validation_error", "details": [f"invalid JSON: {exc}"]})
        raise typer.Exit(EXIT_VALIDATION) from None

    try:
        request = parse_request(payload)
        job_id = JobQueue(_settings(workspace)).submit(request)
    except ValidationError as exc:
        emit({"error": "validation_error", "details": exc.details})
        raise typer.Exit(EXIT_VALIDATION) from None

    emit({"job_id": job_id, "tickers": request.tickers, "children": len(request.tickers)})


@app.command()
def status(
    job_id: str = typer.Option(None, "--job-id", help="Inspect a single job."),
    workspace: str = typer.Option(None, "--workspace"),
) -> None:
    """Print status.json, or one job's state when --job-id is given."""
    from .jobs.state import StateStore

    store = StateStore(_settings(workspace))
    if job_id is None:
        emit(store.read_status())
        return

    record = store.read(job_id)
    if record is None:
        emit({"error": "unknown_job", "job_id": job_id, "phase": None})
        raise typer.Exit(EXIT_OK)
    children = store.children_of(job_id)
    if children:
        record = {**record, "children": children}
    emit(record)


@app.command()
def results(
    job_id: str = typer.Argument(...),
    workspace: str = typer.Option(None, "--workspace"),
) -> None:
    """Print a finished job's summary. Exit 3 while running, 4 if failed or unknown."""
    from .jobs.state import StateStore
    from .reporting.writer import ResultWriter

    settings = _settings(workspace)
    record = StateStore(settings).read(job_id)
    writer = ResultWriter(settings)

    if record is None:
        emit({"error": "unknown_job", "job_id": job_id, "phase": None})
        raise typer.Exit(EXIT_FAILED)

    phase = record.get("phase")
    if phase not in TERMINAL_PHASES:
        emit(
            {
                "job_id": job_id,
                "phase": phase,
                "progress_pct": record.get("progress_pct"),
                "eta_seconds": record.get("eta_seconds"),
            }
        )
        raise typer.Exit(EXIT_PENDING)

    if phase == "failed":
        emit({"error": record.get("error"), "job_id": job_id, "phase": phase})
        raise typer.Exit(EXIT_FAILED)

    summary = writer.summary(job_id)
    if summary is None:
        emit({"error": "missing_results", "job_id": job_id, "phase": phase})
        raise typer.Exit(EXIT_FAILED)

    emit(
        {
            "job_id": job_id,
            "phase": phase,
            "summary": summary,
            "report_path": str(writer.report_path(job_id).resolve()),
            "trades_csv": str((settings.result_dir(job_id) / "trades.csv").resolve()),
            "equity_parquet": str((settings.result_dir(job_id) / "equity.parquet").resolve()),
        }
    )


@app.command()
def worker(
    once: bool = typer.Option(False, "--once", help="Drain the queue and exit."),
    workspace: str = typer.Option(None, "--workspace"),
) -> None:
    """Run the worker pool. Jobs make no progress unless a worker is running."""
    from .jobs.worker import Worker

    settings = _settings(workspace)
    typer.echo(f"worker: {settings!r}", err=True)
    Worker(settings).run(once=once)


@app.command()
def fetch(
    tickers: str = typer.Argument(..., help="Comma-separated symbols."),
    timeframe: str = typer.Option("1Day", "--timeframe"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    workspace: str = typer.Option(None, "--workspace"),
) -> None:
    """Warm the cache without running a backtest."""
    from .data.manager import DataManager

    settings = _settings(workspace)
    manager = DataManager(settings)
    coverage = []
    for symbol in [t.strip().upper() for t in tickers.split(",") if t.strip()]:
        coverage.append(manager.get_bars(symbol, timeframe, start, end).coverage)
    emit({"fetched": coverage})


@app.command()
def strategies(workspace: str = typer.Option(None, "--workspace")) -> None:
    """List registered strategies and their parameter schemas."""
    from .strategies.registry import list_strategies

    emit(
        {
            "strategies": [
                {"name": name, "params_schema": strategy.params_schema()}
                for name, strategy in list_strategies().items()
            ]
        }
    )


def main() -> None:
    try:
        app()
    except SystemExit:
        raise
    except ValidationError as exc:
        emit({"error": "validation_error", "details": exc.details})
        sys.exit(EXIT_VALIDATION)
    except BaseException as exc:  # noqa: BLE001 - keep exit 1 reserved for surprises
        log.exception("unexpected failure")
        emit({"error": "internal_error", "message": str(exc)})
        sys.exit(EXIT_INTERNAL)


if __name__ == "__main__":
    main()
