"""Logging setup. All human-readable output goes to stderr with secrets redacted."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


class RedactingFilter(logging.Filter):
    """Replace known credential values anywhere in a formatted record."""

    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        super().__init__()
        # Short values are too likely to collide with ordinary text to redact safely.
        self._secrets = tuple(s for s in secrets if s and len(s) >= 8)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        message = record.getMessage()
        redacted = message
        for secret in self._secrets:
            redacted = redacted.replace(secret, "***REDACTED***")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def _collect_secrets() -> tuple[str, ...]:
    return tuple(
        v
        for v in (
            os.environ.get("ALPACA_API_KEY"),
            os.environ.get("ALPACA_SECRET_KEY"),
        )
        if v
    )


def setup_logging(level: int | str = logging.INFO) -> None:
    global _CONFIGURED
    root = logging.getLogger("backtester")
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        handler.addFilter(RedactingFilter(_collect_secrets()))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"backtester.{name}")
