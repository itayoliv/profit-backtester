"""Error types whose ``code`` is part of the agent contract."""

from __future__ import annotations


class BacktesterError(Exception):
    code = "internal_error"


class ValidationError(BacktesterError):
    code = "validation_error"

    def __init__(self, message: str, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.details = details or [message]


class NoDataError(BacktesterError):
    """A symbol returned no bars for the requested window."""

    code = "no_data"


class FeedNotPermittedError(BacktesterError):
    """The configured feed requires a subscription this account does not have."""

    code = "feed_not_permitted"


class MissingCredentialsError(BacktesterError):
    code = "missing_credentials"


class StaleWorkerError(BacktesterError):
    """A job was claimed by a worker that stopped heartbeating."""

    code = "stale_worker"
