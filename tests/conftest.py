from __future__ import annotations

import pytest

from backtester.config import load_settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """A throwaway workspace with no credentials, so nothing can reach the network."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setenv("ALPACA_FEED", "iex")
    monkeypatch.setenv("ALPACA_ADJUSTMENT", "all")
    monkeypatch.setenv("BT_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("BT_MAX_WORKERS", "2")
    resolved = load_settings()
    resolved.ensure_dirs()
    return resolved
