from __future__ import annotations

import pytest

from backtester.config import load_settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """A throwaway workspace with no credentials, so nothing can reach the network."""
    # Empty, not deleted: load_settings() reads .env without overriding, so a deleted
    # key would come straight back from a developer's real .env.
    monkeypatch.setenv("ALPACA_API_KEY", "")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "")
    monkeypatch.setenv("ALPACA_FEED", "iex")
    monkeypatch.setenv("ALPACA_ADJUSTMENT", "all")
    monkeypatch.setenv("BT_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("BT_MAX_WORKERS", "2")
    monkeypatch.setenv("BT_STRATEGIES_DIR", str(tmp_path / "custom_strategies"))
    resolved = load_settings()
    resolved.ensure_dirs()
    return resolved
