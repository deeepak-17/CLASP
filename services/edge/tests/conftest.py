"""Shared fixtures for the edge tests."""
from __future__ import annotations

import os

# The suites fire hundreds of requests a second from one address. The services'
# rate limit is switched off for them (before any service module is imported);
# the middleware tests turn the real limits on for themselves.
os.environ.setdefault("CLASP_RATE_LIMIT_PER_MIN", "0")

import pytest


@pytest.fixture(autouse=True)
def _privacy_ledger_in_tmp(tmp_path, monkeypatch):
    """No test may charge the laptop's real privacy ledger under artifacts/."""
    from edge import upload

    monkeypatch.setattr(upload, "DEFAULT_LEDGER_ROOT", tmp_path / "privacy_ledger")
