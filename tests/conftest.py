"""Shared fixtures for the cross-module suites."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _privacy_ledger_in_tmp(tmp_path, monkeypatch):
    """No test may charge the laptop's real privacy ledger under services/edge/artifacts/."""
    try:
        from edge import upload
    except ImportError:  # a suite run without the edge package
        return
    monkeypatch.setattr(upload, "DEFAULT_LEDGER_ROOT", tmp_path / "privacy_ledger")
