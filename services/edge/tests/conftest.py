"""Shared fixtures for the edge tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _privacy_ledger_in_tmp(tmp_path, monkeypatch):
    """No test may charge the laptop's real privacy ledger under artifacts/."""
    from edge import upload

    monkeypatch.setattr(upload, "DEFAULT_LEDGER_ROOT", tmp_path / "privacy_ledger")
