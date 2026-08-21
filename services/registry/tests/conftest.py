"""Shared fixtures for registry tests."""
from __future__ import annotations

import json
import struct

import pytest


def make_safetensors(tensors: dict[str, list[float]] | None = None) -> bytes:
    """Build a minimal-but-valid safetensors blob without numpy/torch.

    Format: <u64 header_len LE><json header><data>. Each tensor is 1-D F32.
    """
    tensors = tensors or {"lora_A": [0.1, 0.2, 0.3, 0.4]}
    header: dict = {}
    body = b""
    offset = 0
    for name, values in tensors.items():
        data = struct.pack(f"<{len(values)}f", *values)
        header[name] = {
            "dtype": "F32",
            "shape": [len(values)],
            "data_offsets": [offset, offset + len(data)],
        }
        body += data
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes + body


@pytest.fixture
def safetensors_blob() -> bytes:
    return make_safetensors()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "registry"))
    from registry.storage import RegistryStore

    return RegistryStore()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "registry_api"))
    from fastapi.testclient import TestClient

    import registry.app as appmod

    appmod._store = None  # force re-read of CLASP_REGISTRY_DATA
    return TestClient(appmod.app)
