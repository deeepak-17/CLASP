"""Shared fixtures for registry tests."""
from __future__ import annotations

import json
import socket
import struct
import threading
import time

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


def forget_database(root) -> None:
    """Delete ``registry.db`` under ``root`` (its store must be closed first).

    The volume then looks like one the file-based registry wrote, and the next
    store opened on it imports it again.
    """
    from pathlib import Path

    for suffix in ("", "-wal", "-shm"):
        Path(f"{Path(root) / 'registry.db'}{suffix}").unlink(missing_ok=True)


@pytest.fixture
def safetensors_blob() -> bytes:
    return make_safetensors()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "registry"))
    monkeypatch.delenv("CLASP_REGISTRY_DB", raising=False)  # registry.db under the root
    from registry.storage import RegistryStore

    s = RegistryStore()
    yield s
    s.close()  # Windows: release registry.db before tmp_path is removed


def _drop_app_store(appmod) -> None:
    if appmod._store is not None:
        appmod._store.close()
    appmod._store = None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "registry_api"))
    monkeypatch.delenv("CLASP_REGISTRY_DB", raising=False)
    from fastapi.testclient import TestClient

    import registry.app as appmod

    _drop_app_store(appmod)  # force re-read of CLASP_REGISTRY_DATA
    yield TestClient(appmod.app)
    _drop_app_store(appmod)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running(tmp_path, monkeypatch):
    """Start the registry via build_config in a thread; yield its port."""
    import uvicorn
    from registry.serve import build_config

    servers = []

    def start(env: dict[str, str]):
        monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "data"))
        import registry.app as appmod

        appmod._store = None
        port = free_port()
        config = build_config({**env, "CLASP_REGISTRY_HOST": "127.0.0.1",
                               "CLASP_REGISTRY_PORT": str(port)})
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        assert server.started, "registry did not start"
        servers.append((server, thread))
        return port

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=5)
