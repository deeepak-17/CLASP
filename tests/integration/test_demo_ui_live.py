"""The live panel (demo_ui) against the real cluster and registry, over real HTTP.

The cluster and the registry run as uvicorn servers on free local ports, so
every hop is a socket call: demo_ui -> cluster, demo_ui -> registry, and the
cluster's own calls to the registry (publish, restore). Edges upload with
``edge.upload`` and pull with ``edge.upload.pull_cluster_adapter``, exactly as
the edge laptops do. Adapters are tiny (2 layers, rank 4, 8-dim); nothing
here needs a GPU.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

uvicorn = pytest.importorskip("uvicorn")
cluster_server = pytest.importorskip("cluster.server")
pytest.importorskip("registry.app")

from demo_ui import server as panel  # noqa: E402
from edge import upload, wire  # noqa: E402

CLIENTS = ("client-flask", "client-requests")


def _serve(app):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert srv.started
    return srv, thread, f"http://127.0.0.1:{port}"


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "registry"))
    import registry.app as registry_app

    registry_app._store = None
    cluster_server._reset_clusters()
    servers = [_serve(registry_app.app), _serve(cluster_server.app)]
    registry_url, cluster_url = servers[0][2], servers[1][2]
    monkeypatch.setattr(panel, "CLUSTER_BASE", cluster_url)
    monkeypatch.setattr(panel, "REGISTRY_BASE", registry_url)
    monkeypatch.setattr(panel, "LIVE_CLUSTERS", {"web": list(CLIENTS)})
    panel._live.clear()
    with TestClient(panel.app) as ui:
        yield ui, cluster_url, registry_url
    for srv, thread, _ in servers:
        srv.should_exit = True
        thread.join(timeout=10)
    cluster_server._reset_clusters()
    registry_app._store = None


def _adapter(root: Path, client: str, seed: int, d3: dict | None = None) -> Path:
    rng = np.random.default_rng(seed)
    sd = {}
    for layer in range(2):
        for m in wire.TARGET_MODULES:
            sd[wire.peft_key(layer, m, "lora_A")] = rng.normal(0, 0.1, (4, 8)).astype(np.float32)
            sd[wire.peft_key(layer, m, "lora_B")] = rng.normal(0, 0.1, (8, 4)).astype(np.float32)
    adapter = wire.save_peft_dir(root / client / "adapter", sd,
                                 {"r": 4, "lora_alpha": 4, "target_modules": list(wire.TARGET_MODULES)})
    (root / client / "manifest.json").write_text(json.dumps({
        "client_id": f"web/{client}", "seed": 0, "privacy": {"epsilon": 2.0, "delta": 1e-5},
        "data": {"train": {"n_chunks": 10}}, "d3": d3}), encoding="utf-8")
    return adapter


def _round(ui, cluster_url, root: Path, seed: int, pulled: dict | None = None) -> dict:
    """Both edges upload (as residuals on ``pulled`` if given), then aggregate + publish."""
    for i, client in enumerate(CLIENTS):
        d3 = {"alpha": 0.5, "cluster_adapter": pulled["path"], "cluster_pull": pulled} if pulled else None
        upload.upload_adapter(_adapter(root, client, seed + i, d3), cluster_url=cluster_url,
                              log=lambda *_: None)
    assert ui.post("/api/live/aggregate/web").status_code == 200
    published = ui.post("/api/live/publish/web")
    assert published.status_code == 200, published.text
    return published.json()["version"]


def _evidence(candidate_es: float, baseline_es: float):
    def fake(cluster_id, **_kw):
        measured = {"edit_similarity": candidate_es, "exact_match": 0.2, "perplexity": 3.0,
                    "n_examples": 60}
        base = {"edit_similarity": baseline_es, "exact_match": 0.2, "perplexity": 3.0,
                "n_examples": 60}
        guard = [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.5}}]
        return {"in_project": measured, "baseline_in_project": base, "guard": guard,
                "baseline_guard": guard, "noise_band": 0.0,
                "provenance": {"stale": False, "note": "test"}}
    return fake


def _two_published_rounds(ui, cluster_url, tmp_path):
    assert ui.post("/api/live/register").status_code == 200
    v1 = _round(ui, cluster_url, tmp_path / "r0", seed=1)
    pulled = upload.pull_cluster_adapter(tmp_path / "from_cluster", cluster_id="web",
                                         cluster_url=cluster_url, log=lambda *_: None)
    assert pulled["sha256"] == v1["sha256"]
    v2 = _round(ui, cluster_url, tmp_path / "r1", seed=11, pulled=pulled)
    assert v2["version"] == 2 and v2["sha256"] != v1["sha256"]
    return v1, v2


def test_a_rollback_on_promote_puts_the_previous_merge_back_on_the_cluster(stack, tmp_path,
                                                                           monkeypatch):
    import requests

    ui, cluster_url, _ = stack
    v1, v2 = _two_published_rounds(ui, cluster_url, tmp_path)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.40, 0.45))
    r = ui.post("/api/live/promote/web")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"]["action"] == "rollback"
    assert body["cluster_restore"]["restored"] == {
        "adapter_name": "cluster-web", "version": 1, "round_id": 0, "sha256": v1["sha256"]}
    listing = requests.get(f"{cluster_url}/clusters", timeout=10).json()["clusters"]["web"]
    assert listing["active_sha256"] == v1["sha256"]
    # an edge now pulls what the registry kept, not the rejected round
    again = upload.pull_cluster_adapter(tmp_path / "from_cluster2", cluster_id="web",
                                        cluster_url=cluster_url, log=lambda *_: None)
    assert again["sha256"] == v1["sha256"]


def test_a_promote_leaves_the_cluster_serving_the_new_merge(stack, tmp_path, monkeypatch):
    import requests

    ui, cluster_url, _ = stack
    _, v2 = _two_published_rounds(ui, cluster_url, tmp_path)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))
    body = ui.post("/api/live/promote/web").json()
    assert body["decision"]["action"] == "promote" and "cluster_restore" not in body
    listing = requests.get(f"{cluster_url}/clusters", timeout=10).json()["clusters"]["web"]
    assert listing["active_sha256"] == v2["sha256"]
