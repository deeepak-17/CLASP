"""``edge.upload``: one edge uploads its adapter to a cluster on another machine.

Runs against P2's real cluster service in a uvicorn thread (no stand-in), with
a small PEFT-shaped adapter written the way ``edge.train_client`` lays it out
(``<client>/adapter/`` beside ``<client>/manifest.json``). Skips only if the
cluster package is not installed.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from edge import upload, wire

server = pytest.importorskip("cluster.server")
uvicorn = pytest.importorskip("uvicorn")


def _adapter(root: Path, client: str, *, seed: int, cluster: str = "web",
             epsilon=None, n_chunks: int = 12) -> Path:
    rng = np.random.default_rng(seed)
    r, d = 4, 8
    sd = {}
    for layer in range(2):
        for m in wire.TARGET_MODULES:
            sd[wire.peft_key(layer, m, "lora_A")] = rng.normal(size=(r, d)).astype(np.float32)
            sd[wire.peft_key(layer, m, "lora_B")] = rng.normal(size=(d, r)).astype(np.float32)
    adapter = wire.save_peft_dir(root / client / "adapter", sd,
                                 {"r": r, "lora_alpha": r, "target_modules": list(wire.TARGET_MODULES)})
    privacy = {"epsilon": epsilon, "delta": 1e-5} if epsilon is not None else None
    (root / client / "manifest.json").write_text(json.dumps({
        "client_id": f"{cluster}/{client}", "seed": 0, "privacy": privacy,
        "data": {"train": {"n_chunks": n_chunks}}}), encoding="utf-8")
    return adapter


@pytest.fixture(scope="module")
def cluster_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert srv.started
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(autouse=True)
def _clean():
    server._reset_clusters()
    yield
    server._reset_clusters()


def _register(url, cluster, members):
    import requests

    assert requests.put(f"{url}/clusters/{cluster}/members",
                        json={"client_ids": members}, timeout=10).status_code == 200


def test_two_edges_upload_into_their_registered_cluster(cluster_url, tmp_path):
    import requests

    _register(cluster_url, "web", ["client-flask", "client-requests"])
    a = _adapter(tmp_path, "client-flask", seed=1, epsilon=7.99)
    b = _adapter(tmp_path, "client-requests", seed=2, epsilon=6.5, n_chunks=30)
    out = upload.upload_adapter(a, cluster_url=cluster_url, log=lambda *_: None)
    assert out["cluster_id"] == "web" and out["round_id"] == 0 and out["num_examples"] == 12
    assert out["privacy"]["epsilon"] == 7.99
    upload.upload_adapter(b, cluster_url=cluster_url, log=lambda *_: None)
    entry = requests.get(f"{cluster_url}/clusters", timeout=10).json()["clusters"]["web"]
    assert entry["uploaded_clients"] == ["client-flask", "client-requests"]
    agg = requests.post(f"{cluster_url}/clusters/web/aggregate", timeout=60).json()
    assert agg["source_clients"] == ["client-flask", "client-requests"]
    assert agg["epsilon"] == 7.99  # D7: the max over the aggregated clients


def test_unregistered_cluster_is_a_404_with_what_to_do(cluster_url, tmp_path):
    a = _adapter(tmp_path, "client-numpy", seed=3, cluster="scientific")
    with pytest.raises(upload.UploadError) as err:
        upload.upload_adapter(a, cluster_url=cluster_url, log=lambda *_: None)
    assert err.value.status == 404 and "Register clusters" in err.value.hint


def test_unassigned_client_falls_back_to_the_manifest_cluster(cluster_url, tmp_path):
    _register(cluster_url, "scientific", ["client-pandas"])
    a = _adapter(tmp_path, "client-numpy", seed=3, cluster="scientific")
    out = upload.upload_adapter(a, cluster_url=cluster_url, log=lambda *_: None)
    assert out["cluster_id"] == "scientific"


def test_stale_round_is_a_409(cluster_url, tmp_path):
    _register(cluster_url, "web", ["client-flask"])
    a = _adapter(tmp_path, "client-flask", seed=1)
    with pytest.raises(upload.UploadError) as err:
        upload.upload_adapter(a, cluster_url=cluster_url, round_id=5, log=lambda *_: None)
    assert err.value.status == 409 and "run this again" in err.value.hint


def test_assigned_elsewhere_refuses_before_sending(cluster_url, tmp_path):
    _register(cluster_url, "web", ["client-flask"])
    _register(cluster_url, "scientific", [])
    a = _adapter(tmp_path, "client-flask", seed=1)
    with pytest.raises(upload.UploadError, match="assigned to cluster 'web'"):
        upload.upload_adapter(a, cluster_url=cluster_url, cluster_id="scientific",
                              log=lambda *_: None)


def test_unreachable_cluster_and_missing_files(tmp_path):
    a = _adapter(tmp_path, "client-flask", seed=1)
    with pytest.raises(upload.UploadError, match="not reachable") as err:
        upload.upload_adapter(a, cluster_url="http://127.0.0.1:9", log=lambda *_: None)
    assert "firewall" in err.value.hint
    with pytest.raises(upload.UploadError, match="no adapter_model.safetensors"):
        upload.upload_adapter(tmp_path, log=lambda *_: None)
    with pytest.raises(upload.UploadError, match="missing certificate files"):
        upload.session_for(tmp_path)


def test_cli(cluster_url, tmp_path, capsys):
    _register(cluster_url, "web", ["client-flask"])
    a = _adapter(tmp_path, "client-flask", seed=1)
    assert upload.main(["--adapter", str(a), "--cluster-url", cluster_url, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cluster_id"] == "web"
    assert upload.main(["--adapter", str(a), "--cluster-url", cluster_url, "--round", "9"]) == 1
    assert "409" in capsys.readouterr().err
