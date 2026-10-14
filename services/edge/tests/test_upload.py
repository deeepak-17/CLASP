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


# --------------------------------------------------------------------------- #
# FedProx under D3: a residual names the merged adapter it was trained on
# --------------------------------------------------------------------------- #
def _merge_round(url, tmp_path, cluster="web", clients=("client-flask", "client-requests")):
    import requests

    _register(url, cluster, list(clients))
    for i, c in enumerate(clients):
        upload.upload_adapter(_adapter(tmp_path / "r0", c, seed=10 + i, cluster=cluster),
                              cluster_url=url, log=lambda *_: None)
    r = requests.post(f"{url}/clusters/{cluster}/aggregate", json={}, timeout=60)
    assert r.status_code == 200, r.text
    return r.json()


def _residual(root, client, pulled: dict | None, *, seed, alpha=0.5, cluster="web"):
    """A client adapter whose manifest says it was trained on top of ``pulled``."""
    adapter = _adapter(root, client, seed=seed, cluster=cluster)
    m = adapter.parent / "manifest.json"
    d3 = {"alpha": alpha, "cluster_adapter": str(root / "somewhere")}
    if pulled is not None:
        d3["cluster_pull"] = pulled
    m.write_text(json.dumps({**json.loads(m.read_text()), "d3": d3}), encoding="utf-8")
    return adapter


def test_pull_records_the_merged_adapters_digest(cluster_url, tmp_path):
    import requests

    merged = _merge_round(cluster_url, tmp_path)
    info = upload.pull_cluster_adapter(tmp_path / "from_cluster", cluster_id="web",
                                       cluster_url=cluster_url, log=lambda *_: None)
    listing = requests.get(f"{cluster_url}/clusters", timeout=10).json()["clusters"]["web"]
    assert info["sha256"] == merged["sha256"] == listing["active_sha256"]
    assert info["update_rule"] == "average"
    on_disk = json.loads((Path(info["path"]) / "pull_manifest.json").read_text())
    assert on_disk["sha256"] == merged["sha256"]
    assert upload.pull_provenance(Path(info["path"]))["sha256"] == merged["sha256"]
    assert upload.pull_provenance(tmp_path) is None


def test_residual_upload_declares_its_base_and_is_accumulated(cluster_url, tmp_path):
    import requests

    _merge_round(cluster_url, tmp_path)
    pulled = upload.pull_cluster_adapter(tmp_path / "from_cluster", cluster_id="web",
                                         cluster_url=cluster_url, log=lambda *_: None)
    out = upload.upload_adapter(_residual(tmp_path / "r1", "client-flask", pulled, seed=20),
                                cluster_url=cluster_url, log=lambda *_: None)
    assert out["base"] == {"cluster_id": "web", "round_id": 0, "sha256": pulled["sha256"],
                           "alpha": 0.5}
    r = requests.post(f"{cluster_url}/clusters/web/aggregate", json={}, timeout=60)
    assert r.status_code == 200, r.text
    manifest = requests.get(f"{cluster_url}/clusters/web/aggregate/manifest", timeout=10).json()
    assert manifest["update_rule"] == "fedprox_cumulative"
    assert manifest["base_sha256"] == pulled["sha256"]


def test_residual_on_an_older_merge_is_refused_with_what_to_do(cluster_url, tmp_path):
    _merge_round(cluster_url, tmp_path)
    pulled = upload.pull_cluster_adapter(tmp_path / "from_cluster", cluster_id="web",
                                         cluster_url=cluster_url, log=lambda *_: None)
    old = dict(pulled, sha256="f" * 64)
    with pytest.raises(upload.UploadError) as err:
        upload.upload_adapter(_residual(tmp_path / "r1", "client-flask", old, seed=21),
                              cluster_url=cluster_url, log=lambda *_: None)
    assert err.value.status == 409 and "pull the merged adapter" in err.value.hint


def test_adapter_on_a_cluster_adapter_from_elsewhere_is_not_sent(cluster_url, tmp_path):
    """Trained on top of a cluster adapter this cluster never served (e.g. one
    materialized from the registry): as a residual it has no base here, and as
    a full adapter it would be wrong. Refused before anything is sent."""
    _register(cluster_url, "web", ["client-flask"])
    a = _residual(tmp_path / "r1", "client-flask", None, seed=22)
    with pytest.raises(upload.UploadError, match="did not come from this cluster") as err:
        upload.upload_adapter(a, cluster_url=cluster_url, log=lambda *_: None)
    assert err.value.status is None and "--allow-foreign-base" in err.value.hint
    out = upload.upload_adapter(a, cluster_url=cluster_url, allow_foreign_base=True,
                                log=lambda *_: None)
    assert out["base"] is None
