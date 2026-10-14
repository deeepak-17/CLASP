"""``edge.webui``: the edge laptop's page, driven over HTTP against the real cluster."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from edge import webui
from tests.test_upload import _adapter, _register, cluster_url  # noqa: F401 - fixture

server = pytest.importorskip("cluster.server")


@pytest.fixture(autouse=True)
def _clean():
    server._reset_clusters()
    yield
    server._reset_clusters()


@pytest.fixture
def page_url(cluster_url, tmp_path):  # noqa: F811 - the imported fixture
    _adapter(tmp_path / "round2_d3", "client-flask", seed=1, epsilon=7.99)
    _adapter(tmp_path / "round1", "client-flask", seed=2)
    _adapter(tmp_path / "round1", "client-numpy", seed=3, cluster="scientific")
    page = webui.EdgePage(cluster_url=cluster_url, client_id="client-flask", adapters_root=tmp_path)
    httpd = webui.serve(page, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", page
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _wait_job(url):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = _get(f"{url}/api/job")
        if not job["running"]:
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_page_lists_only_this_clients_dp_adapters(page_url):
    url, _ = page_url
    with urllib.request.urlopen(url, timeout=10) as r:
        assert b"CLASP Edge" in r.read()
    adapters = _get(f"{url}/api/adapters")["adapters"]
    # round1/client-flask has no ε, so it is not offered (D7).
    assert [a["id"] for a in adapters] == ["round2_d3/client-flask"]
    d3 = adapters[0]
    assert d3["epsilon"] == 7.99 and d3["cluster"] == "web" and "path" not in d3
    assert d3["trained_here"] is False and d3["trained_utc"]


def test_non_dp_adapters_are_offered_only_when_allowed(tmp_path):
    _adapter(tmp_path / "round1", "client-flask", seed=2)
    page = webui.EdgePage(cluster_url="http://x", client_id="client-flask", adapters_root=tmp_path)
    assert page.adapters() == []
    with pytest.raises(ValueError, match="without differential privacy"):
        page.start_upload("round1/client-flask")
    loose = webui.EdgePage(cluster_url="http://x", client_id="client-flask",
                           adapters_root=tmp_path, allow_non_dp=True)
    assert [a["id"] for a in loose.adapters()] == ["round1/client-flask"]


def test_each_live_run_is_its_own_set_newest_first(tmp_path):
    import json as _json
    for stamp, utc in (("live-20261014-100000", "2026-10-14T10:00:00+00:00"),
                       ("live-20261014-110000", "2026-10-14T11:00:00+00:00")):
        _adapter(tmp_path / stamp, "client-flask", seed=1, epsilon=8.0)
        m = tmp_path / stamp / "client-flask" / "manifest.json"
        m.write_text(_json.dumps({**_json.loads(m.read_text()), "utc": utc,
                                  "results": {"optimizer_steps": 1}}))
    found = webui.find_adapters(tmp_path)
    assert [a["id"] for a in found] == ["live-20261014-110000/client-flask",
                                        "live-20261014-100000/client-flask"]
    assert all(a["trained_here"] and a["steps"] == 1 for a in found)


def test_status_then_upload_shows_the_client_arrive(page_url, cluster_url):  # noqa: F811
    url, _ = page_url
    status = _get(f"{url}/api/status")
    assert status["reachable"] and "web" not in status["clusters"] and status["my_cluster"] is None
    _register(cluster_url, "web", ["client-flask", "client-requests"])
    assert _get(f"{url}/api/status")["my_cluster"] == "web"
    code, body = _post(f"{url}/api/upload", {"adapter": "round2_d3/client-flask"})
    assert code == 202, body
    job = _wait_job(url)
    assert job["error"] is None and job["result"]["cluster_id"] == "web"
    assert job["result"]["epsilon"] == 7.99
    status = _get(f"{url}/api/status")
    assert status["clusters"]["web"]["uploaded_clients"] == ["client-flask"]
    assert status["last_sent"]["adapter"] == "round2_d3/client-flask"
    assert status["last_sent"]["round_id"] == status["clusters"]["web"]["round_id"]


def test_refusals_reach_the_page(page_url):
    url, _ = page_url
    code, _ = _post(f"{url}/api/upload", {"adapter": "../../etc/passwd"})
    assert code == 404
    code, body = _post(f"{url}/api/upload", {"adapter": "round1/client-flask"})
    assert code == 422 and "differential privacy" in body["detail"]
    code, _ = _post(f"{url}/api/upload", {"adapter": "round2_d3/client-flask"})
    assert code == 202
    job = _wait_job(url)
    assert "not registered" in job["error"]  # nobody registered 'web' on the cluster
    code, body = _post(f"{url}/api/train", {"client": "web/client-flask", "max_steps": 0})
    assert code == 422 and "between 1 and 2000" in body["detail"]
    code, body = _post(f"{url}/api/train", {"client": "../x", "max_steps": 5})
    assert code == 422
    # Another client's adapter would never show on this page.
    code, body = _post(f"{url}/api/train", {"client": "web/client-requests", "max_steps": 5,
                                            "dp": True})
    assert code == 422 and "this page is for client-flask" in body["detail"]
    code, body = _post(f"{url}/api/train", {"client": "web/client-flask", "max_steps": 5})
    assert code == 422 and "train with DP" in body["detail"]


def _contract_adapter(root, client, *, seed, epsilon=8.0):
    """A client adapter at the D3 contract's rank (16) and modules, tiny width."""
    import numpy as np

    from edge import wire

    rng = np.random.default_rng(seed)
    sd = {}
    for layer in range(2):
        for m in wire.TARGET_MODULES:
            sd[wire.peft_key(layer, m, "lora_A")] = rng.normal(size=(16, 8)).astype(np.float32)
            sd[wire.peft_key(layer, m, "lora_B")] = rng.normal(size=(8, 16)).astype(np.float32)
    wire.save_peft_dir(root / client / "adapter", sd,
                       {"r": 16, "lora_alpha": 16, "target_modules": list(wire.TARGET_MODULES)})
    (root / client / "manifest.json").write_text(json.dumps({
        "client_id": f"web/{client}", "seed": 0, "privacy": {"epsilon": epsilon, "delta": 1e-5},
        "data": {"train": {"n_chunks": 10}}}), encoding="utf-8")
    return root / client / "adapter"


def test_merged_adapter_comes_back_and_seeds_the_next_training(page_url, cluster_url,  # noqa: F811
                                                                tmp_path, monkeypatch):
    import requests

    from edge import upload
    from edge.merge import CONTRACT_HYPERPARAMS, validate_compatibility

    url, page = page_url
    _register(cluster_url, "web", ["client-flask", "client-requests"])
    assert _get(f"{url}/api/status")["pulled"] is None

    # Nothing merged yet: the pull says so instead of writing anything.
    assert _post(f"{url}/api/pull", {})[0] == 202
    assert "not merged" in _wait_job(url)["error"]

    for client, seed in (("client-flask", 1), ("client-requests", 2)):
        upload.upload_adapter(_contract_adapter(tmp_path / "edges", client, seed=seed),
                              cluster_url=cluster_url, log=lambda _m: None)
    r = requests.post(f"{cluster_url}/clusters/web/aggregate", json={"aggregation": "naive"},
                      timeout=30)
    assert r.status_code == 200, r.text

    assert _post(f"{url}/api/pull", {})[0] == 202
    job = _wait_job(url)
    assert job["error"] is None and job["result"]["round_id"] == 0
    pulled = _get(f"{url}/api/status")["pulled"]
    assert pulled["round_id"] == 0 and pulled["cluster_id"] == "web"
    assert sorted(pulled["source_clients"]) == ["client-flask", "client-requests"]
    assert pulled["epsilon"] == 8.0 and "path" not in pulled

    # What came back is a PEFT directory the D3 stacking accepts as the cluster layer.
    base = page.latest_pulled("web")["path"]
    cfg = json.loads((base / "adapter_config.json").read_text())
    peft = pytest.importorskip("peft")
    client_cfg = peft.LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=sorted(CONTRACT_HYPERPARAMS["target_modules"]),
        base_model_name_or_path=CONTRACT_HYPERPARAMS["base_model_name_or_path"]).to_dict()
    validate_compatibility([cfg, client_cfg], ["cluster", "client"])
    # ...and is never offered as something to send.
    assert all("from_cluster" not in a["id"] for a in _get(f"{url}/api/adapters")["adapters"])

    seen = {}
    monkeypatch.setattr(page, "start_train", lambda **kw: seen.update(kw) or {"started": "train"})
    code, _ = _post(f"{url}/api/train", {"client": "web/client-flask", "max_steps": 5,
                                         "dp": True, "from_cluster": True})
    assert code == 202 and seen["cluster_adapter"] == str(base)


def test_training_from_cluster_needs_a_pulled_adapter(page_url, cluster_url):  # noqa: F811
    url, _ = page_url
    _register(cluster_url, "web", ["client-flask"])
    code, body = _post(f"{url}/api/train", {"client": "web/client-flask", "max_steps": 5,
                                            "dp": True, "from_cluster": True})
    assert code == 422 and "no merged cluster adapter" in body["detail"]


def test_unreachable_cluster_is_reported(tmp_path):
    page = webui.EdgePage(cluster_url="http://127.0.0.1:9", client_id=None, adapters_root=tmp_path)
    s = page.status()
    assert not s["reachable"] and "firewall" in s["hint"]


def test_train_command(tmp_path):
    page = webui.EdgePage(cluster_url="http://x", client_id="client-flask", adapters_root=tmp_path,
                          corpus_root=tmp_path / "corpus", python="py")
    cmd = page.train_command(client="web/client-flask", max_steps=40, dp=True,
                             cluster_adapter="c/web")
    assert cmd[:6] == ["py", "-m", "edge.train_client", "--client", "web/client-flask", "--max-steps"]
    assert cmd[cmd.index("--out-dir") + 1] == str(tmp_path / webui.LIVE_SET)
    assert "--dp" in cmd and cmd[cmd.index("--cluster-adapter") + 1] == "c/web"
