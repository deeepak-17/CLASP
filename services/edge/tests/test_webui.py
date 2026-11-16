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
        "client_id": f"web/{client}", "seed": 0,
        # a DP round as edge.train_client records it (see tests.test_upload._adapter)
        "privacy": {"epsilon": epsilon, "delta": 1e-5, "noise_multiplier": 0.6,
                    "max_grad_norm": 1.0, "sample_rate": 1 / 160, "steps": 160,
                    "run_id": "run-1", "rounds_total": 2},
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
    assert "promoted" in _wait_job(url)["error"]

    for client, seed in (("client-flask", 1), ("client-requests", 2)):
        upload.upload_adapter(_contract_adapter(tmp_path / "edges", client, seed=seed),
                              cluster_url=cluster_url, log=lambda _m: None)
    r = requests.post(f"{cluster_url}/clusters/web/aggregate", json={"aggregation": "naive"},
                      timeout=30)
    assert r.status_code == 200, r.text
    # merged, but a candidate: nothing comes back until laptop A promotes it
    assert _get(f"{url}/api/status")["merge"]["state"] == "awaiting"
    assert _post(f"{url}/api/pull", {})[0] == 202
    assert "promoted" in _wait_job(url)["error"]
    _promote(cluster_url, r.json()["sha256"])
    assert _get(f"{url}/api/status")["merge"]["label"] == "Merged and promoted"

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


def test_train_command_carries_fedprox_and_the_dp_sequence_length(tmp_path):
    from edge import fedprox

    page = webui.EdgePage(cluster_url="http://x", client_id="client-flask", adapters_root=tmp_path,
                          python="py")
    dp = page.train_command(client="web/client-flask", max_steps=40, dp=True, cluster_adapter=None)
    assert dp[dp.index("--fedprox-mu") + 1] == str(fedprox.DEFAULT_MU)
    # 1024 peaked at 4.447 GB on a 4 GB card under DP (no gradient checkpointing)
    assert dp[dp.index("--seq-len") + 1] == "512"
    loose = webui.EdgePage(cluster_url="http://x", client_id="client-flask", adapters_root=tmp_path,
                           python="py", fedprox_mu=0.1, allow_non_dp=True)
    plain = loose.train_command(client="web/client-flask", max_steps=40, dp=False,
                                cluster_adapter=None)
    assert plain[plain.index("--fedprox-mu") + 1] == "0.1" and "--seq-len" not in plain


def _promote(url, sha256):
    import requests

    r = requests.post(f"{url}/clusters/web/adapters/candidate/promote",
                      json={"sha256": sha256}, timeout=10)
    assert r.status_code == 200, r.text


def _plain_round(url, root, round_id, seeds, *, promote=True):
    import requests

    from edge import upload

    for client, seed in zip(("client-flask", "client-requests"), seeds):
        upload.upload_adapter(_contract_adapter(root / f"r{round_id}", client, seed=seed),
                              cluster_url=url, allow_reset=True,  # a fresh full round
                              log=lambda _m: None)
    # rank-16 contract adapters at width 8: only the naive ablation can keep rank 16
    r = requests.post(f"{url}/clusters/web/aggregate", json={"aggregation": "naive"},
                      timeout=30)
    assert r.status_code == 200, r.text
    if promote:
        _promote(url, r.json()["sha256"])
    return r.json()


def test_the_page_follows_the_adapter_the_cluster_serves_not_the_round_number(
        page_url, cluster_url, tmp_path):  # noqa: F811
    """An operator restore (or a restarted cluster) can serve a different adapter
    under a round number this laptop already has. The page compares digests."""
    url, page = page_url
    _register(cluster_url, "web", ["client-flask", "client-requests"])
    first = _plain_round(cluster_url, tmp_path / "e", 0, (1, 2))
    _post(f"{url}/api/pull", {})
    _wait_job(url)
    status = _get(f"{url}/api/status")
    assert status["pulled"]["sha256"] == first["sha256"] and status["pulled"]["current"] is True
    assert first["sha256"][:12] in str(page.latest_pulled("web", first["sha256"])["path"])

    second = _plain_round(cluster_url, tmp_path / "e", 1, (3, 4))
    status = _get(f"{url}/api/status")
    assert status["clusters"]["web"]["active_sha256"] == second["sha256"]
    assert status["pulled"]["current"] is False
    # training "from the merged adapter" means the one served now, not the last one pulled
    code, body = _post(f"{url}/api/train", {"client": "web/client-flask", "max_steps": 5,
                                            "dp": True, "from_cluster": True})
    assert code == 422 and "pull it first" in body["detail"]
    _post(f"{url}/api/pull", {})
    _wait_job(url)
    status = _get(f"{url}/api/status")
    assert status["pulled"]["sha256"] == second["sha256"] and status["pulled"]["current"] is True


def test_cards_say_what_an_adapter_was_built_on(tmp_path):
    pulled = {"cluster_id": "web", "round_id": 3, "sha256": "ab" * 32}
    for name, d3 in (("client-flask", {"alpha": 0.5, "cluster_adapter": "x", "cluster_pull": pulled}),):
        adapter = _adapter(tmp_path / "live-1", name, seed=1, epsilon=8.0)
        m = adapter.parent / "manifest.json"
        m.write_text(json.dumps({**json.loads(m.read_text()), "d3": d3,
                                 "fedprox": {"mu": 0.01}}), encoding="utf-8")
    foreign = _adapter(tmp_path / "round2_d3", "client-flask", seed=2, epsilon=8.0)
    fm = foreign.parent / "manifest.json"
    fm.write_text(json.dumps({**json.loads(fm.read_text()),
                              "d3": {"alpha": 0.5, "cluster_adapter": "registry/cluster-web/v2"}}),
                  encoding="utf-8")
    page = webui.EdgePage(cluster_url="http://x", client_id="client-flask", adapters_root=tmp_path)
    offered = page.adapters()
    # trained on a cluster adapter that never came from this cluster: cannot be sent, not offered
    assert [a["id"] for a in offered] == ["live-1/client-flask"]
    card = offered[0]
    assert card["base_round"] == 3 and card["base_sha256"] == "ab" * 32 and card["fedprox_mu"] == 0.01
    with pytest.raises(ValueError, match="did not come from this cluster"):
        page.start_upload("round2_d3/client-flask")


# --------------------------------------------------------------------------- #
# step 4 of the tracker follows the candidate: awaiting, promoted or rejected
# --------------------------------------------------------------------------- #
def _fake_clusters(monkeypatch, page, responses):
    """``GET /clusters`` answers each of ``responses`` in turn (web only)."""
    from edge import upload

    queue = list(responses)

    def fake(http, url, timeout=5):
        active, candidate = queue.pop(0)
        return {"web": {"round_id": 1, "pending_uploads": 0, "members": ["client-flask"],
                        "uploaded_clients": [], "aggregated_clients": ["client-flask"],
                        "aggregated_round_id": 0, "has_active_adapter": active is not None,
                        "active_sha256": active, "candidate_sha256": candidate}}

    monkeypatch.setattr(upload, "list_clusters", fake)


def _step4(page):
    return page.status()["merge"]


def test_the_tracker_awaits_evaluation_then_shows_the_promote(tmp_path, monkeypatch):
    page = webui.EdgePage(cluster_url="http://x", client_id="client-flask", adapters_root=tmp_path)
    a, b = "a" * 64, "b" * 64
    _fake_clusters(monkeypatch, page, [(a, b), (a, b), (b, None), (b, None)])
    for _ in range(2):
        m = _step4(page)
        assert m["state"] == "awaiting" and m["label"] == "Awaiting evaluation"
        assert m["candidate_sha256"] == b and m["active_sha256"] == a
    m = _step4(page)
    assert m["state"] == "promoted" and m["label"] == "Merged and promoted"
    assert m["active_sha256"] == b and m["decided_sha256"] == b
    assert _step4(page)["label"] == "Merged and promoted"  # stays until the next candidate


def test_the_tracker_shows_a_reject_and_keeps_the_previous_adapter(tmp_path, monkeypatch):
    page = webui.EdgePage(cluster_url="http://x", client_id="client-flask", adapters_root=tmp_path)
    a, b, c = "a" * 64, "b" * 64, "c" * 64
    _fake_clusters(monkeypatch, page, [(a, b), (a, None), (a, None), (a, c)])
    assert _step4(page)["label"] == "Awaiting evaluation"
    m = _step4(page)
    assert m["state"] == "rejected" and m["label"] == "Rejected, previous adapter kept"
    assert m["active_sha256"] == a and m["decided_sha256"] == b
    assert _step4(page)["state"] == "rejected"
    # the next round's candidate starts the wait again
    assert _step4(page)["state"] == "awaiting"


def test_the_tracker_follows_the_very_first_candidate(tmp_path, monkeypatch):
    page = webui.EdgePage(cluster_url="http://x", client_id="client-flask", adapters_root=tmp_path)
    b = "b" * 64
    _fake_clusters(monkeypatch, page, [(None, None), (None, b), (None, None)])
    assert _step4(page)["state"] is None  # nothing merged yet
    assert _step4(page)["state"] == "awaiting"
    m = _step4(page)  # rejected with nothing active: still nothing to pull
    assert m["state"] == "rejected" and m["active_sha256"] is None


def test_a_candidate_is_never_pulled_and_a_reject_keeps_the_pulled_adapter(
        page_url, cluster_url, tmp_path):  # noqa: F811
    """Against the real cluster: auto-pull keys on active_sha256 only."""
    import requests

    url, page = page_url
    _register(cluster_url, "web", ["client-flask", "client-requests"])
    first = _plain_round(cluster_url, tmp_path / "e", 0, (1, 2))
    _post(f"{url}/api/pull", {})
    _wait_job(url)
    second = _plain_round(cluster_url, tmp_path / "e", 1, (3, 4), promote=False)
    status = _get(f"{url}/api/status")
    assert status["merge"]["label"] == "Awaiting evaluation"
    assert status["clusters"]["web"]["active_sha256"] == first["sha256"]
    assert status["pulled"]["sha256"] == first["sha256"] and status["pulled"]["current"] is True
    r = requests.post(f"{cluster_url}/clusters/web/adapters/candidate/reject",
                      json={"sha256": second["sha256"]}, timeout=10)
    assert r.status_code == 200, r.text
    status = _get(f"{url}/api/status")
    assert status["merge"]["label"] == "Rejected, previous adapter kept"
    assert status["pulled"]["sha256"] == first["sha256"] and status["pulled"]["current"] is True
    assert page.latest_pulled("web", second["sha256"]) is None


# --------------------------------------------------------------------------- #
# D7: the page shows the run's privacy budget
# --------------------------------------------------------------------------- #
def test_page_status_reports_the_budget_and_disables_upload_when_exhausted(tmp_path):
    from edge import upload
    from tests.test_upload import ONE_ROUND, TWO_ROUNDS, FakeCluster

    fake = FakeCluster()
    page = webui.EdgePage(cluster_url="http://cluster", client_id="client-flask",
                          adapters_root=tmp_path, http=fake)
    adapters = [_adapter(tmp_path / f"live-{i}", "client-flask", seed=i, epsilon=ONE_ROUND)
                for i in (1, 2, 3)]
    assert page.status()["privacy"] == {"run_id": "run-1", "epsilon_spent": 0.0, "budget": 8.0,
                                        "rounds_used": 0, "rounds_total": 2}
    assert all(a["budget_ok"] and not a["uploaded"] for a in page.adapters())

    for adapter in adapters[:2]:
        upload.upload_adapter(adapter, cluster_url="http://cluster", http=fake,
                              ledger_root=page.ledger_root, log=lambda _m: None)
    spent = page.status()["privacy"]
    assert spent["rounds_used"] == 2 and spent["rounds_total"] == 2
    assert spent["epsilon_spent"] == pytest.approx(TWO_ROUNDS, abs=1e-3)
    by_id = {a["id"]: a for a in page.adapters()}
    # the two that were sent may be re-sent (a retry charges nothing); the third may not
    assert by_id["live-1/client-flask"]["uploaded"] and by_id["live-1/client-flask"]["budget_ok"]
    assert by_id["live-2/client-flask"]["uploaded"] and by_id["live-2/client-flask"]["budget_ok"]
    third = by_id["live-3/client-flask"]
    assert third["budget_ok"] is False and "run-1" in third["budget_reason"]
    with pytest.raises(ValueError, match="run-1"):
        page.start_upload("live-3/client-flask")
    assert page.job.snapshot()["running"] is False


def test_the_page_trains_for_its_run(tmp_path):
    page = webui.EdgePage(cluster_url="http://x", client_id="client-flask", adapters_root=tmp_path,
                          python="py", run_id="check-1")
    cmd = page.train_command(client="web/client-flask", max_steps=40, dp=True, cluster_adapter=None)
    assert cmd[cmd.index("--run-id") + 1] == "check-1"
    assert page.ledger_root == tmp_path / "privacy_ledger"


def test_the_page_shows_the_budget_line():
    html = webui.PAGE.read_text(encoding="utf-8")
    assert 'id="budget"' in html and "Privacy budget" in html

def test_the_page_signs_its_uploads_with_the_edges_key(tmp_path):
    pytest.importorskip("security.auth.hmac_signer")
    from tests.test_upload import FakeCluster

    fake = FakeCluster()
    _adapter(tmp_path / "live-1", "client-flask", seed=1, epsilon=3.2254)
    page = webui.EdgePage(cluster_url="http://cluster", client_id="client-flask",
                          adapters_root=tmp_path, http=fake, hmac_secret="k" * 64)
    page.start_upload("live-1/client-flask")
    deadline = time.monotonic() + 30
    while page.job.snapshot()["running"] and time.monotonic() < deadline:
        time.sleep(0.05)
    job = page.job.snapshot()
    assert job["error"] is None, job
    assert len(fake.uploads[0]["signature"]) == 64
    # the key is not in the job log or the status the browser reads
    assert "k" * 64 not in json.dumps(job) + json.dumps(page.status(), default=str)

# --------------------------------------------------------------------------- #
# U5: the developer logs in to the edge page (JWT)
# --------------------------------------------------------------------------- #
DEV_PASSWORD = "developer pass phrase"


def _request(url, method="GET", body=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def login_page(tmp_path):
    security_users = pytest.importorskip("security.auth.users")
    from tests.test_upload import FakeCluster

    users = security_users.UserStore(tmp_path / "users.json")
    users.add("dev-flask", DEV_PASSWORD, "developer", subject="client-flask")
    users.add("dev-numpy", DEV_PASSWORD, "developer", subject="client-numpy")
    users.add("operator", DEV_PASSWORD, "operator")
    _adapter(tmp_path / "live-1", "client-flask", seed=1, epsilon=3.2254)
    page = webui.EdgePage(cluster_url="http://cluster", client_id="client-flask",
                          adapters_root=tmp_path, http=FakeCluster(), users=users)
    httpd = webui.serve(page, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", page, users
    httpd.shutdown()
    httpd.server_close()


def _token(url, username="dev-flask", password=DEV_PASSWORD):
    status, body = _request(f"{url}/api/login", "POST", {"username": username,
                                                         "password": password})
    assert status == 200, body
    return body["token"]


def test_the_edge_page_logs_a_developer_in(login_page):
    url, _, _ = login_page
    status, body = _request(f"{url}/api/login", "POST",
                            {"username": "dev-flask", "password": DEV_PASSWORD})
    assert status == 200 and body["role"] == "developer" and body["expires_at"]
    for bad in ({"username": "dev-flask", "password": "wrong"}, {"username": "ghost",
                                                                 "password": DEV_PASSWORD}, {}):
        assert _request(f"{url}/api/login", "POST", bad)[0] == 401


def test_the_edge_pages_api_needs_the_token(login_page):
    url, page, _ = login_page
    for method, path, body in (("GET", "/api/status", None), ("GET", "/api/adapters", None),
                               ("GET", "/api/job", None),
                               ("POST", "/api/upload", {"adapter": "live-1/client-flask"}),
                               ("POST", "/api/pull", {}),
                               ("POST", "/api/train", {"client": "web/client-flask"})):
        assert _request(f"{url}{path}", method, body)[0] == 401, path
    assert page.job.snapshot()["kind"] is None          # nothing was started
    with urllib.request.urlopen(url, timeout=10) as r:   # the page itself loads: it shows the form
        assert b"CLASP Edge" in r.read()
    assert _request(f"{url}/api/health")[1] == {"status": "ok", "login_required": True}
    token = _token(url)
    status, adapters = _request(f"{url}/api/adapters", token=token)
    assert status == 200 and [a["id"] for a in adapters["adapters"]] == ["live-1/client-flask"]
    assert _request(f"{url}/api/status", token=token)[0] == 200


def test_another_clients_developer_or_an_operator_gets_403(login_page):
    url, _, _ = login_page
    for who in ("dev-numpy", "operator"):
        status, body = _request(f"{url}/api/adapters", token=_token(url, who))
        assert status == 403, who
        assert "client-flask" in body["detail"]


def test_an_expired_or_tampered_token_is_401_on_the_edge_page(login_page):
    from security.auth.jwt_handler import JWTConfig
    from security.auth.users import issue_token

    url, page, users = login_page
    expired = issue_token(users, "dev-flask", DEV_PASSWORD,
                          JWTConfig(secret_key=page.jwt.secret_key, token_expiry_minutes=-1))
    status, body = _request(f"{url}/api/status", token=expired)
    assert status == 401 and "expired" in body["detail"]
    assert _request(f"{url}/api/status", token=_token(url)[:-3] + "abc")[0] == 401


def test_without_a_users_file_the_edge_page_runs_open(page_url):
    url, _ = page_url
    assert _request(f"{url}/api/adapters")[0] == 200
    assert _request(f"{url}/api/health")[1] == {"status": "ok", "login_required": False}
    status, body = _request(f"{url}/api/login", "POST", {"username": "x", "password": "y"})
    assert status == 404 and "not enabled" in body["detail"]


def test_the_edge_page_has_a_login_form():
    html = webui.PAGE.read_text(encoding="utf-8")
    assert 'id="login"' in html and 'type="password"' in html and "sessionStorage" in html
    assert "localStorage" not in html

# --------------------------------------------------------------------------- #
# C1: the edge page over HTTPS (a server certificate; people log in)
# --------------------------------------------------------------------------- #
def test_the_edge_page_serves_https_when_given_a_page_certificate(tmp_path):
    import ssl

    pytest.importorskip("cryptography")
    ca_mod = pytest.importorskip("security.mtls.ca")
    certs_mod = pytest.importorskip("security.mtls.certs")
    from cryptography.hazmat.primitives import serialization

    from tests.test_upload import FakeCluster

    ca_key, ca_cert = ca_mod.create_ca(common_name="clasp-dev-ca", key_size=2048)
    key, cert = certs_mod.generate_service_cert(ca_key, ca_cert, "client-flask-page",
                                                sans=["localhost", "127.0.0.1"])
    (tmp_path / "ca.pem").write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    (tmp_path / "page.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (tmp_path / "page.key").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    page = webui.EdgePage(cluster_url="http://cluster", client_id="client-flask",
                          adapters_root=tmp_path, http=FakeCluster())
    httpd = webui.serve(page, "127.0.0.1", 0, tls=(tmp_path / "page.pem", tmp_path / "page.key"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        port = httpd.server_address[1]
        ctx = ssl.create_default_context(cafile=str(tmp_path / "ca.pem"))   # no client certificate
        with urllib.request.urlopen(f"https://localhost:{port}/api/health", timeout=10,
                                    context=ctx) as r:
            assert json.loads(r.read())["status"] == "ok"
        with pytest.raises(OSError):                                        # plain HTTP is refused
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5)
        # ... and a client that says nothing does not stall the page for others
        import socket

        idle = socket.create_connection(("127.0.0.1", port), timeout=5)
        try:
            with urllib.request.urlopen(f"https://localhost:{port}/api/health", timeout=10,
                                        context=ctx) as r:
                assert r.status == 200
        finally:
            idle.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_page_tls_comes_from_flags_or_the_environment_and_is_all_or_nothing(tmp_path, monkeypatch):
    cert, key = tmp_path / "page.pem", tmp_path / "page.key"
    cert.write_text("x", encoding="utf-8")
    key.write_text("x", encoding="utf-8")
    assert webui.page_tls(None, None, {}) is None
    assert webui.page_tls(cert, key, {}) == (cert, key)
    assert webui.page_tls(None, None, {"CLASP_PAGE_TLS_CERT": str(cert),
                                       "CLASP_PAGE_TLS_KEY": str(key)}) == (cert, key)
    with pytest.raises(ValueError, match="both"):
        webui.page_tls(cert, None, {})
    with pytest.raises(ValueError, match="not found"):
        webui.page_tls(tmp_path / "no.pem", key, {})
