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
             epsilon=None, n_chunks: int = 12, sigma: float = 0.6,
             run_id: str = "run-1") -> Path:
    rng = np.random.default_rng(seed)
    r, d = 4, 8
    sd = {}
    for layer in range(2):
        for m in wire.TARGET_MODULES:
            sd[wire.peft_key(layer, m, "lora_A")] = rng.normal(size=(r, d)).astype(np.float32)
            sd[wire.peft_key(layer, m, "lora_B")] = rng.normal(size=(d, r)).astype(np.float32)
    adapter = wire.save_peft_dir(root / client / "adapter", sd,
                                 {"r": r, "lora_alpha": r, "target_modules": list(wire.TARGET_MODULES)})
    # a DP adapter as edge.train_client records it: the round's epsilon plus what
    # the run's ledger composes it from (sigma, sample rate, steps)
    privacy = {"epsilon": epsilon, "delta": 1e-5, "noise_multiplier": sigma,
               "max_grad_norm": 1.0, "sample_rate": 1 / 160, "steps": 160,
               "run_id": run_id, "rounds_total": 2} if epsilon is not None else None
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
def _merge_round(url, tmp_path, cluster="web", clients=("client-flask", "client-requests"),
                 *, promote=True):
    import requests

    _register(url, cluster, list(clients))
    for i, c in enumerate(clients):
        upload.upload_adapter(_adapter(tmp_path / "r0", c, seed=10 + i, cluster=cluster),
                              cluster_url=url, log=lambda *_: None)
    r = requests.post(f"{url}/clusters/{cluster}/aggregate", json={}, timeout=60)
    assert r.status_code == 200, r.text
    merged = r.json()
    # the aggregate is a candidate; edges see it only once it is promoted
    if promote:
        p = requests.post(f"{url}/clusters/{cluster}/adapters/candidate/promote",
                          json={"sha256": merged["sha256"]}, timeout=10)
        assert p.status_code == 200, p.text
    return merged


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


def test_a_merge_that_is_not_promoted_is_not_pulled(cluster_url, tmp_path):
    import requests

    merged = _merge_round(cluster_url, tmp_path, promote=False)
    listing = requests.get(f"{cluster_url}/clusters", timeout=10).json()["clusters"]["web"]
    assert listing["candidate_sha256"] == merged["sha256"] and listing["active_sha256"] is None
    with pytest.raises(upload.UploadError) as err:
        upload.pull_cluster_adapter(tmp_path / "from_cluster", cluster_id="web",
                                    cluster_url=cluster_url, log=lambda *_: None)
    assert err.value.status == 404 and "promoted" in err.value.hint
    # rejected: still nothing to pull
    r = requests.post(f"{cluster_url}/clusters/web/adapters/candidate/reject",
                      json={"sha256": merged["sha256"]}, timeout=10)
    assert r.status_code == 200 and r.json()["active_sha256"] is None
    with pytest.raises(upload.UploadError):
        upload.pull_cluster_adapter(tmp_path / "from_cluster", cluster_id="web",
                                    cluster_url=cluster_url, log=lambda *_: None)


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


def test_a_bare_base_adapter_after_a_merge_is_not_sent(cluster_url, tmp_path):
    """Review #1: once the cluster serves a merged adapter, a full adapter trained on
    the bare base would replace it (averaged), dropping the earlier rounds."""
    _merge_round(cluster_url, tmp_path)
    late = _adapter(tmp_path / "late", "client-flask", seed=50)
    with pytest.raises(upload.UploadError, match="bare base") as err:
        upload.upload_adapter(late, cluster_url=cluster_url, log=lambda *_: None)
    assert err.value.status is None and "--allow-reset" in err.value.hint
    out = upload.upload_adapter(late, cluster_url=cluster_url, allow_reset=True,
                                log=lambda *_: None)
    assert out["base"] is None


# --------------------------------------------------------------------------- #
# D7: the run's privacy budget is charged at upload (U3, C8)
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


class FakeCluster:
    """A requests-shaped session that records every call and answers like the cluster."""

    def __init__(self):
        self.requests = []
        self.upload_status = 201
        self.clusters = {"web": {"round_id": 1, "members": ["client-flask"],
                                 "active_sha256": None}}

    def get(self, url, timeout=None):
        self.requests.append(("GET", url, None))
        return _Resp(200, {"clusters": self.clusters})

    def post(self, url, json=None, timeout=None):
        self.requests.append(("POST", url, json))
        if self.upload_status == 201:
            return _Resp(201, {"pending_uploads": 1, "round_id": json["round_id"]})
        return _Resp(self.upload_status, {"detail": "round_id mismatch: stale round"})

    @property
    def uploads(self):
        return [body for method, _, body in self.requests if method == "POST"]


@pytest.fixture
def fake_cluster():
    return FakeCluster()


# sigma 0.6 at q = 1/160 over 160 steps: 3.2254 for one round, 3.7198 for two (PRV)
ONE_ROUND, TWO_ROUNDS = 3.2254, 3.7198


def _send(adapter, fake, root, **kw):
    return upload.upload_adapter(adapter, cluster_url="http://cluster", http=fake,
                                 ledger_root=root / "ledger", log=lambda _m: None, **kw)


def _ledger(root, client="client-flask", run="run-1"):
    path = root / "ledger" / client / f"{run}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def test_upload_sends_cumulative_epsilon_and_charges_after_201(tmp_path, fake_cluster):
    a1 = _adapter(tmp_path / "r1", "client-flask", seed=1, epsilon=ONE_ROUND)
    out = _send(a1, fake_cluster, tmp_path)
    sent = fake_cluster.uploads[0]["privacy"]
    assert sent["epsilon"] == ONE_ROUND and sent["rounds_accounted"] == 1
    assert sent["epsilon_cumulative"] == pytest.approx(ONE_ROUND, abs=1e-3)
    assert sent["sample_rate"] == 1 / 160 and sent["steps"] == 160
    assert out["privacy"]["epsilon_cumulative"] == sent["epsilon_cumulative"]
    doc = _ledger(tmp_path)
    assert doc["client_id"] == "client-flask" and doc["run_id"] == "run-1"
    assert [r["round"] for r in doc["records"]] == [1]
    assert doc["records"][0]["adapter_sha256"] == upload.adapter_sha256(a1)

    a2 = _adapter(tmp_path / "r2", "client-flask", seed=2, epsilon=ONE_ROUND)
    _send(a2, fake_cluster, tmp_path)
    sent = fake_cluster.uploads[1]["privacy"]
    assert sent["rounds_accounted"] == 2
    # composed by the accountant, not added up
    assert sent["epsilon_cumulative"] == pytest.approx(TWO_ROUNDS, abs=1e-3)
    assert [r["round"] for r in _ledger(tmp_path)["records"]] == [1, 2]


def test_upload_refused_by_the_cluster_does_not_charge(tmp_path, fake_cluster):
    a1 = _adapter(tmp_path / "r1", "client-flask", seed=1, epsilon=ONE_ROUND)
    fake_cluster.upload_status = 409
    with pytest.raises(upload.UploadError) as exc:
        _send(a1, fake_cluster, tmp_path)
    assert exc.value.status == 409
    assert len(fake_cluster.uploads) == 1
    assert _ledger(tmp_path) is None


def test_retrying_the_same_adapter_does_not_charge_twice(tmp_path, fake_cluster):
    a1 = _adapter(tmp_path / "r1", "client-flask", seed=1, epsilon=ONE_ROUND)
    _send(a1, fake_cluster, tmp_path)
    _send(a1, fake_cluster, tmp_path)
    first, again = (u["privacy"] for u in fake_cluster.uploads)
    assert again == first and again["rounds_accounted"] == 1
    assert len(_ledger(tmp_path)["records"]) == 1


def test_third_upload_in_a_run_fails_before_any_request(tmp_path, fake_cluster):
    for seed in (1, 2):
        _send(_adapter(tmp_path / f"r{seed}", "client-flask", seed=seed, epsilon=ONE_ROUND),
              fake_cluster, tmp_path)
    before = _ledger(tmp_path)
    fake_cluster.requests.clear()
    a3 = _adapter(tmp_path / "r3", "client-flask", seed=3, epsilon=ONE_ROUND)
    with pytest.raises(upload.UploadError) as exc:
        _send(a3, fake_cluster, tmp_path)
    assert fake_cluster.requests == []
    text = str(exc.value)
    assert "run-1" in text and "of 8" in text and "--run-id" in text
    assert f"{TWO_ROUNDS:.2f}" in text  # the epsilon already spent
    assert _ledger(tmp_path) == before


def test_a_round_that_would_pass_the_budget_fails_before_any_request(tmp_path, fake_cluster):
    # sigma 0.4726 fits two rounds; a second round at sigma 0.3 does not
    _send(_adapter(tmp_path / "r1", "client-flask", seed=1, epsilon=6.67, sigma=0.4726),
          fake_cluster, tmp_path)
    fake_cluster.requests.clear()
    loud = _adapter(tmp_path / "r2", "client-flask", seed=2, epsilon=20.0, sigma=0.3)
    with pytest.raises(upload.UploadError, match="over the budget of 8"):
        _send(loud, fake_cluster, tmp_path)
    assert fake_cluster.requests == []
    assert len(_ledger(tmp_path)["records"]) == 1


def test_a_new_run_id_is_a_new_ledger(tmp_path, fake_cluster):
    for seed in (1, 2):
        _send(_adapter(tmp_path / f"r{seed}", "client-flask", seed=seed, epsilon=ONE_ROUND),
              fake_cluster, tmp_path)
    fresh = _adapter(tmp_path / "r3", "client-flask", seed=3, epsilon=ONE_ROUND, run_id="run-2")
    _send(fresh, fake_cluster, tmp_path)
    assert fake_cluster.uploads[-1]["privacy"]["rounds_accounted"] == 1
    assert len(_ledger(tmp_path, run="run-2")["records"]) == 1


def test_non_dp_adapter_is_untouched_by_the_ledger(tmp_path, fake_cluster):
    plain = _adapter(tmp_path / "r1", "client-flask", seed=1)
    _send(plain, fake_cluster, tmp_path)
    sent = fake_cluster.uploads[0]["privacy"]
    assert sent["epsilon"] is None and sent["epsilon_cumulative"] is None
    assert not (tmp_path / "ledger").exists()


def test_dp_adapter_trained_before_the_run_budget_is_not_sent(tmp_path, fake_cluster):
    old = _adapter(tmp_path / "r1", "client-flask", seed=1, epsilon=7.99)
    manifest = old.parent / "manifest.json"
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["privacy"] = {"epsilon": 7.99, "delta": 1e-5}
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(upload.UploadError, match="train it again"):
        _send(old, fake_cluster, tmp_path)
    assert fake_cluster.requests == []


def test_a_corrupt_ledger_stops_the_upload(tmp_path, fake_cluster):
    path = tmp_path / "ledger" / "client-flask" / "run-1.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    a1 = _adapter(tmp_path / "r1", "client-flask", seed=1, epsilon=ONE_ROUND)
    with pytest.raises(upload.UploadError, match="unreadable"):
        _send(a1, fake_cluster, tmp_path)
    assert fake_cluster.requests == [] and path.read_text(encoding="utf-8") == "{not json"


def test_an_adapter_trained_before_this_change_is_accounted_from_its_results(tmp_path, fake_cluster):
    """Older manifests keep the sample rate and steps under ``results``: such an
    adapter is still a known DP round, so it is composed, not turned away."""
    old = _adapter(tmp_path / "r1", "client-flask", seed=1, epsilon=ONE_ROUND)
    manifest = old.parent / "manifest.json"
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["privacy"] = {"epsilon": ONE_ROUND, "delta": 1e-5, "noise_multiplier": 0.6,
                      "max_grad_norm": 1.0}
    doc["results"] = {"optimizer_steps": 160, "dp": {"sample_rate": 1 / 160}}
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    _send(old, fake_cluster, tmp_path)
    sent = fake_cluster.uploads[0]["privacy"]
    assert sent["epsilon_cumulative"] == pytest.approx(ONE_ROUND, abs=1e-3)
    assert sent["sample_rate"] == 1 / 160 and sent["steps"] == 160 and sent["rounds_accounted"] == 1
    assert _ledger(tmp_path)["records"][0]["steps"] == 160

# --------------------------------------------------------------------------- #
# U4: the upload is signed with this edge's key
# --------------------------------------------------------------------------- #
def test_upload_signs_with_the_given_key(tmp_path, fake_cluster):
    pytest.importorskip("security.auth.hmac_signer")
    adapter = _adapter(tmp_path / "r1", "client-flask", seed=1)
    out = _send(adapter, fake_cluster, tmp_path, hmac_secret="k" * 64)
    assert len(fake_cluster.uploads[0]["signature"]) == 64
    assert out["signed"] is True
    _send(adapter, fake_cluster, tmp_path)
    assert "signature" not in fake_cluster.uploads[1]


def test_the_key_defaults_to_hmac_key_beside_the_certificates(tmp_path):
    certs = tmp_path / "edge-certs" / "client-flask"
    certs.mkdir(parents=True)
    assert upload.read_hmac_key(None, certs) is None           # no file: unsigned, as before
    (certs / "hmac.key").write_text("f00d" * 16 + "\n", encoding="utf-8")
    assert upload.read_hmac_key(None, certs) == "f00d" * 16
    assert upload.read_hmac_key(None, None) is None
    other = tmp_path / "other.key"
    other.write_text("beef" * 16, encoding="utf-8")
    assert upload.read_hmac_key(other, certs) == "beef" * 16     # --hmac-key wins


def test_a_named_key_file_that_is_missing_or_empty_is_an_error(tmp_path):
    with pytest.raises(upload.UploadError, match="--hmac-key"):
        upload.read_hmac_key(tmp_path / "nope.key", None)
    empty = tmp_path / "empty.key"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(upload.UploadError, match="empty"):
        upload.read_hmac_key(empty, None)


def test_a_signature_refusal_says_to_copy_the_bundle(tmp_path, fake_cluster):
    adapter = _adapter(tmp_path / "r1", "client-flask", seed=1)
    fake_cluster.upload_status = 401

    def refuse(url, json=None, timeout=None):
        fake_cluster.requests.append(("POST", url, json))
        return _Resp(401, {"detail": {"error": "upload_signature_invalid", "hint": "x"}})

    fake_cluster.post = refuse
    with pytest.raises(upload.UploadError) as exc:
        _send(adapter, fake_cluster, tmp_path, hmac_secret="k" * 64)
    assert exc.value.status == 401 and "hmac.key" in exc.value.hint


def test_the_cli_reads_the_key_and_never_prints_it(cluster_url, tmp_path, capsys, monkeypatch):
    pytest.importorskip("security.auth.hmac_signer")
    key = "ab" * 32
    secrets = tmp_path / "upload_hmac.json"
    secrets.write_text(json.dumps({"client-flask": key}), encoding="utf-8")
    monkeypatch.setenv("CLASP_UPLOAD_HMAC_SECRETS", str(secrets))
    _register(cluster_url, "web", ["client-flask"])
    adapter = _adapter(tmp_path / "r1", "client-flask", seed=1)
    keyfile = tmp_path / "hmac.key"
    keyfile.write_text(key, encoding="utf-8")
    # unsigned: the real cluster refuses it
    assert upload.main(["--adapter", str(adapter), "--cluster-url", cluster_url]) == 1
    assert "upload_signature_invalid" in capsys.readouterr().err
    assert upload.main(["--adapter", str(adapter), "--cluster-url", cluster_url,
                        "--hmac-key", str(keyfile)]) == 0
    out = capsys.readouterr()
    assert "signed" in out.out and key not in out.out + out.err
