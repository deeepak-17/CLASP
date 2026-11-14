"""The live panel (demo_ui) against the real cluster and registry, over real HTTP.

The cluster and the registry run as uvicorn servers on free local ports, so
every hop is a socket call: demo_ui -> cluster, demo_ui -> registry, and the
cluster's own call to the registry (publish). Edges upload with
``edge.upload`` and pull with ``edge.upload.pull_cluster_adapter``, exactly as
the edge laptops do. Adapters are tiny (2 layers, rank 4, 8-dim); nothing
here needs a GPU.

The round is upload -> aggregate (the cluster's candidate) -> publish (an
inactive registry version) -> evaluate -> promote, and the registry's decision
goes back to the cluster as ``candidate/promote`` or ``candidate/reject``:
edges pull a merge only once it is promoted.
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
    monkeypatch.setenv("CLASP_REGISTRY_URL", registry_url)  # the cluster's configured registry
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
        guard = [{"benchmark": "LineCompletion", "edit_similarity": 0.6, "exact_match": 0.1,
                  "n_examples": 300}]
        return {"in_project": measured, "baseline_in_project": base, "guard": guard,
                "baseline_guard": guard, "noise_band": 0.0,
                "provenance": {"stale": False, "note": "test"}}
    return fake


def _spy_sends(monkeypatch) -> list:
    """Record every (method, url, json) the panel sends."""
    sent = []
    real_send = panel._send

    async def spy(client, method, url, **kw):
        sent.append((method, url, kw.get("json")))
        return await real_send(client, method, url, **kw)

    monkeypatch.setattr(panel, "_send", spy)
    return sent


def _cluster_web(cluster_url) -> dict:
    import requests

    return requests.get(f"{cluster_url}/clusters", timeout=10).json()["clusters"]["web"]


def _first_round_promoted(ui, cluster_url, tmp_path, monkeypatch):
    """Round 0 published, promoted (against the bare base model) and pulled."""
    assert ui.post("/api/live/register").status_code == 200
    v1 = _round(ui, cluster_url, tmp_path / "r0", seed=1)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))
    first = ui.post("/api/live/promote/web")
    assert first.status_code == 200, first.text
    assert first.json()["decision"]["action"] == "promote"
    pulled = upload.pull_cluster_adapter(tmp_path / "from_cluster", cluster_id="web",
                                         cluster_url=cluster_url, log=lambda *_: None)
    assert pulled["sha256"] == v1["sha256"]
    return v1, pulled


def _two_published_rounds(ui, cluster_url, tmp_path, monkeypatch):
    """v1 promoted and pulled; v2 aggregated on it and published as the candidate."""
    v1, pulled = _first_round_promoted(ui, cluster_url, tmp_path, monkeypatch)
    v2 = _round(ui, cluster_url, tmp_path / "r1", seed=11, pulled=pulled)
    assert v2["version"] == 2 and v2["sha256"] != v1["sha256"]
    return v1, v2


def test_the_first_merge_is_not_served_until_it_is_promoted(stack, tmp_path, monkeypatch):
    import requests

    ui, cluster_url, registry_url = stack
    assert ui.post("/api/live/register").status_code == 200
    v1 = _round(ui, cluster_url, tmp_path / "r0", seed=1)
    # published, but a candidate: the registry and the cluster serve nothing yet
    assert requests.get(f"{registry_url}/adapters/cluster-web/versions",
                        timeout=10).json()["active"] is None
    assert _cluster_web(cluster_url)["active_sha256"] is None
    assert requests.get(f"{cluster_url}/clusters/web/adapters/active", timeout=10).status_code == 404
    sent = _spy_sends(monkeypatch)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))
    body = ui.post("/api/live/promote/web").json()
    assert body["decision"]["action"] == "promote" and body["candidate_version"] == 1
    assert ("POST", f"{cluster_url}/clusters/web/adapters/candidate/promote",
            {"sha256": v1["sha256"]}) in sent
    assert _cluster_web(cluster_url)["active_sha256"] == v1["sha256"]


def test_a_reject_leaves_the_cluster_serving_the_promoted_merge(stack, tmp_path, monkeypatch):
    import requests

    ui, cluster_url, registry_url = stack
    v1, v2 = _two_published_rounds(ui, cluster_url, tmp_path, monkeypatch)
    sent = _spy_sends(monkeypatch)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.40, 0.45))
    r = ui.post("/api/live/promote/web")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"]["action"] == "reject"
    assert body["decision"]["active_version_after"] == 1
    assert body["cluster"]["step"] == "reject" and body["cluster"]["active_sha256"] == v1["sha256"]
    urls = [url for _, url, _ in sent]
    assert ("POST", f"{cluster_url}/clusters/web/adapters/candidate/reject",
            {"sha256": v2["sha256"]}) in sent
    assert not any(u.endswith("/adapters/active/restore") or u.endswith("/restore") for u in urls)
    assert not any(u.endswith("/candidate/promote") for u in urls)
    listing = _cluster_web(cluster_url)
    assert listing["active_sha256"] == v1["sha256"] and listing["candidate_sha256"] is None
    assert requests.get(f"{registry_url}/adapters/cluster-web/versions",
                        timeout=10).json()["active"] == 1
    # the panel no longer offers the rejected version as a candidate
    reg = ui.get("/api/live/state").json()["clusters"]["web"]["registry"]
    assert reg["candidate"] is None and reg["active"] == 1
    assert [v["rejected"] for v in reg["versions"]] == [False, True]
    # an edge keeps pulling what was promoted, never the rejected candidate
    again = upload.pull_cluster_adapter(tmp_path / "from_cluster2", cluster_id="web",
                                        cluster_url=cluster_url, log=lambda *_: None)
    assert again["sha256"] == v1["sha256"]


def test_promote_posts_a_line_completion_guard_for_candidate_and_baseline(stack, tmp_path,
                                                                           monkeypatch):
    ui, cluster_url, _ = stack
    _two_published_rounds(ui, cluster_url, tmp_path, monkeypatch)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))
    sent = _spy_sends(monkeypatch)
    assert ui.post("/api/live/promote/web").status_code == 200
    (body,) = [j for _, url, j in sent if url.endswith("/adapters/cluster-web/promote")]
    assert body["eval"]["adapter"]["version"] == 2
    assert [g["benchmark"] for g in body["eval"]["guard"]] == ["LineCompletion"]
    assert [g["benchmark"] for g in body["baseline_guard"]] == ["LineCompletion"]
    assert body["eval"]["guard"][0]["n_examples"] == 300


def test_a_promote_leaves_the_cluster_serving_the_new_merge(stack, tmp_path, monkeypatch):
    ui, cluster_url, _ = stack
    _, v2 = _two_published_rounds(ui, cluster_url, tmp_path, monkeypatch)
    sent = _spy_sends(monkeypatch)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))
    body = ui.post("/api/live/promote/web").json()
    assert body["decision"]["action"] == "promote" and body["decision"]["active_version_after"] == 2
    assert body["cluster"] == {"step": "promote", "active_sha256": v2["sha256"],
                               "round_id": body["cluster"]["round_id"],
                               "dropped_uploads": []}
    # the cluster is told the registry's sha256 for the promoted version
    assert ("POST", f"{cluster_url}/clusters/web/adapters/candidate/promote",
            {"sha256": v2["sha256"]}) in sent
    assert not any(url.endswith("/restore") for _, url, _ in sent)
    assert _cluster_web(cluster_url)["active_sha256"] == v2["sha256"]


def test_live_state_shows_the_candidate_until_it_is_decided(stack, tmp_path, monkeypatch):
    ui, cluster_url, _ = stack
    v1, pulled = _first_round_promoted(ui, cluster_url, tmp_path, monkeypatch)
    state = ui.get("/api/live/state").json()["clusters"]["web"]
    assert state["candidate_sha256"] is None and state["registry"]["candidate"] is None
    v2 = _round(ui, cluster_url, tmp_path / "r1", seed=11, pulled=pulled)
    state = ui.get("/api/live/state").json()["clusters"]["web"]
    assert state["candidate_sha256"] == v2["sha256"]
    assert state["cluster"]["active_sha256"] == v1["sha256"]
    assert state["registry"]["active"] == 1 and state["registry"]["candidate"] == 2
    ui.post("/api/live/promote/web")
    state = ui.get("/api/live/state").json()["clusters"]["web"]
    assert state["candidate_sha256"] is None and state["registry"]["candidate"] is None


def test_promote_without_a_candidate_is_refused(stack, tmp_path, monkeypatch):
    ui, cluster_url, _ = stack
    _first_round_promoted(ui, cluster_url, tmp_path, monkeypatch)
    refused = ui.post("/api/live/promote/web")
    assert refused.status_code == 409 and "no candidate" in refused.text


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/demo/send-to-cluster/web"),        # uploaded synthetic tensors
    ("POST", "/api/demo/promote/cluster-web"),        # sent hardcoded metrics
    ("POST", "/api/demo/demonstrate-rollback/cluster-web"),  # posted a made-up candidate
    ("POST", "/api/demo/publish-to-registry/web"),
    ("GET", "/api/edge/clients"),
    ("GET", "/static/page1.html"),
    ("GET", "/static/page2.html"),
    ("GET", "/static/page3.html"),
])
def test_the_panel_has_no_synthetic_or_placeholder_paths(method, path):
    with TestClient(panel.app) as ui:
        assert ui.request(method, path).status_code == 404
        assert b"page1.html" not in ui.get("/").content


# --------------------------------------------------------------------------- #
# Evaluate before Promote: the candidate is judged on its own measurements
# --------------------------------------------------------------------------- #
FAKE_EVALUATOR = '''
"""Stands in for the GPU scoring in scripts/evaluate_candidate.py: same CLI,
same registry reads, same evidence file; fixed scores instead of a model."""
import argparse, importlib.util, sys
import requests
spec = importlib.util.spec_from_file_location("ec", {script!r})
ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
ap = argparse.ArgumentParser()
for flag in ("--cluster", "--registry-url", "--registry-name", "--out-dir"):
    ap.add_argument(flag)
a = ap.parse_args()
listing = requests.get(f"{{a.registry_url}}/adapters/{{a.registry_name}}/versions", timeout=10).json()
cand, base = ec.pick_versions(listing, listing.get("active"))
against = f"v{{base['ref']['version']}}" if base else "the bare base model"
print(f"scoring v{{cand['ref']['version']}} against {{against}}", flush=True)
score = lambda es, g: {{"in_project": {{"edit_similarity": es, "exact_match": 0.2,
                                         "perplexity": 3.0, "n_examples": 60}},
                       "guard": {{"benchmark": "LineCompletion", "edit_similarity": g,
                                 "exact_match": 0.1, "n_examples": 4}},
                       "guard_rows": [g] * 4, "guard_examples_sha256": "f" * 64}}
doc = ec.evidence_document(a.cluster, cand, base, score(0.55, 0.6), score(0.45, 0.6),
                           band={{"value": 0.0, "client": "web/client-flask", "source": "test"}},
                           details={{"fake": True}})
ec.write_evidence(a.out_dir, a.cluster, doc)
print("done", flush=True)
'''


def _wait_eval(ui, cid="web"):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        job = ui.get(f"/api/live/evaluate/{cid}").json()
        if not job["running"] and job["exit_code"] is not None:
            return job
        time.sleep(0.2)
    raise AssertionError("evaluation did not finish")


def test_promote_waits_for_the_candidate_to_be_evaluated(stack, tmp_path, monkeypatch):
    import sys

    ui, cluster_url, _ = stack
    monkeypatch.delenv("CLASP_EVAL_EVIDENCE", raising=False)
    monkeypatch.setattr("demo_ui.evidence.LIVE_EVAL_DIR", tmp_path / "live-eval")
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_candidate.py"
    fake = tmp_path / "fake_evaluator.py"
    fake.write_text(FAKE_EVALUATOR.format(script=str(script)), encoding="utf-8")
    monkeypatch.setattr(panel, "EVALUATE_CMD", [sys.executable, str(fake)])
    assert ui.post("/api/live/register").status_code == 200
    v1 = _round(ui, cluster_url, tmp_path / "r0", seed=1)

    refused = ui.post("/api/live/promote/web")
    assert refused.status_code == 503 and "Evaluate" in refused.text

    # the first candidate is judged against the bare base model (nothing active)
    assert ui.post("/api/live/evaluate/web").status_code == 202
    job = _wait_eval(ui)
    assert job["exit_code"] == 0, job["lines"]
    assert any("scoring v1 against the bare base model" in line for line in job["lines"])
    assert ui.get("/api/live/state").json()["clusters"]["web"]["evaluated_version"] == 1
    body = ui.post("/api/live/promote/web").json()
    assert body["decision"]["action"] == "promote" and body["candidate_version"] == 1

    pulled = upload.pull_cluster_adapter(tmp_path / "from_cluster", cluster_id="web",
                                         cluster_url=cluster_url, log=lambda *_: None)
    assert pulled["sha256"] == v1["sha256"]
    _round(ui, cluster_url, tmp_path / "r1", seed=11, pulled=pulled)
    refused = ui.post("/api/live/promote/web")
    assert refused.status_code == 503 and "Evaluate" in refused.text
    assert ui.post("/api/live/evaluate/web").status_code == 202
    job = _wait_eval(ui)
    assert job["exit_code"] == 0 and any("scoring v2 against v1" in line for line in job["lines"])
    state = ui.get("/api/live/state").json()["clusters"]["web"]
    assert state["evaluated_version"] == 2

    body = ui.post("/api/live/promote/web").json()
    assert body["decision"]["action"] == "promote"
    assert body["sent"]["in_project"]["edit_similarity"] == 0.55
    assert body["provenance"]["stale"] is False
