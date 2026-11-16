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
        "client_id": f"web/{client}", "seed": 0,
        # a DP round as edge.train_client records it. These tests are about the
        # promote flow, not the budget, so every round directory is its own run.
        "privacy": {"epsilon": 2.0, "delta": 1e-5, "noise_multiplier": 0.7,
                    "max_grad_norm": 1.0, "sample_rate": 1 / 160, "steps": 160,
                    "run_id": f"run-{root.name}", "rounds_total": 2},
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


def test_a_version_an_old_d5_rollback_refused_is_shown_rejected_and_never_offered(
        stack, tmp_path, monkeypatch):
    """A volume from before the candidate flow: D5 recorded ``rollback`` naming
    the refused version. The panel, the registry and the cluster all treat it
    as refused."""
    import registry.app as registry_app
    import requests
    from contracts import AdapterKind, AdapterRef, PromotionAction, PromotionDecision

    ui, cluster_url, registry_url = stack
    _, v2 = _two_published_rounds(ui, cluster_url, tmp_path, monkeypatch)
    registry_app.get_store().record_promotion("cluster-web", PromotionDecision(
        adapter=AdapterRef("cluster-web", 2, AdapterKind.CLUSTER, cluster_id="web"),
        action=PromotionAction.ROLLBACK, active_version_after=1,
        reason="rolled back: edit_similarity delta 0.0000 does not clear noise band 0.0200"))
    reg = ui.get("/api/live/state").json()["clusters"]["web"]["registry"]
    assert reg["candidate"] is None and reg["active"] == 1
    assert [v["rejected"] for v in reg["versions"]] == [False, True]
    refused = ui.post("/api/live/promote/web")
    assert refused.status_code == 409 and "no candidate" in refused.text
    r = requests.post(f"{registry_url}/adapters/cluster-web/restore",
                      json={"reason": "drill", "to_version": 2}, timeout=10)
    assert r.status_code == 409 and "rejected" in r.text
    r = requests.post(f"{cluster_url}/clusters/web/adapters/active/restore",
                      json={"version": 2, "adapter_name": "cluster-web",
                            "registry_url": registry_url}, timeout=30)
    assert r.status_code == 409 and r.json()["detail"]["error"] == "rejected_version"
    assert _cluster_web(cluster_url)["active_sha256"] != v2["sha256"]


def test_a_second_merge_while_a_candidate_waits_is_explained_in_a_sentence(
        stack, tmp_path, monkeypatch):
    ui, cluster_url, _ = stack
    _, v2 = _two_published_rounds(ui, cluster_url, tmp_path, monkeypatch)
    r = ui.post("/api/live/aggregate/web")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert isinstance(detail, str) and v2["sha256"][:12] in detail
    assert "{" not in detail and "candidate_pending" not in detail


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/demo/send-to-cluster/web"),      # uploaded synthetic tensors
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


# --------------------------------------------------------------------------- #
# The cluster's promote / reject is retried from the registry's cluster_sync rows
# --------------------------------------------------------------------------- #
def _registry_pending(registry_url) -> list:
    import requests

    return requests.get(f"{registry_url}/sync/pending", timeout=10).json()["pending"]


def _cluster_steps(sent) -> list:
    return [(url.rsplit("/", 1)[1], j) for _, url, j in sent if "/adapters/candidate/" in url]


def test_a_promote_with_the_cluster_unreachable_is_finished_by_the_next_state_call(
        stack, tmp_path, monkeypatch):
    ui, cluster_url, registry_url = stack
    assert ui.post("/api/live/register").status_code == 200
    v1 = _round(ui, cluster_url, tmp_path / "r0", seed=1)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))
    monkeypatch.setattr(panel, "CLUSTER_BASE", "http://127.0.0.1:9")  # nothing listens there
    body = ui.post("/api/live/promote/web")
    assert body.status_code == 200, body.text  # the registry decided; only the cluster step is open
    body = body.json()
    assert body["decision"]["action"] == "promote"
    assert body["cluster"]["step"] == "promote" and "not reachable" in body["cluster"]["error"]
    (row,) = _registry_pending(registry_url)
    assert row["action"] == "promote" and row["sha256"] == v1["sha256"] and row["attempts"] == 1
    assert row["last_error"]

    # a state call while the cluster is still down reports the open step and does not raise
    down = ui.get("/api/live/state")
    assert down.status_code == 502  # GET /clusters itself is unreachable: the pre-existing answer
    assert _registry_pending(registry_url)[0]["attempts"] == 2

    monkeypatch.setattr(panel, "CLUSTER_BASE", cluster_url)  # the cluster comes back
    sent = _spy_sends(monkeypatch)
    state = ui.get("/api/live/state").json()
    assert _cluster_steps(sent) == [("promote", {"sha256": v1["sha256"]})]
    assert _registry_pending(registry_url) == []
    assert _cluster_web(cluster_url)["active_sha256"] == v1["sha256"]
    assert state["cluster_sync"] == []

    sent.clear()  # a second call has nothing left to send
    ui.get("/api/live/state")
    assert _cluster_steps(sent) == []
    assert not any(m == "POST" and "/sync/" in url for m, url, _ in sent)


def test_state_reports_a_pending_step_instead_of_raising(stack, tmp_path, monkeypatch):
    ui, cluster_url, registry_url = stack
    assert ui.post("/api/live/register").status_code == 200
    _round(ui, cluster_url, tmp_path / "r0", seed=1)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))

    # the cluster answers GET /clusters but refuses the step: a 5xx
    real_send = panel._send

    async def broken(client, method, url, **kw):
        if "/adapters/candidate/" in url:
            return httpx.Response(500, text="boom", request=httpx.Request(method, url))
        return await real_send(client, method, url, **kw)

    import httpx

    monkeypatch.setattr(panel, "_send", broken)
    body = ui.post("/api/live/promote/web").json()
    assert body["cluster"]["step"] == "promote" and "500" in body["cluster"]["error"]
    state = ui.get("/api/live/state").json()
    (open_step,) = state["clusters"]["web"]["cluster_sync"]
    assert open_step["action"] == "promote" and open_step["attempts"] == 2
    assert "500" in open_step["last_error"]
    assert state["cluster_sync"] == [open_step]


def test_a_409_for_a_step_the_cluster_already_took_counts_as_done(stack, tmp_path, monkeypatch):
    import requests

    ui, cluster_url, registry_url = stack
    assert ui.post("/api/live/register").status_code == 200
    v1 = _round(ui, cluster_url, tmp_path / "r0", seed=1)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))
    monkeypatch.setattr(panel, "CLUSTER_BASE", "http://127.0.0.1:9")
    assert ui.post("/api/live/promote/web").status_code == 200
    # the cluster took the step meanwhile (a previous attempt whose report was lost)
    assert requests.post(f"{cluster_url}/clusters/web/adapters/candidate/promote",
                         json={"sha256": v1["sha256"]}, timeout=10).status_code == 200
    monkeypatch.setattr(panel, "CLUSTER_BASE", cluster_url)
    sent = _spy_sends(monkeypatch)
    state = ui.get("/api/live/state").json()
    assert [s for s, _ in _cluster_steps(sent)] == ["promote"]  # answered 409, judged done
    assert _registry_pending(registry_url) == [] and state["cluster_sync"] == []


def test_a_409_for_a_different_state_stays_pending(stack, tmp_path, monkeypatch):
    ui, cluster_url, registry_url = stack
    assert ui.post("/api/live/register").status_code == 200
    v1 = _round(ui, cluster_url, tmp_path / "r0", seed=1)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))
    monkeypatch.setattr(panel, "CLUSTER_BASE", "http://127.0.0.1:9")
    assert ui.post("/api/live/promote/web").status_code == 200
    monkeypatch.setattr(panel, "CLUSTER_BASE", cluster_url)
    # the cluster dropped the candidate by other means, so the promote has no
    # candidate to take and the cluster is not serving the row's sha either
    import requests

    assert requests.post(f"{cluster_url}/clusters/web/adapters/candidate/reject",
                         json={"sha256": v1["sha256"]}, timeout=10).status_code == 200
    state = ui.get("/api/live/state").json()
    (row,) = _registry_pending(registry_url)
    assert row["attempts"] >= 2 and "409" in row["last_error"]
    assert state["clusters"]["web"]["cluster_sync"][0]["id"] == row["id"]


def test_a_reject_whose_candidate_is_gone_counts_as_done(stack, tmp_path, monkeypatch):
    ui, cluster_url, registry_url = stack
    v1, v2 = _two_published_rounds(ui, cluster_url, tmp_path, monkeypatch)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.40, 0.45))
    monkeypatch.setattr(panel, "CLUSTER_BASE", "http://127.0.0.1:9")
    body = ui.post("/api/live/promote/web").json()
    assert body["decision"]["action"] == "reject" and body["cluster"]["step"] == "reject"
    (row,) = _registry_pending(registry_url)
    assert row["action"] == "reject" and row["sha256"] == v2["sha256"]
    monkeypatch.setattr(panel, "CLUSTER_BASE", cluster_url)
    sent = _spy_sends(monkeypatch)
    ui.get("/api/live/state")
    assert _cluster_steps(sent) == [("reject", {"sha256": v2["sha256"]})]
    assert _registry_pending(registry_url) == []
    listing = _cluster_web(cluster_url)
    assert listing["candidate_sha256"] is None and listing["active_sha256"] == v1["sha256"]
    sent.clear()
    ui.get("/api/live/state")
    assert _cluster_steps(sent) == []


# --------------------------------------------------------------------------- #
# A state poll that finishes the step first; steps apply in order per cluster
# --------------------------------------------------------------------------- #
def test_a_poll_that_finished_the_step_first_is_not_reported_as_an_error(stack, tmp_path,
                                                                         monkeypatch):
    ui, cluster_url, registry_url = stack
    assert ui.post("/api/live/register").status_code == 200
    v1 = _round(ui, cluster_url, tmp_path / "r0", seed=1)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))
    real = panel._reconcile_cluster_sync
    polls = []

    async def poll_first(client):
        if not polls:  # a page poll's reconciliation runs between the registry POST and ours
            polls.append(await real(client))
        return await real(client)

    monkeypatch.setattr(panel, "_reconcile_cluster_sync", poll_first)
    body = ui.post("/api/live/promote/web")
    assert body.status_code == 200, body.text
    body = body.json()
    assert body["decision"]["cluster_sync_id"] is not None
    assert "error" not in body["cluster"] and body["cluster"]["step"] == "promote"
    assert body["cluster"]["applied"] is True and body["cluster"]["active_sha256"] == v1["sha256"]
    assert "error" not in panel._live["web"]["promote"]["cluster"]
    assert _registry_pending(registry_url) == []


def _blob() -> bytes:
    import struct

    header = json.dumps({"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)


def _decide_cluster_version(registry_url, version: int, good: bool) -> int:
    """Upload version ``version`` of cluster-web (a candidate) and decide it; the sync row id."""
    import requests

    r = requests.post(f"{registry_url}/adapters/cluster-web/versions",
                      files={"file": ("a.safetensors", _blob(), "application/octet-stream")},
                      data={"meta": json.dumps({"kind": "cluster", "cluster_id": "web",
                                                "set_active": False})}, timeout=10)
    assert r.status_code == 201 and r.json()["ref"]["version"] == version, r.text
    guard = [{"benchmark": "LineCompletion", "edit_similarity": 0.6 if good else 0.4,
              "exact_match": 0.3, "n_examples": 300, "pass_at_k": {}}]
    def measured(es):
        return {"edit_similarity": es, "exact_match": 0.5, "perplexity": 3.0, "n_examples": 20}

    r = requests.post(f"{registry_url}/adapters/cluster-web/promote", timeout=10, json={
        "eval": {"adapter": {"name": "cluster-web", "version": version, "kind": "cluster",
                             "cluster_id": "web"},
                 "in_project": measured(0.8 if good else 0.705), "guard": guard,
                 "baseline_in_project": measured(0.7), "baseline_noise_band": 0.02},
        "baseline_guard": [{**guard[0], "edit_similarity": 0.6}]})
    assert r.status_code == 200, r.text
    assert r.json()["action"] == ("promote" if good else "reject")
    return r.json()["cluster_sync_id"]


def _script_cluster(monkeypatch, answers: list) -> list:
    """Answer the panel's candidate/<step> calls from ``answers`` (one status per call, in
    order) and record (step, sha) of each; every other request goes through."""
    calls, real_send = [], panel._send

    async def scripted(client, method, url, **kw):
        if "/adapters/candidate/" in url:
            calls.append((url.rsplit("/", 1)[1], kw["json"]["sha256"]))
            status = answers.pop(0)
            return httpx.Response(status, json={"active_sha256": "x"} if status == 200 else
                                  {"detail": {"error": "scripted"}},
                                  request=httpx.Request(method, url))
        return await real_send(client, method, url, **kw)

    import httpx

    monkeypatch.setattr(panel, "_send", scripted)
    return calls


def test_a_failed_reject_holds_back_the_later_promote_of_the_cluster(stack, monkeypatch):
    ui, _, registry_url = stack
    reject_id = _decide_cluster_version(registry_url, 1, good=False)
    promote_id = _decide_cluster_version(registry_url, 2, good=True)
    assert reject_id < promote_id
    calls = _script_cluster(monkeypatch, [500, 200, 200])

    state = ui.get("/api/live/state").json()  # the reject fails: the promote is not even sent
    assert [step for step, _ in calls] == ["reject"]
    assert [p["id"] for p in state["cluster_sync"]] == [reject_id, promote_id]
    assert [p["attempts"] for p in _registry_pending(registry_url)] == [1, 0]

    state = ui.get("/api/live/state").json()  # next pass: the reject lands, then the promote
    assert [step for step, _ in calls] == ["reject", "reject", "promote"]
    assert state["cluster_sync"] == [] and _registry_pending(registry_url) == []


def test_a_superseded_promote_stops_being_pending(stack, monkeypatch):
    ui, _, registry_url = stack
    old_id = _decide_cluster_version(registry_url, 1, good=True)
    new_id = _decide_cluster_version(registry_url, 2, good=True)
    # the cluster refuses the old promote (409: not its candidate, not serving it) and takes the new
    calls = _script_cluster(monkeypatch, [409, 200])
    state = ui.get("/api/live/state").json()
    assert [step for step, _ in calls] == ["promote", "promote"]
    assert state["cluster_sync"] == [] and _registry_pending(registry_url) == []
    assert old_id < new_id


def test_a_promote_behind_an_already_done_later_row_is_superseded(stack, monkeypatch):
    import requests

    ui, _, registry_url = stack
    old_id = _decide_cluster_version(registry_url, 1, good=True)
    new_id = _decide_cluster_version(registry_url, 2, good=True)
    assert requests.post(f"{registry_url}/sync/{new_id}/done",
                         json={"ok": True}, timeout=10).status_code == 200
    (row,) = _registry_pending(registry_url)
    assert row["id"] == old_id and row["superseded_by"] == new_id
    calls = _script_cluster(monkeypatch, [409])
    assert ui.get("/api/live/state").json()["cluster_sync"] == []
    assert [step for step, _ in calls] == ["promote"] and _registry_pending(registry_url) == []


# --------------------------------------------------------------------------- #
# D7: each client's privacy spend, edge ledger -> cluster -> registry -> panel
# --------------------------------------------------------------------------- #
def test_the_edges_cumulative_epsilon_reaches_the_registry_ledger_and_the_panel(stack, tmp_path):
    import requests

    ui, cluster_url, registry_url = stack
    assert ui.post("/api/live/register").status_code == 200
    sent = {}
    for i, client in enumerate(CLIENTS):
        out = upload.upload_adapter(_adapter(tmp_path / "r0", client, 1 + i), cluster_url=cluster_url,
                                    log=lambda *_: None)
        sent[client] = out["privacy"]["epsilon_cumulative"]
        assert sent[client] is not None and out["privacy"]["rounds_accounted"] == 1
    # nothing is on the registry's ledger until the merge is published
    assert requests.get(f"{registry_url}/privacy/ledger", timeout=10).json()["rows"] == []
    before = ui.get("/api/live/state").json()["clusters"]["web"]["privacy"]
    assert before == {"budget": 8.0, "rounds_total": 2, "clients": [], "error": None}

    assert ui.post("/api/live/aggregate/web").status_code == 200
    assert ui.post("/api/live/publish/web").status_code == 200
    ledger = requests.get(f"{registry_url}/privacy/ledger", params={"cluster_id": "web"},
                          timeout=10).json()
    assert ledger["budget"] == 8.0 and ledger["rounds_total"] == 2
    recorded = {r["client_id"]: r for r in ledger["rows"]}
    assert set(recorded) == set(CLIENTS)
    for client in CLIENTS:
        # the number the edge's own ledger composed, unchanged end to end
        assert recorded[client]["epsilon_cumulative"] == sent[client]
        assert recorded[client]["rounds_accounted"] == 1
        assert recorded[client]["adapter"] == "cluster-web" and recorded[client]["version"] == 1

    shown = ui.get("/api/live/state").json()["clusters"]["web"]["privacy"]
    assert shown["budget"] == 8.0 and shown["rounds_total"] == 2 and shown["error"] is None
    assert shown["clients"] == [
        {"client_id": c, "epsilon_cumulative": sent[c], "rounds_accounted": 1} for c in CLIENTS]


def test_the_panel_shows_each_clients_latest_spend(stack, tmp_path, monkeypatch):
    ui, cluster_url, registry_url = stack
    v1, pulled = _first_round_promoted(ui, cluster_url, tmp_path, monkeypatch)
    # round 2 in the same run for both clients: the cumulative grows, the row count per client is 1
    for i, client in enumerate(CLIENTS):
        d3 = {"alpha": 0.5, "cluster_adapter": pulled["path"], "cluster_pull": pulled}
        adapter = _adapter(tmp_path / "r0b", client, 21 + i, d3)
        manifest = adapter.parent / "manifest.json"
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        doc["privacy"]["run_id"] = "run-r0"      # the run round 0 was charged to
        manifest.write_text(json.dumps(doc), encoding="utf-8")
        out = upload.upload_adapter(adapter, cluster_url=cluster_url, log=lambda *_: None)
        assert out["privacy"]["rounds_accounted"] == 2
    assert ui.post("/api/live/aggregate/web").status_code == 200
    assert ui.post("/api/live/publish/web").status_code == 200
    shown = ui.get("/api/live/state").json()["clusters"]["web"]["privacy"]["clients"]
    assert [c["client_id"] for c in shown] == list(CLIENTS)
    assert all(c["rounds_accounted"] == 2 and 2.0 < c["epsilon_cumulative"] <= 8.0 for c in shown)


def test_the_panel_state_survives_a_registry_without_the_ledger_route(stack, monkeypatch):
    """An older registry (404 on /privacy/ledger) costs the privacy lines, not the page."""
    ui, _, _ = stack
    assert ui.post("/api/live/register").status_code == 200
    real = panel._send

    async def no_ledger(client, method, url, **kw):
        if "/privacy/ledger" in url:
            import httpx

            return httpx.Response(404, json={"detail": "Not Found"})
        return await real(client, method, url, **kw)

    monkeypatch.setattr(panel, "_send", no_ledger)
    state = ui.get("/api/live/state")
    assert state.status_code == 200
    privacy = state.json()["clusters"]["web"]["privacy"]
    assert privacy["clients"] == [] and "404" in privacy["error"]


def test_the_page_lists_the_privacy_spend():
    html = (panel.STATIC_DIR / "live.html").read_text(encoding="utf-8")
    assert "function privacyLines" in html and "rounds_accounted" in html
    # the merge card no longer describes the cluster epsilon as a sum
    assert "plus the adapter it builds on" not in html


# --------------------------------------------------------------------------- #
# U4: evaluation results are signed by the evaluator and verified by the registry
# --------------------------------------------------------------------------- #
EVAL_SECRET = "5e" * 32


def _signed_evidence(candidate_es: float, baseline_es: float, *, secret=EVAL_SECRET, tamper=None):
    """Evidence as scripts/evaluate_candidate.py writes it: the promote body it
    measured, signed with the evaluator's secret."""
    from security.auth.hmac_signer import sign_evaluation

    plain = _evidence(candidate_es, baseline_es)

    def fake(cluster_id, version=None, sha256=None, **_kw):
        ev = plain(cluster_id)
        result = {"adapter": {"name": f"cluster-{cluster_id}", "version": version,
                              "kind": "cluster", "cluster_id": cluster_id},
                  "in_project": ev["in_project"], "guard": ev["guard"],
                  "baseline_in_project": ev["baseline_in_project"],
                  "baseline_noise_band": ev["noise_band"], "seed": 0}
        signature = sign_evaluation(adapter=f"cluster-{cluster_id}", version=version,
                                    candidate_sha256=sha256, eval_result=result,
                                    baseline_guard=ev["baseline_guard"], secret=secret)
        promote = {"eval": result, "baseline_guard": ev["baseline_guard"],
                   "candidate_sha256": sha256, "eval_signature": signature}
        if tamper:
            tamper(promote)
        return {**ev, "promote": promote}
    return fake


def test_a_signed_evaluation_is_promoted_through_the_panel(stack, tmp_path, monkeypatch):
    ui, cluster_url, registry_url = stack
    monkeypatch.setenv("CLASP_EVAL_HMAC_SECRET", EVAL_SECRET)   # the registry's copy
    assert ui.post("/api/live/register").status_code == 200
    v1 = _round(ui, cluster_url, tmp_path / "r0", seed=1)
    sent = _spy_sends(monkeypatch)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _signed_evidence(0.50, 0.45))
    r = ui.post("/api/live/promote/web")
    assert r.status_code == 200, r.text
    assert r.json()["decision"]["action"] == "promote"
    body = next(j for m, u, j in sent if u.endswith("/adapters/cluster-web/promote"))
    assert body["candidate_sha256"] == v1["sha256"] and body["eval_signature"]
    assert _cluster_web(cluster_url)["active_sha256"] == v1["sha256"]


def test_a_result_changed_after_signing_is_refused_and_nothing_is_decided(stack, tmp_path,
                                                                           monkeypatch):
    import requests

    ui, cluster_url, registry_url = stack
    monkeypatch.setenv("CLASP_EVAL_HMAC_SECRET", EVAL_SECRET)
    assert ui.post("/api/live/register").status_code == 200
    _round(ui, cluster_url, tmp_path / "r0", seed=1)

    def inflate(promote):  # the numbers were edited after the evaluator signed them
        promote["eval"]["in_project"] = {**promote["eval"]["in_project"], "edit_similarity": 0.99}

    monkeypatch.setattr("demo_ui.evidence.evidence_for",
                        _signed_evidence(0.40, 0.45, tamper=inflate))
    r = ui.post("/api/live/promote/web")
    assert r.status_code == 401, r.text
    assert "signature" in r.json()["detail"]
    decisions = requests.get(f"{registry_url}/adapters/cluster-web/promotions", timeout=10).json()
    assert decisions["decisions"] == []
    web = _cluster_web(cluster_url)
    assert web["active_sha256"] is None and web["candidate_sha256"]  # still waiting, unserved


def test_unsigned_evidence_against_a_registry_with_a_secret_is_refused(stack, tmp_path, monkeypatch):
    ui, cluster_url, _ = stack
    monkeypatch.setenv("CLASP_EVAL_HMAC_SECRET", EVAL_SECRET)
    assert ui.post("/api/live/register").status_code == 200
    _round(ui, cluster_url, tmp_path / "r0", seed=1)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))  # no promote body
    r = ui.post("/api/live/promote/web")
    assert r.status_code == 401 and "eval_signature" in r.json()["detail"]


def test_the_panel_always_names_the_bytes_it_asks_about(stack, tmp_path, monkeypatch):
    """Without a secret the body still carries the candidate's sha256, so the
    registry can refuse an evaluation of other bytes."""
    ui, cluster_url, _ = stack
    assert ui.post("/api/live/register").status_code == 200
    v1 = _round(ui, cluster_url, tmp_path / "r0", seed=1)
    sent = _spy_sends(monkeypatch)
    monkeypatch.setattr("demo_ui.evidence.evidence_for", _evidence(0.50, 0.45))
    assert ui.post("/api/live/promote/web").status_code == 200
    body = next(j for m, u, j in sent if u.endswith("/adapters/cluster-web/promote"))
    assert body["candidate_sha256"] == v1["sha256"] and "eval_signature" not in body

# --------------------------------------------------------------------------- #
# U4: edge uploads are signed by the edge and verified by the cluster
# --------------------------------------------------------------------------- #
def test_a_signed_edge_upload_is_accepted_and_a_wrong_key_is_refused(stack, tmp_path, monkeypatch):
    ui, cluster_url, _ = stack
    keys = {"client-flask": "f1" * 32, "client-requests": "e2" * 32}
    secrets = tmp_path / "upload_hmac.json"
    secrets.write_text(json.dumps(keys), encoding="utf-8")
    monkeypatch.setenv("CLASP_UPLOAD_HMAC_SECRETS", str(secrets))
    assert ui.post("/api/live/register").status_code == 200

    flask = _adapter(tmp_path / "r0", "client-flask", 1)
    out = upload.upload_adapter(flask, cluster_url=cluster_url, hmac_secret=keys["client-flask"],
                                log=lambda *_: None)
    assert out["response"]["pending_uploads"] == 1 and out["signed"] is True

    requests_adapter = _adapter(tmp_path / "r0", "client-requests", 2)
    with pytest.raises(upload.UploadError) as wrong:       # signed with another edge's key
        upload.upload_adapter(requests_adapter, cluster_url=cluster_url,
                              hmac_secret=keys["client-flask"], log=lambda *_: None)
    assert wrong.value.status == 401 and "hmac.key" in wrong.value.hint
    with pytest.raises(upload.UploadError) as unsigned:
        upload.upload_adapter(requests_adapter, cluster_url=cluster_url, log=lambda *_: None)
    assert unsigned.value.status == 401
    assert _cluster_web(cluster_url)["uploaded_clients"] == ["client-flask"]
    # a refused DP upload was not charged to the laptop's privacy ledger
    ledger = upload.DEFAULT_LEDGER_ROOT / "client-requests"
    assert not ledger.exists() or not any(ledger.iterdir())

# --------------------------------------------------------------------------- #
# U5: people log in to the panel (JWT); tokens carry a role
# --------------------------------------------------------------------------- #
PANEL_PASSWORD = "panel pass phrase"
JWT_SECRET = "j" * 48


@pytest.fixture
def logins(stack, tmp_path, monkeypatch):
    """The stack with a users file configured: (ui, cluster_url, registry_url, users)."""
    from security.auth.users import UserStore

    users = UserStore(tmp_path / "users" / "panel.json")
    users.add("operator", PANEL_PASSWORD, "operator")
    users.add("dev-flask", "developer pass phrase", "developer", subject="client-flask")
    monkeypatch.setenv("CLASP_USERS_FILE", str(users.path))
    monkeypatch.setenv("CLASP_JWT_SECRET", JWT_SECRET)
    return (*stack, users)


def _login(ui, username="operator", password=PANEL_PASSWORD):
    return ui.post("/api/login", json={"username": username, "password": password})


def _bearer(ui, **kw):
    r = _login(ui, **kw)
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_login_gives_an_operator_token_and_a_wrong_password_does_not(logins):
    ui = logins[0]
    r = _login(ui)
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "operator" and body["token"].count(".") == 2
    import datetime as dt

    expires = dt.datetime.fromisoformat(body["expires_at"])
    left = expires - dt.datetime.now(dt.timezone.utc)
    assert dt.timedelta(hours=7, minutes=59) < left <= dt.timedelta(hours=8)
    for bad in ({"username": "operator", "password": "wrong"},
                {"username": "nobody", "password": PANEL_PASSWORD},
                {"username": "operator"}, {}):
        refused = ui.post("/api/login", json=bad)
        assert refused.status_code == 401, bad
        assert "token" not in refused.json()


def test_every_api_route_needs_a_token_except_health_and_login(logins):
    ui = logins[0]
    for method, path in (("GET", "/api/live/state"), ("GET", "/api/live/plan"),
                         ("POST", "/api/live/register"), ("POST", "/api/live/aggregate/web"),
                         ("POST", "/api/live/publish/web"), ("POST", "/api/live/promote/web"),
                         ("POST", "/api/live/evaluate/web"), ("GET", "/api/live/evaluate/web"),
                         ("GET", "/api/cluster/clusters"), ("GET", "/api/registry/adapters")):
        r = ui.request(method, path)
        assert r.status_code == 401, (method, path, r.status_code)
        assert r.headers["www-authenticate"].startswith("Bearer")
    assert ui.get("/api/health").status_code == 200
    assert ui.get("/").status_code == 200 and ui.get("/static/common.js").status_code == 200
    headers = _bearer(ui)
    assert ui.get("/api/live/plan", headers=headers).status_code == 200
    assert ui.post("/api/live/register", headers=headers).status_code == 200
    assert ui.get("/api/cluster/clusters", headers=headers).status_code == 200


def test_a_refused_call_does_nothing(logins):
    ui, cluster_url, _, _ = logins
    assert ui.post("/api/live/register").status_code == 401
    import requests

    assert requests.get(f"{cluster_url}/clusters", timeout=10).json()["clusters"].get("web") is None


def test_a_developer_token_is_not_an_operator(logins):
    ui = logins[0]
    headers = _bearer(ui, username="dev-flask", password="developer pass phrase")
    r = ui.get("/api/live/state", headers=headers)
    assert r.status_code == 403 and "operator" in r.json()["detail"]


def test_a_tampered_or_foreign_token_is_401(logins):
    from security.auth.jwt_handler import JWTConfig
    from security.auth.users import issue_token

    ui, _, _, users = logins
    good = _bearer(ui)["Authorization"]
    assert ui.get("/api/live/plan", headers={"Authorization": good[:-3] + "abc"}).status_code == 401
    forged = issue_token(users, "operator", PANEL_PASSWORD, JWTConfig(secret_key="x" * 48))
    assert ui.get("/api/live/plan",
                  headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    assert ui.get("/api/live/plan", headers={"Authorization": "Basic b3A6cHc="}).status_code == 401


def test_an_expired_token_mid_round_is_401_and_the_round_is_still_there(logins, tmp_path):
    """Review Focus 4: the page loses its login, the server loses nothing."""
    from security.auth.jwt_handler import JWTConfig
    from security.auth.users import issue_token

    ui, cluster_url, _, users = logins
    headers = _bearer(ui)
    assert ui.post("/api/live/register", headers=headers).status_code == 200
    for i, client in enumerate(CLIENTS):
        upload.upload_adapter(_adapter(tmp_path / "r0", client, 1 + i), cluster_url=cluster_url,
                              log=lambda *_: None)
    assert ui.post("/api/live/aggregate/web", headers=headers).status_code == 200
    assert ui.post("/api/live/publish/web", headers=headers).status_code == 200
    before = ui.get("/api/live/state", headers=headers).json()
    assert before["clusters"]["web"]["registry"]["candidate"] == 1

    expired = issue_token(users, "operator", PANEL_PASSWORD,
                          JWTConfig(secret_key=JWT_SECRET, token_expiry_minutes=-1))
    stale = {"Authorization": f"Bearer {expired}"}
    for method, path in (("GET", "/api/live/state"), ("GET", "/api/live/plan"),
                         ("POST", "/api/live/promote/web"), ("POST", "/api/live/aggregate/web")):
        r = ui.request(method, path, headers=stale)
        assert r.status_code == 401 and "expired" in r.json()["detail"], (method, path)

    after = ui.get("/api/live/state", headers=_bearer(ui)).json()
    web_before, web_after = before["clusters"]["web"], after["clusters"]["web"]
    for key in ("registered", "cluster", "candidate_sha256", "registry", "last", "privacy"):
        assert web_after[key] == web_before[key], key


def test_without_a_users_file_the_panel_runs_open(stack, monkeypatch):
    ui = stack[0]
    monkeypatch.delenv("CLASP_USERS_FILE", raising=False)
    assert ui.get("/api/live/plan").status_code == 200
    assert ui.post("/api/live/register").status_code == 200
    r = ui.post("/api/login", json={"username": "operator", "password": PANEL_PASSWORD})
    assert r.status_code == 404 and "not enabled" in r.json()["detail"]
    assert ui.get("/api/health").json()["login_required"] is False


def test_health_says_whether_a_login_is_required(logins):
    assert logins[0].get("/api/health").json()["login_required"] is True


def test_a_users_file_that_cannot_be_read_closes_the_panel(stack, tmp_path, monkeypatch):
    """Configured but broken is not "open"."""
    ui = stack[0]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("CLASP_USERS_FILE", str(bad))
    assert ui.get("/api/live/plan").status_code == 401
    assert ui.post("/api/login", json={"username": "operator",
                                       "password": PANEL_PASSWORD}).status_code == 500


def test_the_page_has_a_login_form_and_keeps_the_token_for_the_session():
    html = (panel.STATIC_DIR / "live.html").read_text(encoding="utf-8")
    js = (panel.STATIC_DIR / "common.js").read_text(encoding="utf-8")
    assert 'id="login"' in html and 'type="password"' in html
    assert "sessionStorage" in js and "Authorization" in js and "401" in js
    assert "localStorage" not in js and "localStorage" not in html
