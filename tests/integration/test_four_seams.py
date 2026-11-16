"""CPU-only end-to-end proof of all four integration seams (Sprint · B8).

    A   Edge  -> Cluster    POST /uploads
    B   Cluster -> Registry POST /versions          (an inactive candidate)
    C1  Registry -> Edge    GET /versions/{n}/file -> materialize -> compose
    C2  Edge  -> Registry   POST /promote         -> D5 decision, then the
                            cluster's candidate/promote or candidate/reject

The round is upload -> aggregate (the cluster's candidate) -> publish
(inactive) -> evaluate -> promote -> cluster promote -> pull: nothing is
served to an edge until the candidate is promoted, and a rejected candidate
is never served.

Everything runs in-process through FastAPI's TestClient: no sockets, no
services to start, no GPU, no model download, no HumanEval, no training. The
adapters are tiny synthetic ones (2 layers x q/k/v/o, rank 4, 32-dim, fp32)
that keep every structural property the real path depends on — multi-layer
keys, the PEFT key convention, the fp32 wire contract, rank-r truncation — and
shrink only the dimensions the contract does not pin.

The whole file is deterministic: the adapters come from a seeded RNG, the
aggregation is exact, and the "evaluation" step scores fixed prediction strings
rather than sampling from a model. Running it twice gives identical bytes, and
one test asserts exactly that.

What this test does NOT prove, stated so nobody reads more into a green tick:
the numbers are structural, not quality measurements. Real 24-layer adapters,
the real aggregation error against the exact weighted average, and real
in-project completion metrics are the live demo's job
(``scripts/demo_round.py``), not CI's.
"""
from __future__ import annotations

import json
from contextlib import contextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cluster.adapter_format import TARGET_MODULES
from edge import wire
from edge.promote import promote_candidate, resolve_guard
from edge.registry_client import ChecksumMismatch, RegistryClient
from evaluation.completion import collect_examples, in_project_dict, score

NUM_LAYERS = 2
RANK = 4
DIM = 32
CLUSTER_ID = "web"
REGISTRY_NAME = f"cluster-{CLUSTER_ID}"
BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
CLIENTS = {"client-flask": 80, "client-requests": 52, "client-werkzeug": 170}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def registry_client(tmp_path, monkeypatch):
    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "registry"))
    import registry.app as appmod

    appmod._store = None  # force re-read of CLASP_REGISTRY_DATA
    with TestClient(appmod.app) as c:
        yield c


@pytest.fixture
def cluster_client():
    import cluster.server as srv

    srv._reset_clusters()
    with TestClient(srv.app) as c:
        yield c
    srv._reset_clusters()


@pytest.fixture
def edge_registry(registry_client):
    """``RegistryClient`` talking to the registry app in-process."""
    return RegistryClient(base_url="", session=registry_client)


# --------------------------------------------------------------------------- #
# tiny synthetic client adapters, in the real PEFT key convention
# --------------------------------------------------------------------------- #
def make_client_adapter(seed: int):
    """(state_dict, config) shaped like a real trained client adapter.

    r == lora_alpha, so PEFT's scaling is exactly 1.0 and delta_W == B @ A —
    the same convention the six real adapters use, which is what lets the
    cluster's unscaled ``delta_w()`` mean the right thing.
    """
    rng = np.random.default_rng(seed)
    sd = {}
    for layer in range(NUM_LAYERS):
        for module in TARGET_MODULES:
            prefix = wire.peft_key(layer, module, "lora_A").rsplit(".lora_A", 1)[0]
            sd[f"{prefix}.lora_A.weight"] = rng.normal(0, 0.1, (RANK, DIM)).astype(np.float32)
            sd[f"{prefix}.lora_B.weight"] = rng.normal(0, 0.1, (DIM, RANK)).astype(np.float32)
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": RANK, "lora_alpha": RANK, "lora_dropout": 0.0,
        "target_modules": list(TARGET_MODULES),
        "base_model_name_or_path": BASE_MODEL,
        "use_rslora": False, "use_dora": False, "fan_in_fan_out": False,
        "lora_bias": False, "bias": "none", "inference_mode": True,
        "rank_pattern": {}, "alpha_pattern": {},
    }
    return sd, cfg


def upload_all(cluster_client, round_id: int = 0):
    for i, (name, n) in enumerate(CLIENTS.items()):
        sd, cfg = make_client_adapter(seed=100 + i + 10 * round_id)
        payload = wire.upload_payload(sd, cfg, client_id=name, cluster_id=CLUSTER_ID,
                                      round_id=round_id, num_examples=n, seed=0)
        resp = cluster_client.post("/uploads", json=payload)
        assert resp.status_code == 201, resp.text
        assert resp.json()["tensors_received"] == 2 * len(TARGET_MODULES) * NUM_LAYERS


def aggregate(cluster_client, method: str, retain: bool):
    resp = cluster_client.post("/aggregate", json={
        "cluster_id": CLUSTER_ID, "aggregation": method,
        "retain_uploads": retain, "include_manifest": True})
    assert resp.status_code == 200, resp.text
    return resp.json()


def decide_on_cluster(cluster_client, step: str) -> dict:
    """``candidate/promote`` or ``candidate/reject`` for the pending candidate."""
    sha = cluster_client.get(f"/clusters/{CLUSTER_ID}/adapters/candidate").json()["sha256"]
    resp = cluster_client.post(f"/clusters/{CLUSTER_ID}/adapters/candidate/{step}",
                               json={"sha256": sha})
    assert resp.status_code == 200, resp.text
    return resp.json()


@contextmanager
def publish_over_http(registry_client, monkeypatch):
    """Point ``cluster.server``'s outbound httpx client at the registry app.

    Seam B in the live system is a real HTTP POST from the cluster process to
    the registry process. In CI there is no second process, so the transport is
    swapped for the registry's own TestClient — the cluster's ``/publish``
    handler itself, including its metadata envelope, still runs unchanged.
    """
    import httpx

    class _Shim:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, files=None, data=None, **kw):
            path = url.split("://", 1)[-1].split("/", 1)[-1]
            return registry_client.post("/" + path, files=files, data=data)

    monkeypatch.setattr(httpx, "Client", _Shim)
    yield
    monkeypatch.undo()


# --------------------------------------------------------------------------- #
# seam A
# --------------------------------------------------------------------------- #
def test_seam_a_accepts_multilayer_peft_upload(cluster_client):
    """Defect 2 regression at the structural level: a multi-layer adapter in
    the real PEFT key convention is accepted, not rejected by the validator."""
    upload_all(cluster_client)
    health = cluster_client.get("/healthz").json()
    assert health["clusters"][CLUSTER_ID]["pending_uploads"] == 3
    assert health["clusters"][CLUSTER_ID]["pending_clients"] == sorted(CLIENTS)


def test_seam_a_enforces_fp32_wire_contract(cluster_client):
    sd, cfg = make_client_adapter(seed=1)
    sd = {k: v.astype(np.float16) for k, v in sd.items()}
    payload = wire.upload_payload(sd, cfg, client_id="c0", cluster_id=CLUSTER_ID,
                                  round_id=0, num_examples=10)
    # wire.upload_payload casts back to fp32, so the contract holds end to end;
    # bypassing it must be rejected by the server, not silently accepted.
    assert all(t["dtype"] == "float32" for t in payload["tensors"])
    payload["tensors"][0]["dtype"] = "float16"
    assert cluster_client.post("/uploads", json=payload).status_code == 422


def test_edge_and_cluster_tensor_encodings_are_identical():
    """``edge.wire`` writes its own copy of the tensor encoding so edge stays
    importable without the cluster package (docs/architecture.md §2: contracts
    is the only cross-module import path). This is the assertion that keeps the
    copy honest — it is the one place both packages are installed."""
    from cluster.schemas.messages import TensorPayload

    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    mine = wire.encode_tensor("layers.0.q_proj.lora_A.weight", arr)
    theirs = TensorPayload.from_numpy("layers.0.q_proj.lora_A.weight", arr).model_dump(mode="json")
    assert mine["dtype"] == theirs["dtype"]
    assert list(mine["shape"]) == list(theirs["shape"])
    assert mine["data_b64"] == theirs["data_b64"]
    np.testing.assert_array_equal(wire.decode_tensor(theirs), arr)
    np.testing.assert_array_equal(TensorPayload(**mine).to_numpy(), arr)


def test_edge_and_cluster_canonicalizers_agree():
    """Both modules canonicalize the safetensors header so a published adapter
    is byte-reproducible. Same reason as the tensor encoding: duplicated across
    the module boundary, so something has to keep the copies honest."""
    from safetensors.numpy import save

    from cluster.server import _canonical_safetensors

    sd = {wire.peft_key(0, "q_proj", p): np.arange(4, dtype=np.float32).reshape(2, 2)
          for p in ("lora_A", "lora_B")}
    raw = save(sd, metadata={"z": "1", "a": "2", "format": "pt"})
    assert wire.canonicalize_safetensors(raw) == _canonical_safetensors(raw)


def test_published_payload_is_byte_reproducible(cluster_client):
    """The registry records a sha256 per payload. If the same aggregate
    serialized differently every time, that digest would change run to run for
    an unchanged adapter."""
    from cluster.server import _clusters, _peft_bytes

    upload_all(cluster_client)
    aggregate(cluster_client, "svd", retain=False)
    broadcast = _clusters[CLUSTER_ID].candidate  # what /publish sends
    assert len({_peft_bytes(broadcast) for _ in range(8)}) == 1


def test_seam_a_rejects_stale_round(cluster_client):
    sd, cfg = make_client_adapter(seed=2)
    payload = wire.upload_payload(sd, cfg, client_id="c0", cluster_id=CLUSTER_ID,
                                  round_id=9, num_examples=10)
    assert cluster_client.post("/uploads", json=payload).status_code == 409


def test_clusters_do_not_share_an_upload_buffer(cluster_client):
    """Two statically-assigned clusters (D1) must aggregate separately."""
    sd, cfg = make_client_adapter(seed=3)
    for cid in ("web", "scientific"):
        payload = wire.upload_payload(sd, cfg, client_id="c0", cluster_id=cid,
                                      round_id=0, num_examples=10)
        assert cluster_client.post("/uploads", json=payload).status_code == 201
    health = cluster_client.get("/healthz").json()["clusters"]
    assert health["web"]["pending_uploads"] == 1
    assert health["scientific"]["pending_uploads"] == 1


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_returns_peft_keys_and_reconstruction_error(cluster_client):
    upload_all(cluster_client)
    broadcast = aggregate(cluster_client, "svd", retain=False)
    assert broadcast["rank"] == RANK
    assert broadcast["num_layers"] == NUM_LAYERS
    assert len(broadcast["tensors"]) == 2 * len(TARGET_MODULES) * NUM_LAYERS
    assert sorted(broadcast["source_clients"]) == sorted(CLIENTS)
    names = [t["name"] for t in broadcast["tensors"]]
    assert all(n.startswith("base_model.model.model.layers.") and ".self_attn." in n
               for n in names)

    manifest = cluster_client.get(f"/adapters/{CLUSTER_ID}/manifest").json()
    err = manifest["svd_reconstruction_error"]
    assert err["n_modules"] == len(TARGET_MODULES) * NUM_LAYERS
    assert 0.0 <= err["mean"] <= err["max"] <= 1.0
    assert manifest["aggregation_path"] == "exact_lowrank"


def test_lowrank_path_matches_the_reference_aggregate_svd(cluster_client):
    """The fast path is the same mathematics, not an approximation of it."""
    upload_all(cluster_client)
    fast = aggregate(cluster_client, "svd", retain=True)
    decide_on_cluster(cluster_client, "reject")  # one candidate at a time
    ref = cluster_client.post("/aggregate", json={
        "cluster_id": CLUSTER_ID, "aggregation": "svd",
        "exact_lowrank": False, "include_manifest": True}).json()
    assert ref["num_clients"] == fast["num_clients"]
    fast_sd = wire.tensors_from_payload(fast["tensors"])
    ref_sd = wire.tensors_from_payload(ref["tensors"])
    cfg = {"r": RANK, "lora_alpha": RANK}
    for layer in range(NUM_LAYERS):
        for module in TARGET_MODULES:
            np.testing.assert_allclose(
                wire.delta_w(fast_sd, cfg, layer, module),
                wire.delta_w(ref_sd, cfg, layer, module),
                rtol=1e-5, atol=1e-6)


def test_aggregation_is_deterministic(cluster_client):
    upload_all(cluster_client)
    first = aggregate(cluster_client, "svd", retain=True)
    decide_on_cluster(cluster_client, "reject")  # one candidate at a time
    second = aggregate(cluster_client, "svd", retain=True)
    assert [t["data_b64"] for t in first["tensors"]] == \
           [t["data_b64"] for t in second["tensors"]]


# --------------------------------------------------------------------------- #
# seam B
# --------------------------------------------------------------------------- #
def test_seam_b_publish_records_full_provenance(cluster_client, registry_client,
                                                monkeypatch):
    upload_all(cluster_client)
    aggregate(cluster_client, "svd", retain=False)
    with publish_over_http(registry_client, monkeypatch):
        resp = cluster_client.post(
            f"/adapters/{CLUSTER_ID}/publish",
            json={"registry_url": "http://registry:8004",
                  "adapter_name": REGISTRY_NAME, "round": 1})
    assert resp.status_code == 201, resp.text
    meta = resp.json()["version"]
    assert meta["ref"]["version"] == 1
    assert meta["ref"]["kind"] == "cluster"
    assert meta["ref"]["cluster_id"] == CLUSTER_ID
    assert meta["aggregation"] == "svd_exact"
    assert sorted(meta["source_clients"]) == sorted(CLIENTS)
    assert meta["hparams"]["rank"] == RANK
    assert meta["hparams"]["lora_alpha"] == RANK  # scaling pinned to 1.0
    assert len(meta["sha256"]) == 64
    assert meta["round"] == 1


def test_cluster_download_is_a_valid_safetensors_blob(cluster_client):
    upload_all(cluster_client)
    aggregate(cluster_client, "svd", retain=False)
    # a candidate is not served: /download answers only once it is promoted
    assert cluster_client.get(f"/adapters/{CLUSTER_ID}/download").status_code == 404
    decide_on_cluster(cluster_client, "promote")
    resp = cluster_client.get(f"/adapters/{CLUSTER_ID}/download")
    assert resp.status_code == 200
    sd, cfg = wire.deserialize(resp.content)
    assert len(sd) == 2 * len(TARGET_MODULES) * NUM_LAYERS
    assert cfg["r"] == RANK  # config travels inside the blob (Defect 4)
    assert all(v.dtype == np.float32 for v in sd.values())


# --------------------------------------------------------------------------- #
# seam C1
# --------------------------------------------------------------------------- #
def publish_candidate(cluster_client, registry_client, monkeypatch, round_id: int = 0) -> dict:
    """Upload, aggregate (the cluster's candidate) and publish it: an inactive
    registry version. Returns the registry's version metadata."""
    upload_all(cluster_client, round_id=round_id)
    with publish_over_http(registry_client, monkeypatch):
        aggregate(cluster_client, "svd", retain=False)
        resp = cluster_client.post(f"/adapters/{CLUSTER_ID}/publish",
                                   json={"registry_url": "http://r", "adapter_name": REGISTRY_NAME,
                                         "round": round_id + 1})
    assert resp.status_code == 201, resp.text
    return resp.json()["version"]


def test_seam_c1_materializes_a_loadable_peft_directory(
        cluster_client, registry_client, edge_registry, monkeypatch, tmp_path):
    v1 = publish_candidate(cluster_client, registry_client, monkeypatch)
    # the evaluator pulls the candidate by number: nothing is active yet
    assert edge_registry.list_versions(REGISTRY_NAME)["active"] is None
    manifest = edge_registry.materialize(REGISTRY_NAME, tmp_path / "pulled", version=1)

    assert manifest["version"] == 1 and manifest["sha256_recorded"] == v1["sha256"]
    assert manifest["sha256_verified"] == manifest["sha256_recorded"]
    assert sorted(manifest["source_clients"]) == sorted(CLIENTS)
    assert manifest["aggregation"] == "svd_exact"

    out = tmp_path / "pulled"
    assert (out / "adapter_config.json").exists()
    assert (out / "adapter_model.safetensors").exists()
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["r"] == RANK
    assert cfg["base_model_name_or_path"] == BASE_MODEL
    prov = manifest["adapter_config_provenance"]
    assert prov["r"].startswith("registry metadata")
    assert prov["target_modules"].startswith("registry metadata")
    assert "producer-embedded" in prov["base_model_name_or_path"]

    sd, _cfg = wire.load_peft_dir(out)
    assert len(sd) == 2 * len(TARGET_MODULES) * NUM_LAYERS
    assert all(v.dtype == np.float32 for v in sd.values())


def test_seam_c1_rejects_a_corrupted_payload(
        cluster_client, registry_client, edge_registry, monkeypatch, tmp_path):
    """A checksum mismatch must fail loudly, not produce a corrupt adapter."""
    publish_candidate(cluster_client, registry_client, monkeypatch)
    original = edge_registry.download

    def flip_one_byte(name, version):
        payload = bytearray(original(name, version))
        payload[-1] ^= 0xFF
        return bytes(payload)

    monkeypatch.setattr(edge_registry, "download", flip_one_byte)
    with pytest.raises(ChecksumMismatch):
        edge_registry.materialize(REGISTRY_NAME, tmp_path / "corrupt", version=1)
    assert not (tmp_path / "corrupt").exists()


def test_materialized_cluster_adapter_composes_with_a_client_adapter(
        cluster_client, registry_client, edge_registry, monkeypatch, tmp_path):
    """C1 output goes straight into the existing D6 merge, contract check on."""
    torch = pytest.importorskip("torch")
    from edge.merge import (
        CONTRACT_HYPERPARAMS,
        compose,
        load_adapter,
        merge_max_error,
        module_prefixes,
        validate_compatibility,
    )

    publish_candidate(cluster_client, registry_client, monkeypatch)
    edge_registry.materialize(REGISTRY_NAME, tmp_path / "cluster", version=1)
    client_sd, client_cfg = make_client_adapter(seed=100)
    wire.save_peft_dir(tmp_path / "client", client_sd, client_cfg)

    cluster_sd, cluster_cfg = load_adapter(tmp_path / "cluster")
    client_sd_t, client_cfg_t = load_adapter(tmp_path / "client")
    # rank 4 here vs the contract's 16: check adapter-vs-adapter compatibility,
    # which is what composition actually requires.
    info = validate_compatibility([cluster_cfg, client_cfg_t], ["cluster", "client"],
                                  contract=None)
    assert info["scalings"] == {"cluster": 1.0, "client": 1.0}
    assert set(CONTRACT_HYPERPARAMS["target_modules"]) == set(cluster_cfg["target_modules"])

    parts = [(cluster_sd, cluster_cfg, 0.5), (client_sd_t, client_cfg_t, 1.0)]
    sd, cfg = compose(parts)
    assert cfg["r"] == 2 * RANK  # rank concatenation
    assert len(module_prefixes(sd)) == len(TARGET_MODULES) * NUM_LAYERS
    err = merge_max_error(parts, sd, cfg)
    assert err["max_rel_err"] < 1e-5, err
    assert torch.isfinite(next(iter(sd.values()))).all()


# --------------------------------------------------------------------------- #
# evaluation (CPU, no model: fixed predictions scored by the real metric)
# --------------------------------------------------------------------------- #
SAMPLE_FILE = '''\
import os
import sys


def resolve(path):
    return os.path.abspath(path)


def announce(name):
    print(f"hello {name}", file=sys.stderr)
'''


def test_completion_metric_produces_a_fillable_in_project_block(tmp_path):
    held = tmp_path / "held_out"
    held.mkdir()
    (held / "mod.py").write_text(SAMPLE_FILE, encoding="utf-8")
    examples = collect_examples(held, max_examples=10, stride=1)
    assert len(examples) >= 3

    perfect = score([ex.target for ex in examples], examples)
    assert perfect["edit_similarity"] == pytest.approx(1.0)
    assert perfect["exact_match"] == pytest.approx(1.0)

    wrong = score(["# nothing like the gold line" for _ in examples], examples)
    assert wrong["exact_match"] == 0.0
    assert wrong["edit_similarity"] < perfect["edit_similarity"]

    block = in_project_dict(perfect, perplexity=2.5)
    assert set(block) == {"edit_similarity", "exact_match", "perplexity", "n_examples"}
    assert block["n_examples"] == len(examples)


# --------------------------------------------------------------------------- #
# seam C2 — both D5 outcomes, on a candidate above an active v1
# --------------------------------------------------------------------------- #
def _guard(candidate_edit_similarity: float, baseline_edit_similarity: float, tmp_path):
    """A line-guard evidence file shaped like ``scripts/evaluate_candidate.py``'s output."""
    def side(es: float) -> dict:
        return {"benchmark": "LineCompletion", "edit_similarity": es, "exact_match": 0.1,
                "n_examples": 300}

    path = tmp_path / "web-v2.json"
    path.write_text(json.dumps({"clusters": {CLUSTER_ID: {
        "guard": [side(candidate_edit_similarity)],
        "baseline_guard": [side(baseline_edit_similarity)]}}}), encoding="utf-8")
    return resolve_guard(path)


def _metrics(edit_similarity: float):
    return {"edit_similarity": edit_similarity, "exact_match": 0.25,
            "perplexity": 3.0, "n_examples": 40}


def candidate_above_an_active_v1(cluster_client, registry_client, edge_registry,
                                 monkeypatch, tmp_path) -> dict:
    """Round 1: v1 published, promoted (against the bare base model) and
    served by the cluster. Round 2: v2 aggregated and published as the
    candidate. Returns v2's registry metadata."""
    publish_candidate(cluster_client, registry_client, monkeypatch, round_id=0)
    first = promote_candidate(
        edge_registry, REGISTRY_NAME, version=1, kind="cluster", cluster_id=CLUSTER_ID,
        in_project=_metrics(0.60), baseline_in_project=_metrics(0.50),
        guard=_guard(0.30, 0.30, tmp_path), noise_band=0.0)
    assert first["decision"]["action"] == "promote"
    decide_on_cluster(cluster_client, "promote")
    v2 = publish_candidate(cluster_client, registry_client, monkeypatch, round_id=1)
    assert v2["ref"]["version"] == 2
    assert edge_registry.list_versions(REGISTRY_NAME)["active"] == 1
    return v2


def test_seam_c2_promotes_when_both_halves_of_d5_hold(
        cluster_client, registry_client, edge_registry, monkeypatch, tmp_path):
    v2 = candidate_above_an_active_v1(cluster_client, registry_client, edge_registry,
                                      monkeypatch, tmp_path)
    guard = _guard(candidate_edit_similarity=0.30, baseline_edit_similarity=0.30, tmp_path=tmp_path)
    assert guard.available

    record = promote_candidate(
        edge_registry, REGISTRY_NAME, version=2, kind="cluster", cluster_id=CLUSTER_ID,
        in_project=_metrics(0.80), baseline_in_project=_metrics(0.70),
        guard=guard, noise_band=0.02)

    assert record["decision"]["action"] == "promote"
    assert record["decision"]["active_version_after"] == 2
    assert record["decision_is_authoritative"] is True
    assert record["registry_agrees_with_local_prediction"] is True
    assert "edit_similarity" in record["decision"]["reason"]
    assert edge_registry.get_active(REGISTRY_NAME)["ref"]["version"] == 2
    trail = edge_registry.list_promotions(REGISTRY_NAME)
    assert [(d["adapter"]["version"], d["action"]) for d in trail] == [(1, "promote"), (2, "promote")]
    # and the cluster serves it once told to
    assert decide_on_cluster(cluster_client, "promote")["active_sha256"] == v2["sha256"]


def test_seam_c2_rejects_when_the_in_project_metric_does_not_improve(
        cluster_client, registry_client, edge_registry, monkeypatch, tmp_path):
    v2 = candidate_above_an_active_v1(cluster_client, registry_client, edge_registry,
                                      monkeypatch, tmp_path)
    guard = _guard(candidate_edit_similarity=0.30, baseline_edit_similarity=0.30, tmp_path=tmp_path)
    served = cluster_client.get(f"/clusters/{CLUSTER_ID}/adapters/active").json()["sha256"]

    record = promote_candidate(
        edge_registry, REGISTRY_NAME, version=2, kind="cluster", cluster_id=CLUSTER_ID,
        in_project=_metrics(0.70), baseline_in_project=_metrics(0.75),
        guard=guard, noise_band=0.02)

    assert record["decision"]["action"] == "reject"
    assert record["decision"]["reason"].startswith("rejected: ")
    assert record["decision"]["active_version_after"] == 1
    assert edge_registry.get_active(REGISTRY_NAME)["ref"]["version"] == 1
    trail = edge_registry.list_promotions(REGISTRY_NAME)
    assert [(d["adapter"]["version"], d["action"]) for d in trail] == [(1, "promote"), (2, "reject")]
    # the cluster drops the candidate and keeps serving v1; v2 is never served
    assert decide_on_cluster(cluster_client, "reject")["active_sha256"] == served
    assert cluster_client.get(f"/clusters/{CLUSTER_ID}/adapters/active").json()["sha256"] == served
    assert served != v2["sha256"]


def test_seam_c2_rejects_when_the_guard_regresses(
        cluster_client, registry_client, edge_registry, monkeypatch, tmp_path):
    """The rule is two-sided: an in-project win does not buy a guard regression."""
    candidate_above_an_active_v1(cluster_client, registry_client, edge_registry,
                                 monkeypatch, tmp_path)
    guard = _guard(candidate_edit_similarity=0.20, baseline_edit_similarity=0.30, tmp_path=tmp_path)

    record = promote_candidate(
        edge_registry, REGISTRY_NAME, version=2, kind="cluster", cluster_id=CLUSTER_ID,
        in_project=_metrics(0.90), baseline_in_project=_metrics(0.70),
        guard=guard, noise_band=0.0)

    assert record["decision"]["action"] == "reject"
    assert "LineCompletion edit similarity dropped" in record["decision"]["reason"]
    assert edge_registry.get_active(REGISTRY_NAME)["ref"]["version"] == 1


def test_a_first_candidate_that_fails_leaves_nothing_active(
        cluster_client, registry_client, edge_registry, monkeypatch, tmp_path):
    """Rejecting the very first candidate serves nothing, rather than the reject."""
    publish_candidate(cluster_client, registry_client, monkeypatch)
    record = promote_candidate(
        edge_registry, REGISTRY_NAME, version=1, kind="cluster", cluster_id=CLUSTER_ID,
        in_project=_metrics(0.40), baseline_in_project=_metrics(0.50),
        guard=_guard(0.30, 0.30, tmp_path), noise_band=0.0)
    assert record["decision"]["action"] == "reject"
    assert record["decision"]["active_version_after"] == 0
    assert record["registry_agrees_with_local_prediction"] is True
    assert edge_registry.list_versions(REGISTRY_NAME)["active"] is None
    assert decide_on_cluster(cluster_client, "reject")["active_sha256"] is None
    assert cluster_client.get(f"/clusters/{CLUSTER_ID}/adapters/active").status_code == 404


def test_seam_c2_is_not_sent_when_the_guard_is_unavailable(
        cluster_client, registry_client, edge_registry, monkeypatch, tmp_path):
    """A missing guard is never a pass: the registry refuses it, so the edge does not send it."""
    candidate_above_an_active_v1(cluster_client, registry_client, edge_registry,
                                 monkeypatch, tmp_path)
    guard = resolve_guard()
    assert not guard.available

    record = promote_candidate(
        edge_registry, REGISTRY_NAME, version=2, kind="cluster", cluster_id=CLUSTER_ID,
        in_project=_metrics(0.99), baseline_in_project=_metrics(0.10),
        guard=guard, noise_band=0.0)

    assert record["decision"]["action"] == "not_sent"
    assert record["decision"]["registry_called"] is False
    assert record["decision"]["active_version_after"] == 1  # the pointer where it was
    assert record["decision_is_authoritative"] is False
    assert "PROVISIONAL" in record["authority_note"]
    assert "guard" in record["decision"]["reason"].lower()
    assert record["decision"]["predicted"]["action"] == "reject"
    assert [d["adapter"]["version"] for d in edge_registry.list_promotions(REGISTRY_NAME)] == [1]
    assert edge_registry.get_active(REGISTRY_NAME)["ref"]["version"] == 1


def test_a_first_candidate_not_sent_records_that_nothing_is_active(
        cluster_client, registry_client, edge_registry, monkeypatch, tmp_path):
    publish_candidate(cluster_client, registry_client, monkeypatch)
    record = promote_candidate(
        edge_registry, REGISTRY_NAME, version=1, kind="cluster", cluster_id=CLUSTER_ID,
        in_project=_metrics(0.99), baseline_in_project=_metrics(0.10),
        guard=resolve_guard(), noise_band=0.0)
    assert record["decision"]["action"] == "not_sent"
    assert record["decision"]["active_version_after"] is None  # nothing active, not "v0"


def test_seam_c2_marks_an_unmeasured_metric_instead_of_inventing_one(
        cluster_client, registry_client, edge_registry, monkeypatch, tmp_path):
    candidate_above_an_active_v1(cluster_client, registry_client, edge_registry,
                                 monkeypatch, tmp_path)
    record = promote_candidate(
        edge_registry, REGISTRY_NAME, version=2, kind="cluster", cluster_id=CLUSTER_ID,
        in_project=None, baseline_in_project=None,
        guard=_guard(0.30, 0.30, tmp_path), noise_band=0.0)

    assert record["in_project_measured"] is False
    assert record["eval_result_sent"]["in_project"]["n_examples"] == 0
    assert record["decision"]["action"] == "reject"
    assert record["decision_is_authoritative"] is False


# --------------------------------------------------------------------------- #
# the whole loop, once, in order
# --------------------------------------------------------------------------- #
def test_all_four_seams_in_one_pass(cluster_client, registry_client, edge_registry,
                                    monkeypatch, tmp_path):
    # A: upload -> aggregate (the cluster's candidate); B: publish (inactive)
    published = publish_candidate(cluster_client, registry_client, monkeypatch)
    versions = edge_registry.list_versions(REGISTRY_NAME)
    assert [v["ref"]["version"] for v in versions["versions"]] == [1]
    assert [v["aggregation"] for v in versions["versions"]] == ["svd_exact"]
    assert versions["active"] is None  # a candidate, not served
    assert cluster_client.get(f"/clusters/{CLUSTER_ID}/adapters/active").status_code == 404
    listing = cluster_client.get("/clusters").json()["clusters"][CLUSTER_ID]
    assert listing["candidate_sha256"] == published["sha256"]
    assert listing["active_sha256"] is None

    # C1 for the evaluator: the candidate, by number
    candidate = edge_registry.materialize(REGISTRY_NAME, tmp_path / "v1", version=1)
    assert candidate["sha256_verified"] == published["sha256"]

    # evaluation: the candidate against the bare base model (nothing active)
    held = tmp_path / "held_out"
    held.mkdir()
    (held / "mod.py").write_text(SAMPLE_FILE, encoding="utf-8")
    examples = collect_examples(held, max_examples=10, stride=1)
    base_block = in_project_dict(
        score(["# baseline guess"] * len(examples), examples), perplexity=3.4)
    cand_block = in_project_dict(
        score([ex.target for ex in examples], examples), perplexity=3.1)
    assert cand_block["edit_similarity"] > base_block["edit_similarity"]

    # C2: the registry promotes, then the cluster promotes the same bytes
    record = promote_candidate(
        edge_registry, REGISTRY_NAME, version=1, kind="cluster", cluster_id=CLUSTER_ID,
        in_project=cand_block, baseline_in_project=base_block,
        guard=_guard(0.30, 0.30, tmp_path), noise_band=0.0)
    assert record["decision"]["action"] == "promote"
    assert edge_registry.get_active(REGISTRY_NAME)["ref"]["version"] == 1
    assert len(edge_registry.list_promotions(REGISTRY_NAME)) == 1
    promoted = cluster_client.post(f"/clusters/{CLUSTER_ID}/adapters/candidate/promote",
                                   json={"sha256": published["sha256"]})
    assert promoted.status_code == 200, promoted.text

    # pull: what the cluster and the registry now serve is the promoted candidate
    served = cluster_client.get(f"/clusters/{CLUSTER_ID}/adapters/active").json()
    assert served["sha256"] == published["sha256"]
    pulled = edge_registry.materialize(REGISTRY_NAME, tmp_path / "pulled")
    assert pulled["version"] == 1 and pulled["sha256_verified"] == published["sha256"]

# --------------------------------------------------------------------------- #
# Review Focus 5: a normal round of six clients, real-size uploads (192 tensors
# each), through the security middleware of the real apps at its real limits
# --------------------------------------------------------------------------- #
SIX_CLIENTS = {
    "web": {"client-flask": 80, "client-requests": 52, "client-werkzeug": 170},
    "scientific": {"client-numpy": 292, "client-pandas": 335, "client-scikit-learn": 335},
}


def _full_size_adapter(seed: int, dim: int = 64):
    """The layout of a real client adapter: 24 layers x q/k/v/o x (A, B) = 192
    tensors at rank 16. The hidden width is 64 instead of 2048: the middleware
    counts requests and bytes, not numbers, and the real byte size is checked
    against the limit separately."""
    rng = np.random.default_rng(seed)
    sd = {}
    for layer in range(24):
        for module in wire.TARGET_MODULES:
            sd[wire.peft_key(layer, module, "lora_A")] = (
                rng.standard_normal((16, dim)) * 0.05).astype(np.float32)
            sd[wire.peft_key(layer, module, "lora_B")] = (
                rng.standard_normal((dim, 16)) * 0.05).astype(np.float32)
    return sd, {"r": 16, "lora_alpha": 16, "target_modules": list(wire.TARGET_MODULES)}


def test_a_six_client_round_of_192_tensor_uploads_passes_the_middleware(
        cluster_client, registry_client, monkeypatch):
    import cluster.server as cluster_server
    import registry.app as registry_app
    from security.middleware import RateLimitConfig, rate_limit_from_env

    assert len(wire.TARGET_MODULES) == 4
    limits = rate_limit_from_env({})
    assert (limits.requests_per_minute, limits.burst_size) == (600, 60)
    cluster_server.rate_limiter.reconfigure(limits)      # the real, default limits
    registry_app.rate_limiter.reconfigure(rate_limit_from_env({}))
    statuses = []
    try:
        for cluster_id, members in SIX_CLIENTS.items():
            r = cluster_client.put(f"/clusters/{cluster_id}/members",
                                   json={"client_ids": list(members)})
            statuses.append(r.status_code)
            assert r.status_code == 200, r.text
        seed = 0
        for cluster_id, members in SIX_CLIENTS.items():
            for client_id, examples in members.items():
                seed += 1
                sd, cfg = _full_size_adapter(seed)
                payload = wire.upload_payload(sd, cfg, client_id=client_id, cluster_id=cluster_id,
                                              round_id=0, num_examples=examples, seed=0)
                assert len(payload["tensors"]) == 192
                r = cluster_client.post(f"/clusters/{cluster_id}/uploads", json=payload)
                statuses.append(r.status_code)
                assert r.status_code == 201, (client_id, r.status_code, r.text[:300])
                # what the laptops' pages do meanwhile: poll the cluster
                statuses += [cluster_client.get("/clusters").status_code for _ in range(3)]
        for cluster_id in SIX_CLIENTS:
            r = cluster_client.post(f"/clusters/{cluster_id}/aggregate",
                                    json={"aggregation": "svd", "include_manifest": True})
            statuses.append(r.status_code)
            assert r.status_code == 200, r.text[:300]
            assert r.json()["num_clients"] == 3
            with publish_over_http(registry_client, monkeypatch):
                r = cluster_client.post(
                    f"/adapters/{cluster_id}/publish",
                    json={"registry_url": "http://registry:8004",
                          "adapter_name": f"cluster-{cluster_id}"})
            statuses.append(r.status_code)
            assert r.status_code == 201, r.text[:300]
            # the panel's poll of the registry
            for path in (f"/adapters/cluster-{cluster_id}/versions", "/sync/pending",
                         "/privacy/ledger"):
                statuses.append(registry_client.get(path).status_code)
    finally:
        cluster_server.rate_limiter.reconfigure(RateLimitConfig(enabled=False))
        registry_app.rate_limiter.reconfigure(RateLimitConfig(enabled=False))
    assert 413 not in statuses and 429 not in statuses
    assert set(statuses) <= {200, 201}
    assert registry_client.get("/adapters").json()["adapters"] == ["cluster-scientific",
                                                                    "cluster-web"]


def test_a_real_size_upload_is_far_below_every_body_limit():
    import cluster.server as cluster_server
    import registry.app as registry_app

    as_json = 192 * 16 * 2048 * 4 * 4 // 3 + 192 * 200     # 192 fp32 tensors, base64, in JSON
    as_safetensors = 192 * 16 * 2048 * 4 + 64 * 1024       # what the cluster publishes
    assert as_json < cluster_server.MAX_BODY_BYTES / 10
    assert as_safetensors < registry_app.MAX_BODY_BYTES / 10
