"""FedProx's server step when the edges upload D3 residuals.

An edge trains its own adapter R_k on top of the frozen ``base + alpha * C_N``,
C_N being the merged adapter it pulled from this cluster. FedProx averages the
clients' whole models, so the next merged adapter must satisfy

    alpha * C_{N+1} = alpha * C_N + sum_k w_k R_k    (w normalized)
    C_{N+1}         = C_N + (1/alpha) * sum_k w_k R_k

re-factorized to rank r through the same exact low-rank SVD as every other
aggregate (D2). Such an upload names the merged adapter it trained on by its
sha256, so one trained on anything but the adapter the cluster serves now is
refused, never folded in.

Uploads that declare no base are full adapters (the classic case) and are
still plain-averaged; one round cannot mix the two.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cluster import server
from cluster.aggregation import accumulate_svd_lowrank, aggregate_svd_lowrank
from tests.conftest import trained_adapter
from tests.http_helpers import create_cluster, reset_server_state, restore_server_state, upload_body

client = TestClient(server.app)
ALPHA = 0.5


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    yield
    restore_server_state(saved)


def _dense(adapter, module, layer=0):
    return adapter.delta_w(module, layer).astype(np.float64)


def _best_rank(mat, r):
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    return (u[:, :r] * s[:r]) @ vt[:r]


# --------------------------------------------------------------------------- #
# the mathematics
# --------------------------------------------------------------------------- #
def test_accumulate_is_start_plus_weighted_residuals_over_alpha():
    start, r1, r2 = trained_adapter(1), trained_adapter(2), trained_adapter(3)
    # rank 32 on a 32x32 module: no truncation, so the identity is exact
    merged, _ = accumulate_svd_lowrank(start, [r1, r2], [1.0, 3.0], alpha=ALPHA, rank=32)
    for m in start.target_modules:
        want = _dense(start, m) + (0.25 * _dense(r1, m) + 0.75 * _dense(r2, m)) / ALPHA
        assert np.allclose(_dense(merged, m), want, atol=1e-5)


def test_accumulate_truncates_like_every_other_aggregate():
    start, r1 = trained_adapter(1), trained_adapter(2)
    merged, errors = accumulate_svd_lowrank(start, [r1], [5.0], alpha=ALPHA, rank=16)
    m = start.target_modules[0]
    want = _best_rank(_dense(start, m) + _dense(r1, m) / ALPHA, 16)
    assert np.allclose(_dense(merged, m), want, atol=1e-4)
    assert merged.rank == 16 and all(0.0 <= e < 1.0 for e in errors.values())


def test_accumulate_refuses_a_non_positive_alpha():
    with pytest.raises(ValueError, match="alpha"):
        accumulate_svd_lowrank(trained_adapter(1), [trained_adapter(2)], [1.0], alpha=0.0)


# --------------------------------------------------------------------------- #
# over HTTP
# --------------------------------------------------------------------------- #
def _round0(cid="web", members=("a", "b")):
    create_cluster(client, cid, members={cid: list(members)})
    for i, c in enumerate(members):
        r = client.post(f"/clusters/{cid}/uploads",
                        json=upload_body(c, trained_adapter(10 + i), n=10 * (i + 1)))
        assert r.status_code == 201, r.text
    r = client.post(f"/clusters/{cid}/aggregate", json={})
    assert r.status_code == 200, r.text
    return r.json()


def _based(c, adapter, active, round_id=1, n=10, alpha=ALPHA, **extra):
    return upload_body(c, adapter, round_id=round_id, n=n, base_cluster_id=active["cluster_id"],
                       base_round_id=active["round_id"], base_sha256=active["sha256"],
                       composition_alpha=alpha, **extra)


def test_the_merged_adapter_carries_its_own_digest():
    active = _round0()
    blob = client.get("/adapters/web/download").content
    assert active["sha256"] == hashlib.sha256(blob).hexdigest()
    listing = client.get("/clusters").json()["clusters"]["web"]
    assert listing["active_sha256"] == active["sha256"]
    manifest = client.get("/clusters/web/aggregate/manifest").json()
    assert manifest["update_rule"] == "average"


def test_a_round_of_residuals_builds_on_the_merged_adapter():
    active = _round0()
    start = server.adapter_from_broadcast(server._clusters["web"].active)
    r_a, r_b = trained_adapter(20), trained_adapter(21)
    assert client.post("/clusters/web/uploads", json=_based("a", r_a, active, n=10)).status_code == 201
    assert client.post("/clusters/web/uploads", json=_based("b", r_b, active, n=30)).status_code == 201
    r = client.post("/clusters/web/aggregate", json={})
    assert r.status_code == 200, r.text
    merged = server.adapter_from_broadcast(server._clusters["web"].active)
    for m in start.target_modules:
        want = _best_rank(_dense(start, m) + (0.25 * _dense(r_a, m) + 0.75 * _dense(r_b, m)) / ALPHA,
                          16)
        assert np.allclose(_dense(merged, m), want, atol=1e-4)
    manifest = client.get("/clusters/web/aggregate/manifest").json()
    assert manifest["update_rule"] == "fedprox_cumulative"
    assert manifest["composition_alpha"] == ALPHA
    assert manifest["base_sha256"] == active["sha256"]
    assert manifest["base_round_id"] == 0
    # the new merged adapter is a new digest, and its header says how it was built
    assert r.json()["sha256"] != active["sha256"]
    assert r.json()["update_rule"] == "fedprox_cumulative"
    # D4 evidence: under D3 a client's update IS its residual (start None)
    assert all(start_ is None for _, start_ in server._clusters["web"].last_round.values())


def test_plain_round_averages_like_before():
    _round0()
    a, b = trained_adapter(30), trained_adapter(31)
    client.post("/clusters/web/uploads", json=upload_body("a", a, round_id=1, n=10))
    client.post("/clusters/web/uploads", json=upload_body("b", b, round_id=1, n=10))
    r = client.post("/clusters/web/aggregate", json={})
    merged = server.adapter_from_broadcast(server._clusters["web"].active)
    ref, _ = aggregate_svd_lowrank([a, b], [10.0, 10.0], rank=16)
    m = a.target_modules[0]
    assert np.allclose(_dense(merged, m), _dense(ref, m), atol=1e-5)
    assert r.json()["update_rule"] == "average"


def test_residual_trained_on_an_old_merge_is_refused():
    active = _round0()
    stale = dict(active, sha256="0" * 64)
    r = client.post("/clusters/web/uploads", json=_based("a", trained_adapter(5), stale))
    assert r.status_code == 409 and "stale_adapter" in r.text


def test_residual_before_any_merge_is_refused():
    create_cluster(client, "web", members={"web": ["a"]})
    fake = {"cluster_id": "web", "round_id": 0, "sha256": "1" * 64}
    r = client.post("/clusters/web/uploads", json=_based("a", trained_adapter(5), fake, round_id=0))
    assert r.status_code == 409 and "no merged adapter" in r.text


def test_a_round_cannot_mix_residuals_and_full_adapters():
    active = _round0()
    assert client.post("/clusters/web/uploads",
                       json=_based("a", trained_adapter(5), active)).status_code == 201
    r = client.post("/clusters/web/uploads", json=upload_body("b", trained_adapter(6), round_id=1))
    assert r.status_code == 422 and "base" in r.text


def test_a_round_has_one_composition_alpha():
    active = _round0()
    assert client.post("/clusters/web/uploads",
                       json=_based("a", trained_adapter(5), active)).status_code == 201
    r = client.post("/clusters/web/uploads", json=_based("b", trained_adapter(6), active, alpha=0.25))
    assert r.status_code == 422 and "composition_alpha" in r.text


def test_alpha_and_digest_come_together():
    active = _round0()
    body = upload_body("a", trained_adapter(5), round_id=1, composition_alpha=ALPHA)
    assert client.post("/clusters/web/uploads", json=body).status_code == 422
    body = upload_body("a", trained_adapter(5), round_id=1, base_sha256=active["sha256"])
    assert client.post("/clusters/web/uploads", json=body).status_code == 422


@pytest.mark.parametrize("override", [{"aggregation": "naive"}, {"exact_lowrank": False},
                                      {"retain_uploads": True}])
def test_residual_rounds_only_take_the_cumulative_path(override):
    active = _round0()
    client.post("/clusters/web/uploads", json=_based("a", trained_adapter(5), active))
    r = client.post("/clusters/web/aggregate", json=override)
    assert r.status_code == 422
    assert server._clusters["web"].uploads  # buffer kept for a correct retry


def test_epsilon_composes_across_the_rounds_it_builds_on():
    """The new adapter is a function of the one under it: by basic sequential
    composition its epsilon is that adapter's plus this round's (D7)."""
    create_cluster(client, "web", members={"web": ["a", "b"]})
    for c, eps, seed in (("a", 2.0, 10), ("b", 3.0, 11)):
        body = upload_body(c, trained_adapter(seed), privacy={"epsilon": eps})
        assert client.post("/clusters/web/uploads", json=body).status_code == 201
    active = client.post("/clusters/web/aggregate", json={}).json()
    assert active["epsilon"] == 3.0
    for c, eps, seed in (("a", 4.0, 20), ("b", 1.0, 21)):
        body = _based(c, trained_adapter(seed), active, privacy={"epsilon": eps})
        assert client.post("/clusters/web/uploads", json=body).status_code == 201
    merged = client.post("/clusters/web/aggregate", json={}).json()
    assert merged["epsilon"] == pytest.approx(7.0)
    manifest = client.get("/clusters/web/aggregate/manifest").json()
    assert manifest["epsilon_this_round"] == 4.0


def test_epsilon_is_unknown_when_the_base_had_none():
    active = _round0()  # no epsilon reported in round 0
    body = _based("a", trained_adapter(20), active, privacy={"epsilon": 1.0})
    assert client.post("/clusters/web/uploads", json=body).status_code == 201
    assert client.post("/clusters/web/aggregate", json={}).json()["epsilon"] is None
