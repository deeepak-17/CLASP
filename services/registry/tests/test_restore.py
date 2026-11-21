"""Operator restore (rollback drill) + the D11 restore-to-previous <= 10 s NFR."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from safetensors.numpy import save

from .conftest import make_safetensors

#: D11: registry restore-to-previous must complete within 10 s.
RESTORE_NFR_SECONDS = 10.0


def _upload(client, name, blob=None, kind="client", active=True):
    r = client.post(
        f"/adapters/{name}/versions",
        files={"file": (f"{name}.safetensors", blob or make_safetensors(), "application/octet-stream")},
        data={"meta": json.dumps({"kind": kind, "set_active": active})},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _promote(client, name, version, *, good=True):
    from .test_promote_api import _promote_body

    r = client.post(f"/adapters/{name}/promote", json=_promote_body(name, version, good=good))
    assert r.status_code == 200, r.text
    return r.json()


def _live_history(client, name, n=2):
    """n versions, each uploaded then promoted, so every one was live."""
    for v in range(1, n + 1):
        _upload(client, name, active=False)
        _promote(client, name, v)


def test_restore_defaults_to_previous_version(client):
    _live_history(client, "flask")
    r = client.post("/adapters/flask/restore", json={"reason": "bad adapter drill"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "rollback"
    assert body["active_version_after"] == 1
    assert body["adapter"]["version"] == 2
    assert "operator restore" in body["reason"] and "bad adapter drill" in body["reason"]
    assert client.get("/adapters/flask/active").json()["ref"]["version"] == 1


def test_restore_to_explicit_version_and_forward(client):
    for _ in range(3):
        _upload(client, "flask")
    assert client.post("/adapters/flask/restore",
                       json={"to_version": 1, "reason": "x"}).json()["active_version_after"] == 1
    fwd = client.post("/adapters/flask/restore", json={"to_version": 3, "reason": "fixed"})
    assert fwd.json()["action"] == "promote"
    assert "not a D5 decision" in fwd.json()["reason"]
    assert client.get("/adapters/flask/active").json()["ref"]["version"] == 3


def test_restore_is_recorded_in_the_audit_trail(client):
    _live_history(client, "flask")
    client.post("/adapters/flask/restore", json={"reason": "drill"})
    decisions = client.get("/adapters/flask/promotions").json()["decisions"]
    assert len(decisions) == 3 and decisions[-1]["active_version_after"] == 1


@pytest.mark.parametrize("body,status", [
    ({}, 422),                                    # reason is required
    ({"reason": ""}, 422),
    ({"reason": "x", "to_version": "two"}, 422),
    ({"reason": "x", "to_version": 9}, 404),      # unknown version
    ({"reason": "x", "to_version": 2}, 409),      # already active
])
def test_restore_validation(client, body, status):
    _upload(client, "flask")
    _upload(client, "flask")
    assert client.post("/adapters/flask/restore", json=body).status_code == status


def test_restore_with_no_earlier_live_version_is_409(client):
    _live_history(client, "flask", n=1)
    r = client.post("/adapters/flask/restore", json={"reason": "x"})
    assert r.status_code == 409
    assert r.json()["detail"] == "no earlier live version of flask"


def test_default_restore_skips_a_rejected_candidate(client):
    _upload(client, "flask", active=False)
    _promote(client, "flask", 1)
    _upload(client, "flask", active=False)
    assert _promote(client, "flask", 2, good=False)["action"] == "reject"
    _upload(client, "flask", active=False)
    _promote(client, "flask", 3)
    r = client.post("/adapters/flask/restore", json={"reason": "drill"})
    assert r.status_code == 200, r.text
    assert r.json()["active_version_after"] == 1
    assert client.get("/adapters/flask/active").json()["ref"]["version"] == 1


def test_explicit_restore_to_a_rejected_version_is_409(client):
    _upload(client, "flask", active=False)
    _promote(client, "flask", 1)
    _upload(client, "flask", active=False)
    _promote(client, "flask", 2, good=False)
    _upload(client, "flask", active=False)
    _promote(client, "flask", 3)
    r = client.post("/adapters/flask/restore", json={"reason": "x", "to_version": 2})
    assert r.status_code == 409
    assert r.json()["detail"] == "v2 was rejected and cannot be restored"
    assert client.get("/adapters/flask/active").json()["ref"]["version"] == 3


def test_a_restore_never_puts_an_unevaluated_candidate_live(client):
    """D5: a candidate goes live only through /promote. A forward restore
    re-activates a version that was live before; it never skips the rule."""
    _upload(client, "flask", active=False)
    _promote(client, "flask", 1)
    _upload(client, "flask", active=False)  # v2: never decided, never live
    r = client.post("/adapters/flask/restore", json={"reason": "operator choice", "to_version": 2})
    assert r.status_code == 409
    assert "never live" in r.json()["detail"] and "/promote" in r.json()["detail"]
    listing = client.get("/adapters/flask/versions").json()
    assert listing["active"] == 1 and listing["candidate"] == 2
    assert len(client.get("/adapters/flask/promotions").json()["decisions"]) == 1


def test_a_forward_restore_to_a_version_that_was_live_is_allowed(client):
    _live_history(client, "flask", n=2)
    client.post("/adapters/flask/restore", json={"reason": "drill"})  # back to v1
    fwd = client.post("/adapters/flask/restore", json={"reason": "fixed", "to_version": 2})
    assert fwd.status_code == 200, fwd.text
    assert fwd.json()["action"] == "promote" and "not a D5 decision" in fwd.json()["reason"]


@pytest.fixture
def as_cn(client):
    """Send requests as a certificate name (the x-test-cn header), as test_authz does."""
    import registry.app as appmod

    appmod.configure_identity(lambda request: request.headers.get("x-test-cn"))
    yield lambda name: {"x-test-cn": name}
    appmod.configure_identity(None)


@pytest.mark.parametrize("cn, status", [("demo-ui", 403), ("clasp-admin", 200)])
def test_only_an_admin_restores_back_to_a_version_that_was_never_live(client, as_cn, cn, status):
    """Backwards, a never-live version is an undecided candidate or a version
    from an older volume that went live without a record. Only an admin may
    name one: for the evaluator's certificate it would be the same bypass."""
    _upload(client, "flask", active=False)
    _promote(client, "flask", 1)
    _upload(client, "flask", active=False)  # v2: never decided
    _upload(client, "flask", active=False)
    _promote(client, "flask", 3)
    r = client.post("/adapters/flask/restore", json={"reason": "x", "to_version": 2},
                    headers=as_cn(cn))
    assert r.status_code == status, r.text
    expected = 2 if status == 200 else 3
    assert client.get("/adapters/flask/active").json()["ref"]["version"] == expected


def test_the_evaluator_may_restore_back_to_a_live_version(client, as_cn):
    _live_history(client, "flask", n=2)
    r = client.post("/adapters/flask/restore", json={"reason": "drill"}, headers=as_cn("demo-ui"))
    assert r.status_code == 200, r.text
    assert r.json()["active_version_after"] == 1


def test_an_earlier_operator_restore_target_counts_as_live(client):
    _upload(client, "flask", active=False)
    _promote(client, "flask", 1)
    _upload(client, "flask", active=False)  # v2 undecided, made live by the operator
    _upload(client, "flask", active=False)
    _promote(client, "flask", 3)
    assert client.post("/adapters/flask/restore",
                       json={"reason": "go", "to_version": 2}).status_code == 200
    # active v2; v1 was promoted, so the default goes there
    assert client.post("/adapters/flask/restore", json={"reason": "d"}).json()["active_version_after"] == 1
    # now active v1 -> forward to 3, then default must land on v2 (an operator target), not skip it
    client.post("/adapters/flask/restore", json={"reason": "fwd", "to_version": 3})
    assert client.post("/adapters/flask/restore", json={"reason": "d"}).json()["active_version_after"] == 2


def test_versions_live_without_a_decision_record_are_not_default_targets(client):
    _upload(client, "flask")  # set_active upload, no decision: an older-volume version
    _upload(client, "flask")
    r = client.post("/adapters/flask/restore", json={"reason": "x"})
    assert r.status_code == 409
    assert client.post("/adapters/flask/restore",
                       json={"reason": "x", "to_version": 1}).status_code == 200


def test_restore_unknown_adapter_404(client):
    assert client.post("/adapters/ghost/restore", json={"reason": "x"}).status_code == 404


def _realistic_adapter(seed: int) -> bytes:
    """~24 MB: 24 layers x q/k/v/o, rank 16, hidden 2048 — a 1.3B-class adapter."""
    rng = np.random.default_rng(seed)
    sd = {}
    for layer in range(24):
        for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key = f"base_model.model.model.layers.{layer}.self_attn.{m}"
            sd[f"{key}.lora_A.weight"] = rng.standard_normal((16, 2048), dtype=np.float32)
            sd[f"{key}.lora_B.weight"] = rng.standard_normal((2048, 16), dtype=np.float32)
    return save(sd)


def test_restore_nfr_under_10_seconds_on_a_real_sized_adapter(client):
    """D11: restore-to-previous, measured as the edge sees it — repoint the
    pointer, then fetch the active metadata and the full payload back."""
    good, bad = _realistic_adapter(1), _realistic_adapter(2)
    _upload(client, "flask", good, active=False)
    _promote(client, "flask", 1)
    _upload(client, "flask", bad, active=False)
    _promote(client, "flask", 2)

    start = time.perf_counter()
    client.post("/adapters/flask/restore", json={"reason": "nfr"}).raise_for_status()
    active = client.get("/adapters/flask/active").json()
    payload = client.get(f"/adapters/flask/versions/{active['ref']['version']}/file").content
    elapsed = time.perf_counter() - start

    assert payload == good
    assert elapsed < RESTORE_NFR_SECONDS, f"restore took {elapsed:.2f}s"


def test_versions_listing_names_the_candidate_and_the_refused(client):
    listing = lambda: client.get("/adapters/flask/versions").json()  # noqa: E731
    _upload(client, "flask", active=False)
    assert listing()["candidate"] == 1 and listing()["rejected"] == []
    _promote(client, "flask", 1)
    assert listing()["candidate"] is None
    _upload(client, "flask", active=False)
    _upload(client, "flask", active=False)
    assert listing()["candidate"] == 3  # the newest undecided above active
    _promote(client, "flask", 3, good=False)
    body = listing()
    assert body["candidate"] == 2 and body["rejected"] == [3]
    _promote(client, "flask", 2, good=False)
    assert listing()["candidate"] is None and listing()["rejected"] == [2, 3]
