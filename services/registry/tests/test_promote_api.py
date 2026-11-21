"""API tests for POST /adapters/{name}/promote and GET .../promotions (P4, W5 catch-up)."""
from __future__ import annotations

import json

from .conftest import make_safetensors


def _upload(client, name, blob=None, meta=None):
    return client.post(
        f"/adapters/{name}/versions",
        files={"file": (f"{name}.safetensors", blob or make_safetensors(), "application/octet-stream")},
        data={"meta": json.dumps(meta or {"kind": "client"})},
    )


def _in_project(edit_similarity=0.8, exact_match=0.5, perplexity=3.0, n=20):
    return {
        "edit_similarity": edit_similarity,
        "exact_match": exact_match,
        "perplexity": perplexity,
        "n_examples": n,
    }


def _line_guard(edit_similarity):
    return {"benchmark": "LineCompletion", "edit_similarity": edit_similarity,
            "exact_match": 0.3, "n_examples": 300, "pass_at_k": {}}


def _promote_body(name, version, *, good: bool):
    """A payload that clears both D5 checks (good) or fails both (not good)."""
    candidate_sim = 0.80 if good else 0.705
    candidate_guard = 0.60 if good else 0.40
    return {
        "eval": {
            "adapter": {"name": name, "version": version, "kind": "client"},
            "in_project": _in_project(edit_similarity=candidate_sim),
            "guard": [_line_guard(candidate_guard)],
            "baseline_in_project": _in_project(edit_similarity=0.70),
            "baseline_noise_band": 0.02,
        },
        "baseline_guard": [_line_guard(0.60)],
    }


def test_promote_confirms_current_active(client):
    _upload(client, "django")
    _upload(client, "django")
    r = client.post("/adapters/django/promote", json=_promote_body("django", 2, good=True))
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "promote"
    assert body["active_version_after"] == 2
    assert client.get("/adapters/django/active").json()["ref"]["version"] == 2

    log = client.get("/adapters/django/promotions").json()["decisions"]
    assert len(log) == 1
    assert log[0]["action"] == "promote"


def test_promote_rejects_a_stale_candidate_older_than_active(client):
    for _ in range(3):
        _upload(client, "numpy")
    client.post("/adapters/numpy/promote", json=_promote_body("numpy", 3, good=True))
    r = client.post("/adapters/numpy/promote", json=_promote_body("numpy", 2, good=True))
    assert r.status_code == 409


def test_promote_with_nothing_active_and_a_bad_eval_is_a_clean_reject(client):
    _upload(client, "pandas")  # only v1 exists, and it is not active
    r = client.post("/adapters/pandas/promote", json=_promote_body("pandas", 1, good=False))
    assert r.status_code == 200
    assert r.json()["action"] == "reject"
    assert client.get("/adapters/pandas/active").status_code == 404


def test_humaneval_only_body_is_422_and_records_nothing(client):
    _upload(client, "scipy", meta={"kind": "client", "set_active": True})
    _upload(client, "scipy")
    body = _promote_body("scipy", 2, good=True)
    body["eval"]["guard"] = [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.3}}]
    body["baseline_guard"] = [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.3}}]
    r = client.post("/adapters/scipy/promote", json=body)
    assert r.status_code == 422
    assert "LineCompletion guard metrics missing" in r.json()["detail"]
    assert "1.2.0" in r.json()["detail"]
    assert client.get("/adapters/scipy/promotions").json()["decisions"] == []
    assert client.get("/adapters/scipy/active").json()["ref"]["version"] == 1


def test_promote_name_mismatch_422(client):
    _upload(client, "requests")
    body = _promote_body("requests", 1, good=True)
    r = client.post("/adapters/other-name/promote", json=body)
    assert r.status_code == 422


def test_promotions_404_for_unknown_adapter(client):
    assert client.get("/adapters/ghost/promotions").status_code == 404


def test_promote_corrupt_pointer_is_a_clean_500(client):
    """L1: same guard as GET .../active — promote() must not 500 raw on a
    hand-corrupted active pointer."""
    import os
    from pathlib import Path

    import registry.app as appmod

    from .conftest import _drop_app_store, forget_database

    _upload(client, "corrupt")
    # pointers live in the database; a file-based volume's `active` file is
    # read by the import when the registry next starts on an empty database
    _drop_app_store(appmod)
    forget_database(os.environ["CLASP_REGISTRY_DATA"])
    active_path = Path(os.environ["CLASP_REGISTRY_DATA"]) / "adapters" / "corrupt" / "active"
    active_path.write_text("not-an-int")

    r = client.post("/adapters/corrupt/promote", json=_promote_body("corrupt", 1, good=True))
    assert r.status_code == 500
    assert "corrupt" in r.json()["detail"].lower()


# --- Task 5.1: uploads are candidates; only a promote makes a version active ---


def test_upload_does_not_activate(client):
    assert _upload(client, "cand").status_code == 201
    assert client.get("/adapters/cand/active").status_code == 404


def test_promote_makes_the_candidate_active(client):
    _upload(client, "cand")
    assert client.post("/adapters/cand/promote", json=_promote_body("cand", 1, good=True)).status_code == 200
    assert client.get("/adapters/cand/active").json()["ref"]["version"] == 1
    _upload(client, "cand")  # v2 is a candidate, v1 stays live
    assert client.get("/adapters/cand/active").json()["ref"]["version"] == 1
    r = client.post("/adapters/cand/promote", json=_promote_body("cand", 2, good=True))
    assert r.status_code == 200
    assert r.json()["action"] == "promote"
    assert client.get("/adapters/cand/active").json()["ref"]["version"] == 2


def test_reject_leaves_the_pointer_and_records_reject(client):
    _upload(client, "cand")
    client.post("/adapters/cand/promote", json=_promote_body("cand", 1, good=True))
    _upload(client, "cand")
    r = client.post("/adapters/cand/promote", json=_promote_body("cand", 2, good=False))
    assert r.status_code == 200
    assert r.json()["action"] == "reject"
    assert r.json()["active_version_after"] == 1
    assert client.get("/adapters/cand/active").json()["ref"]["version"] == 1
    decisions = client.get("/adapters/cand/promotions").json()["decisions"]
    assert decisions[-1]["action"] == "reject"


def test_first_candidate_rejected_leaves_nothing_active(client):
    _upload(client, "cand")
    r = client.post("/adapters/cand/promote", json=_promote_body("cand", 1, good=False))
    assert r.status_code == 200
    assert r.json()["action"] == "reject"
    assert r.json()["active_version_after"] == 0
    assert client.get("/adapters/cand/active").status_code == 404


def test_first_candidate_can_be_promoted_with_no_previous_version(client):
    _upload(client, "cand")
    r = client.post("/adapters/cand/promote", json=_promote_body("cand", 1, good=True))
    assert r.status_code == 200
    assert r.json()["action"] == "promote"
    assert r.json()["active_version_after"] == 1


def test_promoting_the_active_or_an_older_version_is_409(client):
    for _ in range(2):
        _upload(client, "cand")
    client.post("/adapters/cand/promote", json=_promote_body("cand", 2, good=True))
    active = client.post("/adapters/cand/promote", json=_promote_body("cand", 2, good=True))
    assert active.status_code == 409
    assert active.json()["detail"] == "v2 is not a candidate: active is v2"
    older = client.post("/adapters/cand/promote", json=_promote_body("cand", 1, good=True))
    assert older.status_code == 409
    assert older.json()["detail"] == "v1 is not a candidate: active is v2"


def test_deciding_a_version_twice_is_409(client):
    _upload(client, "cand")
    client.post("/adapters/cand/promote", json=_promote_body("cand", 1, good=False))
    r = client.post("/adapters/cand/promote", json=_promote_body("cand", 1, good=True))
    assert r.status_code == 409
    assert r.json()["detail"] == "v1 was already decided"
    assert client.get("/adapters/cand/active").status_code == 404


def test_promoting_a_version_that_does_not_exist_is_404(client):
    _upload(client, "cand")
    assert client.post("/adapters/cand/promote", json=_promote_body("cand", 5, good=True)).status_code == 404


def test_an_operator_restore_does_not_block_deciding_that_version(client):
    _upload(client, "cand")
    client.post("/adapters/cand/promote", json=_promote_body("cand", 1, good=True))
    # v2: live without a decision (a set_active upload, as on an older volume)
    _upload(client, "cand", meta={"kind": "client", "set_active": True})
    # the operator moves back to v1; that restore records a decision naming v2
    assert client.post("/adapters/cand/restore", json={"reason": "back", "to_version": 1}).status_code == 200
    assert client.get("/adapters/cand/promotions").json()["decisions"][-1]["adapter"]["version"] == 2
    r = client.post("/adapters/cand/promote", json=_promote_body("cand", 2, good=True))
    assert r.status_code == 200
    assert client.get("/adapters/cand/active").json()["ref"]["version"] == 2


def test_explicit_set_active_upload_is_still_honoured(client):
    _upload(client, "cand", meta={"kind": "client", "set_active": True})
    assert client.get("/adapters/cand/active").json()["ref"]["version"] == 1


# make_safetensors imported so conftest helper is reachable from tests
_ = make_safetensors
