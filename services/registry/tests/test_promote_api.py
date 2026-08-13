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


def _promote_body(name, version, *, good: bool):
    """A payload that clears both D5 checks (good) or fails both (not good)."""
    candidate_sim = 0.80 if good else 0.705
    candidate_pass1 = 0.30 if good else 0.10
    return {
        "eval": {
            "adapter": {"name": name, "version": version, "kind": "client"},
            "in_project": _in_project(edit_similarity=candidate_sim),
            "guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": candidate_pass1}}],
            "baseline_in_project": _in_project(edit_similarity=0.70),
            "baseline_noise_band": 0.02,
        },
        "baseline_guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.30}}],
    }


def test_promote_confirms_current_active(client):
    _upload(client, "django")
    _upload(client, "django")  # v2 now active
    r = client.post("/adapters/django/promote", json=_promote_body("django", 2, good=True))
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "promote"
    assert body["active_version_after"] == 2
    assert client.get("/adapters/django/active").json()["ref"]["version"] == 2

    log = client.get("/adapters/django/promotions").json()["decisions"]
    assert len(log) == 1
    assert log[0]["action"] == "promote"


def test_promote_rolls_back_to_previous_version(client):
    _upload(client, "flask")
    _upload(client, "flask")  # v2 active
    r = client.post("/adapters/flask/promote", json=_promote_body("flask", 2, good=False))
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "rollback"
    assert body["active_version_after"] == 1
    # the pointer actually moved, not just the response
    assert client.get("/adapters/flask/active").json()["ref"]["version"] == 1


def test_promote_rejects_stale_candidate(client):
    _upload(client, "numpy")
    _upload(client, "numpy")
    _upload(client, "numpy")  # v3 active
    r = client.post("/adapters/numpy/promote", json=_promote_body("numpy", 2, good=True))
    assert r.status_code == 409


def test_promote_with_no_fallback_version_is_a_clean_422(client):
    _upload(client, "pandas")  # only v1 exists, and it's active
    r = client.post("/adapters/pandas/promote", json=_promote_body("pandas", 1, good=False))
    assert r.status_code == 422
    # active pointer must be untouched by the failed attempt
    assert client.get("/adapters/pandas/active").json()["ref"]["version"] == 1


def test_promote_name_mismatch_422(client):
    _upload(client, "requests")
    body = _promote_body("requests", 1, good=True)
    r = client.post("/adapters/other-name/promote", json=body)
    assert r.status_code == 422


def test_promotions_404_for_unknown_adapter(client):
    assert client.get("/adapters/ghost/promotions").status_code == 404


# make_safetensors imported so conftest helper is reachable from tests
_ = make_safetensors
