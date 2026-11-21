"""Evaluation results on ``/promote`` are signed and tied to the candidate's bytes (U4).

The evaluator signs ``{adapter, version, candidate_sha256, eval, baseline_guard}``
with a shared secret (``CLASP_EVAL_HMAC_SECRET`` or the file
``CLASP_EVAL_HMAC_SECRET_FILE`` names). With the secret configured the registry
refuses a promote whose signature is missing or wrong (401) and records
nothing. A ``candidate_sha256`` that is not the stored version's is "evaluated
on different bytes" (409), signed or not. Without a secret the check is skipped
with a warning (the fall-back, C11).
"""
from __future__ import annotations

import logging

import pytest
from security.auth.hmac_signer import sign_evaluation

from .test_composite import make_adapter
from .test_composite_api import _save
from .test_promote_api import _promote_body

SECRET = "5e" * 32
NAME = "cluster-web"


@pytest.fixture
def candidate(client):
    """v1 active, v2 a candidate; returns (client, v2's sha256)."""
    _save(client, NAME, make_adapter(1), kind="cluster", cluster_id="web")
    v2 = _save(client, NAME, make_adapter(2), kind="cluster", cluster_id="web", set_active=False)
    return client, v2["sha256"]


def _body(sha, *, good=True, secret=SECRET, version=2):
    body = _promote_body(NAME, version, good=good)
    body["eval"]["adapter"].update(kind="cluster", cluster_id="web")
    body["candidate_sha256"] = sha
    if secret is not None:
        body["eval_signature"] = sign_evaluation(
            adapter=NAME, version=version, candidate_sha256=sha, eval_result=body["eval"],
            baseline_guard=body["baseline_guard"], secret=secret)
    return body


def _decisions(client):
    return client.get(f"/adapters/{NAME}/promotions").json()["decisions"]


def _active(client):
    return client.get(f"/adapters/{NAME}/active").json()["ref"]["version"]


@pytest.fixture
def signed(monkeypatch):
    monkeypatch.setenv("CLASP_EVAL_HMAC_SECRET", SECRET)


def test_a_valid_signature_records_the_decision(candidate, signed):
    client, sha = candidate
    r = client.post(f"/adapters/{NAME}/promote", json=_body(sha))
    assert r.status_code == 200, r.text
    assert r.json()["action"] == "promote" and _active(client) == 2
    assert len(_decisions(client)) == 1


def test_a_signed_reject_is_recorded_too(candidate, signed):
    client, sha = candidate
    r = client.post(f"/adapters/{NAME}/promote", json=_body(sha, good=False))
    assert r.status_code == 200 and r.json()["action"] == "reject"
    assert _active(client) == 1 and len(_decisions(client)) == 1


def test_one_changed_metric_is_401_and_no_decision(candidate, signed):
    client, sha = candidate
    body = _body(sha, good=False)                      # measured: a failing candidate
    body["eval"]["in_project"]["edit_similarity"] = 0.99  # ...edited on the way to the registry
    body["eval"]["guard"][0]["edit_similarity"] = 0.99
    r = client.post(f"/adapters/{NAME}/promote", json=body)
    assert r.status_code == 401, r.text
    assert "signature" in r.json()["detail"]
    assert _decisions(client) == [] and _active(client) == 1
    # the genuine result can still be submitted afterwards
    assert client.post(f"/adapters/{NAME}/promote", json=_body(sha, good=False)).status_code == 200


def test_a_changed_baseline_guard_is_401(candidate, signed):
    client, sha = candidate
    body = _body(sha)
    body["baseline_guard"][0]["edit_similarity"] = 0.0
    assert client.post(f"/adapters/{NAME}/promote", json=body).status_code == 401
    assert _decisions(client) == []


def test_a_missing_signature_with_a_secret_set_is_401(candidate, signed):
    client, sha = candidate
    r = client.post(f"/adapters/{NAME}/promote", json=_body(sha, secret=None))
    assert r.status_code == 401 and "eval_signature" in r.json()["detail"]
    assert _decisions(client) == []


def test_a_missing_candidate_sha_with_a_secret_set_is_401(candidate, signed):
    client, sha = candidate
    body = _body(sha)
    del body["candidate_sha256"]
    r = client.post(f"/adapters/{NAME}/promote", json=body)
    assert r.status_code == 401 and "candidate_sha256" in r.json()["detail"]
    assert _decisions(client) == []


def test_a_signature_made_with_another_secret_is_401(candidate, signed):
    client, sha = candidate
    r = client.post(f"/adapters/{NAME}/promote", json=_body(sha, secret="00" * 32))
    assert r.status_code == 401 and _decisions(client) == []


def test_the_sha_of_another_version_is_409_evaluated_on_different_bytes(candidate, signed):
    client, _ = candidate
    other = client.get(f"/adapters/{NAME}/versions/1").json()["sha256"]
    r = client.post(f"/adapters/{NAME}/promote", json=_body(other))  # correctly signed, wrong bytes
    assert r.status_code == 409, r.text
    assert "evaluation was run on different bytes" in r.json()["detail"]
    assert _decisions(client) == [] and _active(client) == 1


def test_without_a_secret_the_sha_is_still_checked(candidate):
    client, _ = candidate
    other = client.get(f"/adapters/{NAME}/versions/1").json()["sha256"]
    r = client.post(f"/adapters/{NAME}/promote", json=_body(other, secret=None))
    assert r.status_code == 409 and _decisions(client) == []


def test_no_secret_configured_is_accepted_with_a_warning(candidate, caplog):
    client, sha = candidate
    with caplog.at_level(logging.WARNING, logger="registry"):
        r = client.post(f"/adapters/{NAME}/promote", json=_body(sha, secret=None))
    assert r.status_code == 200 and r.json()["action"] == "promote"
    assert any("CLASP_EVAL_HMAC_SECRET" in rec.message for rec in caplog.records)


def test_no_secret_and_an_old_body_without_the_new_fields_still_works(candidate):
    """A panel that has not pulled this change, against a registry without a secret."""
    client, _ = candidate
    body = _promote_body(NAME, 2, good=True)
    body["eval"]["adapter"].update(kind="cluster", cluster_id="web")
    assert client.post(f"/adapters/{NAME}/promote", json=body).status_code == 200


def test_the_secret_can_come_from_a_file(candidate, monkeypatch, tmp_path):
    client, sha = candidate
    key = tmp_path / "eval_hmac.key"
    key.write_text(SECRET + "\n", encoding="utf-8")
    monkeypatch.setenv("CLASP_EVAL_HMAC_SECRET_FILE", str(key))
    assert client.post(f"/adapters/{NAME}/promote",
                       json=_body(sha, secret=None)).status_code == 401
    assert client.post(f"/adapters/{NAME}/promote", json=_body(sha)).status_code == 200


def test_a_configured_secret_that_cannot_be_read_refuses_the_promote(candidate, monkeypatch, tmp_path):
    """Configured but broken is not "not configured": nothing is decided."""
    client, sha = candidate
    monkeypatch.setenv("CLASP_EVAL_HMAC_SECRET_FILE", str(tmp_path / "missing.key"))
    r = client.post(f"/adapters/{NAME}/promote", json=_body(sha))
    assert r.status_code == 500 and "CLASP_EVAL_HMAC_SECRET_FILE" in r.json()["detail"]
    assert _decisions(client) == []


def test_a_refusal_never_echoes_a_signature_or_the_secret(candidate, signed):
    client, sha = candidate
    body = _body(sha)
    body["eval"]["in_project"]["edit_similarity"] = 0.99
    r = client.post(f"/adapters/{NAME}/promote", json=body)
    assert r.status_code == 401
    assert SECRET not in r.text and body["eval_signature"] not in r.text
