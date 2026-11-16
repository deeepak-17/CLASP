"""Edge uploads are signed by the edge and verified before they are buffered (U4).

Each edge holds its own HMAC key (``edges/<client_id>/hmac.key`` from
``scripts/make_dev_certs.sh``); the cluster holds all of them in the JSON file
``CLASP_UPLOAD_HMAC_SECRETS`` names (client id -> key). The signature covers
every tensor's raw bytes and the whole upload body except ``tensors``,
``signature`` and ``timestamp``, so a flipped byte or a changed sample count is
refused (401 ``upload_signature_invalid``) and never buffered. A client with no
configured key is accepted with a warning (the fall-back, C11).
"""
from __future__ import annotations

import base64
import json
import logging

import pytest
from fastapi.testclient import TestClient
from security.auth.hmac_signer import sign_adapter_upload

from cluster import server
from tests.conftest import trained_adapter
from tests.http_helpers import (
    create_cluster,
    dp,
    reset_server_state,
    restore_server_state,
    upload_body,
)

client = TestClient(server.app)
KEY_A, KEY_B = "a1" * 32, "b2" * 32
UNSIGNED_KEYS = ("tensors", "signature", "timestamp")


def sign(body: dict, secret: str) -> str:
    """The edge's recipe (``edge.wire.upload_payload``), written out independently."""
    tensors = {t["name"]: base64.b64decode(t["data_b64"]) for t in body["tensors"]}
    metadata = {k: v for k, v in body.items() if k not in UNSIGNED_KEYS}
    return sign_adapter_upload(tensors, metadata, secret)


def _body(who="a", seed=1, **extra):
    return json.loads(json.dumps(upload_body(who, trained_adapter(seed), **extra)))


def _signed(who="a", seed=1, secret=KEY_A, **extra):
    body = _body(who, seed, **extra)
    body["signature"] = sign(body, secret)
    return body


def _post(body):
    return client.post("/clusters/web/uploads", json=body)


def _buffered():
    return sorted(server._clusters["web"].uploads)


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    saved = reset_server_state()
    create_cluster(client, "web", members={"web": ["a", "b", "c"]})
    secrets = tmp_path / "upload_hmac.json"
    secrets.write_text(json.dumps({"a": KEY_A, "b": KEY_B}), encoding="utf-8")
    monkeypatch.setenv("CLASP_UPLOAD_HMAC_SECRETS", str(secrets))
    yield secrets
    restore_server_state(saved)


def test_a_signed_upload_is_accepted():
    r = _post(_signed())
    assert r.status_code == 201, r.text
    assert _buffered() == ["a"]


def test_a_signed_dp_upload_with_its_privacy_block_is_accepted():
    assert _post(_signed(privacy=dp(3.9))).status_code == 201


def test_one_flipped_tensor_byte_is_401_and_not_buffered():
    body = _signed()
    raw = bytearray(base64.b64decode(body["tensors"][0]["data_b64"]))
    raw[5] ^= 0x01
    body["tensors"][0]["data_b64"] = base64.b64encode(bytes(raw)).decode("ascii")
    r = _post(body)
    assert r.status_code == 401, r.text
    assert r.json()["detail"]["error"] == "upload_signature_invalid"
    assert _buffered() == []


def test_a_changed_sample_count_is_401():
    body = _signed()
    body["num_examples"] = 10_000   # a heavier FedAvg weight than the edge signed
    r = _post(body)
    assert r.status_code == 401 and r.json()["detail"]["error"] == "upload_signature_invalid"
    assert _buffered() == []


def test_a_changed_privacy_block_is_401():
    body = _signed(privacy=dp(7.9))
    body["privacy"]["epsilon"] = body["privacy"]["epsilon_cumulative"] = 1.0
    assert _post(body).status_code == 401 and _buffered() == []


def test_an_unsigned_upload_from_a_client_that_has_a_secret_is_401():
    r = _post(_body())
    assert r.status_code == 401
    detail = r.json()["detail"]
    assert detail["error"] == "upload_signature_invalid" and "hmac.key" in detail["hint"]
    assert _buffered() == []


def test_a_signature_made_with_another_clients_key_is_401():
    assert _post(_signed("a", secret=KEY_B)).status_code == 401
    assert _buffered() == []


def test_a_client_without_a_secret_is_accepted_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="cluster"):
        r = _post(_body("c", 3))
    assert r.status_code == 201 and _buffered() == ["c"]
    assert any("'c'" in rec.getMessage() and "unsigned" in rec.getMessage()
               for rec in caplog.records)


def test_a_signature_from_a_client_without_a_secret_is_not_an_error():
    """An edge that signs before laptop A has its key: nothing to check it against."""
    assert _post(_signed("c", 3, secret="cc" * 32)).status_code == 201


def test_with_no_secrets_file_configured_every_upload_is_accepted(monkeypatch):
    monkeypatch.delenv("CLASP_UPLOAD_HMAC_SECRETS")
    assert _post(_body("a")).status_code == 201
    assert _post(_signed("b", 2, secret="00" * 32)).status_code == 201


@pytest.mark.parametrize("content", ["{not json", "[]", '{"a": 5}', '{"a": ""}'])
def test_a_secrets_file_that_cannot_be_used_refuses_uploads(_clean, content):
    """Configured but broken is not "not configured"."""
    _clean.write_text(content, encoding="utf-8")
    r = _post(_signed())
    assert r.status_code == 500 and "CLASP_UPLOAD_HMAC_SECRETS" in r.json()["detail"]
    assert _buffered() == []


def test_a_missing_secrets_file_refuses_uploads(_clean):
    _clean.unlink()
    assert _post(_signed()).status_code == 500 and _buffered() == []


def test_a_refused_upload_does_not_replace_the_one_already_buffered():
    assert _post(_signed(seed=1)).status_code == 201
    forged = _body(seed=2)
    forged["signature"] = "0" * 64
    assert _post(forged).status_code == 401
    assert _buffered() == ["a"]


def test_the_legacy_upload_route_checks_the_signature_too():
    create_cluster(client, "cluster-web", members={"cluster-web": ["a"]})
    ok = _signed(cluster_id="cluster-web")
    assert client.post("/uploads", json=ok).status_code == 201
    bad = _body(seed=2, cluster_id="cluster-web")
    assert client.post("/uploads", json=bad).status_code == 401


def test_a_rotated_key_takes_effect_without_a_restart(_clean):
    assert _post(_signed()).status_code == 201
    _clean.write_text(json.dumps({"a": "f0" * 32}), encoding="utf-8")
    import os
    import time

    os.utime(_clean, (time.time() + 5, time.time() + 5))  # a newer file, whatever the clock grain
    assert _post(_signed(seed=2)).status_code == 401
    assert _post(_signed(seed=2, secret="f0" * 32)).status_code == 201


def test_no_response_or_log_line_contains_a_key_or_signature(caplog):
    body = _body()
    body["signature"] = "de" * 32
    with caplog.at_level(logging.DEBUG):
        r = _post(body)
    text = r.text + " ".join(rec.getMessage() for rec in caplog.records)
    assert KEY_A not in text and KEY_B not in text and "de" * 32 not in text


def test_the_certificate_is_checked_before_the_signature():
    """Uploading under another client's name is "not your identity" (403), whatever
    the signature says: the clearer of the two refusals."""
    server.configure_identity(lambda request: request.headers.get("x-test-cn"))
    body = _signed("b", 2, secret=KEY_A)             # a's key, b's name, a's certificate
    r = client.post("/clusters/web/uploads", json=body, headers={"x-test-cn": "a"})
    assert r.status_code == 403 and "may not upload as client 'b'" in r.json()["detail"]
    own = client.post("/clusters/web/uploads", json=_signed("a", 1), headers={"x-test-cn": "a"})
    assert own.status_code == 201
    assert _buffered() == ["a"]
