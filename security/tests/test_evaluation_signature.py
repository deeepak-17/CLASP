"""HMAC over an evaluation result, tied to the candidate's bytes (U4).

The evaluator signs what it measured; the registry verifies it on ``/promote``.
Whoever carries the result in between (the demo panel) cannot change a metric
or attach the result to another version without breaking the signature.
"""
from __future__ import annotations

import json

import pytest

from security.auth.hmac_signer import (
    SecretConfigError,
    evaluation_message,
    secret_from_env,
    sign_evaluation,
    sign_payload,
    verify_evaluation,
)

SECRET = "a" * 64
SHA = "ab" * 32
EVAL = {
    "adapter": {"name": "cluster-web", "version": 2, "kind": "cluster", "cluster_id": "web"},
    "in_project": {"edit_similarity": 0.5123, "exact_match": 0.1, "n_examples": 60},
    "guard": [{"benchmark": "LineCompletion", "edit_similarity": 0.61, "n_examples": 300}],
    "baseline_in_project": {"edit_similarity": 0.45, "exact_match": 0.08, "n_examples": 60},
    "baseline_noise_band": 0.02,
    "seed": 0,
}
BASE_GUARD = [{"benchmark": "LineCompletion", "edit_similarity": 0.6, "n_examples": 300}]


def _sign(**over):
    args = {"adapter": "cluster-web", "version": 2, "candidate_sha256": SHA, "eval_result": EVAL,
            "baseline_guard": BASE_GUARD, **over}
    return sign_evaluation(secret=SECRET, **args)


def test_the_message_is_the_canonical_json_the_plan_fixes():
    message = evaluation_message("cluster-web", 2, SHA, EVAL, BASE_GUARD)
    expected = json.dumps(
        {"adapter": "cluster-web", "version": 2, "candidate_sha256": SHA, "eval": EVAL,
         "baseline_guard": BASE_GUARD}, sort_keys=True, separators=(",", ":")).encode()
    assert message == expected
    assert _sign() == sign_payload(expected, SECRET)


def test_key_order_and_a_json_round_trip_do_not_change_the_signature():
    shuffled = dict(reversed(list(EVAL.items())))
    carried = json.loads(json.dumps(EVAL))  # what arrives after HTTP
    assert _sign(eval_result=shuffled) == _sign() == _sign(eval_result=carried)


def test_a_valid_signature_verifies():
    assert verify_evaluation("cluster-web", 2, SHA, EVAL, BASE_GUARD, _sign(), SECRET) is True


@pytest.mark.parametrize("change", [
    {"adapter": "cluster-scientific"},
    {"version": 3},
    {"candidate_sha256": "cd" * 32},
    {"eval_result": {**EVAL, "in_project": {**EVAL["in_project"], "edit_similarity": 0.9}}},
    {"eval_result": {**EVAL, "baseline_noise_band": 0.0}},
    {"baseline_guard": [{**BASE_GUARD[0], "edit_similarity": 0.1}]},
    {"baseline_guard": []},
])
def test_any_change_breaks_the_signature(change):
    args = {"adapter": "cluster-web", "version": 2, "candidate_sha256": SHA, "eval_result": EVAL,
            "baseline_guard": BASE_GUARD, **change}
    assert verify_evaluation(args["adapter"], args["version"], args["candidate_sha256"],
                             args["eval_result"], args["baseline_guard"], _sign(), SECRET) is False


def test_the_wrong_secret_or_a_malformed_signature_does_not_verify():
    assert verify_evaluation("cluster-web", 2, SHA, EVAL, BASE_GUARD, _sign(), "b" * 64) is False
    for bad in ("", "zz", None, 12, _sign()[:-1]):
        assert verify_evaluation("cluster-web", 2, SHA, EVAL, BASE_GUARD, bad, SECRET) is False


# ---- where the secret comes from (C3) ---------------------------------------------
def test_secret_from_the_variable_or_from_the_file_it_names(tmp_path):
    assert secret_from_env("CLASP_EVAL_HMAC_SECRET", {}) is None  # not configured: the fall-back
    assert secret_from_env("CLASP_EVAL_HMAC_SECRET", {"CLASP_EVAL_HMAC_SECRET": " s3 \n"}) == "s3"
    key = tmp_path / "eval_hmac.key"
    key.write_text("f00d\n", encoding="utf-8")
    assert secret_from_env("CLASP_EVAL_HMAC_SECRET",
                           {"CLASP_EVAL_HMAC_SECRET_FILE": str(key)}) == "f00d"
    # the variable itself wins over the file
    assert secret_from_env("CLASP_EVAL_HMAC_SECRET", {
        "CLASP_EVAL_HMAC_SECRET": "direct", "CLASP_EVAL_HMAC_SECRET_FILE": str(key)}) == "direct"


def test_a_configured_secret_that_cannot_be_read_is_an_error_not_a_fallback(tmp_path):
    empty = tmp_path / "empty.key"
    empty.write_text("  \n", encoding="utf-8")
    for env in ({"CLASP_EVAL_HMAC_SECRET_FILE": str(tmp_path / "missing.key")},
                {"CLASP_EVAL_HMAC_SECRET_FILE": str(empty)},
                {"CLASP_EVAL_HMAC_SECRET": "   "}):
        with pytest.raises(SecretConfigError) as err:
            secret_from_env("CLASP_EVAL_HMAC_SECRET", env)
        assert "CLASP_EVAL_HMAC_SECRET" in str(err.value)


def test_an_error_never_contains_the_secret(tmp_path):
    with pytest.raises(SecretConfigError) as err:
        secret_from_env("CLASP_EVAL_HMAC_SECRET", {"CLASP_EVAL_HMAC_SECRET": " "})
    assert "s3cr3t" not in str(err.value)
