"""security.auth.users: who may log in to a page, and the token a login gives (U5, C2)."""
from __future__ import annotations

import json
import time

import pytest

from security.auth.jwt_handler import JWTConfig, TokenError, verify_token
from security.auth.users import (
    LOGIN_LIFETIME_MINUTES,
    UserStore,
    UserStoreError,
    bearer_claims,
    issue_token,
    jwt_config_from_env,
)

CONFIG = JWTConfig(secret_key="s" * 48, token_expiry_minutes=LOGIN_LIFETIME_MINUTES)
PASSWORD = "correct horse battery staple"


@pytest.fixture
def store(tmp_path):
    s = UserStore(tmp_path / "users" / "panel.json")
    s.add("operator", PASSWORD, "operator")
    s.add("dev-flask", "another pass phrase", "developer", subject="client-flask")
    return s


def test_logins_last_eight_hours():
    assert LOGIN_LIFETIME_MINUTES == 8 * 60


def test_round_trip_and_wrong_password_is_none(store):
    assert store.verify("operator", PASSWORD) == {"role": "operator", "subject": None}
    assert store.verify("dev-flask", "another pass phrase") == {"role": "developer",
                                                               "subject": "client-flask"}
    assert store.verify("operator", "wrong") is None
    assert store.verify("operator", "") is None
    assert store.verify("nobody", PASSWORD) is None
    assert store.verify("OPERATOR", PASSWORD) is None  # names are exact


def test_the_file_survives_a_reopen_and_holds_no_plaintext(store):
    again = UserStore(store.path)
    assert again.verify("operator", PASSWORD) == {"role": "operator", "subject": None}
    raw = store.path.read_text(encoding="utf-8")
    assert PASSWORD not in raw and "another pass phrase" not in raw
    doc = json.loads(raw)
    entry = doc["users"]["operator"]
    assert set(entry) == {"salt", "hash", "role", "subject"}
    assert len(bytes.fromhex(entry["salt"])) >= 16 and len(bytes.fromhex(entry["hash"])) == 64
    # each user has its own salt, so equal passwords would not hash alike
    assert doc["users"]["operator"]["salt"] != doc["users"]["dev-flask"]["salt"]
    assert doc["scrypt"] == {"n": 2 ** 14, "r": 8, "p": 1}


def test_adding_a_user_again_replaces_the_password(store):
    store.add("operator", "a new pass phrase", "operator")
    assert store.verify("operator", PASSWORD) is None
    assert store.verify("operator", "a new pass phrase") == {"role": "operator", "subject": None}


@pytest.mark.parametrize("args", [
    ("", PASSWORD, "operator", None),
    ("op", "short", "operator", None),               # under 8 characters
    ("op", PASSWORD, "root", None),                  # not a role a page knows
    ("op", PASSWORD, "developer", None),             # a developer is a developer of one client
    ("op", PASSWORD, "operator", "client-flask"),    # an operator has no client
])
def test_bad_users_are_refused_and_nothing_is_written(tmp_path, args):
    s = UserStore(tmp_path / "u.json")
    with pytest.raises(ValueError):
        s.add(*args[:3], subject=args[3])
    assert not (tmp_path / "u.json").exists()


def test_a_missing_file_is_an_empty_store_and_a_corrupt_one_is_an_error(tmp_path):
    assert UserStore(tmp_path / "none.json").verify("operator", PASSWORD) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(UserStoreError, match="bad.json"):
        UserStore(bad)
    assert bad.read_text(encoding="utf-8") == "{not json"


def test_a_login_gives_a_token_with_the_role_and_the_subject(store):
    token = issue_token(store, "dev-flask", "another pass phrase", CONFIG)
    claims = verify_token(token, CONFIG)
    assert claims["sub"] == "dev-flask" and claims["role"] == "developer"
    assert claims["subject"] == "client-flask"
    assert 8 * 3600 - 5 <= claims["exp"] - claims["iat"] <= 8 * 3600
    assert issue_token(store, "dev-flask", "wrong", CONFIG) is None
    assert issue_token(store, "nobody", PASSWORD, CONFIG) is None


def test_bearer_claims_reads_the_authorization_header(store):
    token = issue_token(store, "operator", PASSWORD, CONFIG)
    assert bearer_claims(f"Bearer {token}", CONFIG)["role"] == "operator"
    assert bearer_claims(f"bearer {token}", CONFIG)["role"] == "operator"
    for bad in (None, "", token, "Basic abc", "Bearer ", "Bearer not.a.token",
                f"Bearer {token[:-2]}xx"):
        with pytest.raises(TokenError):
            bearer_claims(bad, CONFIG)
    other = JWTConfig(secret_key="t" * 48)
    with pytest.raises(TokenError):
        bearer_claims(f"Bearer {token}", other)


def test_an_expired_token_is_refused(store):
    expired = JWTConfig(secret_key=CONFIG.secret_key, token_expiry_minutes=-1)
    token = issue_token(store, "operator", PASSWORD, expired)
    with pytest.raises(TokenError, match="expired"):
        bearer_claims(f"Bearer {token}", CONFIG)


def test_verifying_takes_about_as_long_for_an_unknown_user(store):
    """No early exit that tells an attacker which names exist."""
    def timed(name):
        t0 = time.perf_counter()
        store.verify(name, "x" * 12)
        return time.perf_counter() - t0

    known, unknown = min(timed("operator") for _ in range(3)), min(timed("ghost") for _ in range(3))
    assert unknown > known / 4


def test_jwt_config_from_env(tmp_path):
    cfg, generated = jwt_config_from_env({"CLASP_JWT_SECRET": "x" * 40})
    assert cfg.secret_key == "x" * 40 and cfg.token_expiry_minutes == 480 and generated is False
    key = tmp_path / "jwt.key"
    key.write_text("y" * 40 + "\n", encoding="utf-8")
    cfg, generated = jwt_config_from_env({"CLASP_JWT_SECRET_FILE": str(key)})
    assert cfg.secret_key == "y" * 40 and generated is False
    # nothing configured: a key for this process only (logins end when it stops)
    one, generated = jwt_config_from_env({})
    two, _ = jwt_config_from_env({})
    assert generated is True and len(one.secret_key) >= 43 and one.secret_key != two.secret_key
