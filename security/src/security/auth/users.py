"""People who may log in to a page, and the token a login gives them (U5, C1, C2).

Services identify each other by certificate; people log in. The demo panel and
each edge page keep their users in a JSON file written by
``scripts/make_users.py``: a name, a scrypt hash of the password, a role and,
for a developer, the one client whose page they may use. There is no
self-registration. A login answers with a JWT (``security.auth.jwt_handler``)
that carries the role and lasts ``LOGIN_LIFETIME_MINUTES``.

Roles: ``operator`` (the demo panel) and ``developer`` with ``subject`` set to
a client id (that client's edge page).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from collections.abc import Mapping
from pathlib import Path

from .hmac_signer import secret_from_env
from .jwt_handler import JWTConfig, TokenError, create_token, verify_token

ROLE_OPERATOR = "operator"
ROLE_DEVELOPER = "developer"
USER_ROLES = (ROLE_OPERATOR, ROLE_DEVELOPER)
#: The spec's login lifetime: 8 hours.
LOGIN_LIFETIME_MINUTES = 8 * 60
JWT_SECRET_ENV = "CLASP_JWT_SECRET"
MIN_PASSWORD_LENGTH = 8
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}
_KEY_LENGTH = 64


class UserStoreError(Exception):
    """The users file exists but cannot be trusted (unreadable or malformed)."""


def _derive(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, dklen=_KEY_LENGTH, **SCRYPT)


class UserStore:
    """Users of one page, in a JSON file:
    ``{"scrypt": {...}, "users": {name: {"salt", "hash", "role", "subject"}}}``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._users: dict[str, dict] = self._load()
        # what an unknown name is checked against, so it costs the same as a known one
        self._decoy_salt = secrets.token_bytes(16)

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            users = json.loads(self.path.read_text(encoding="utf-8"))["users"]
            for entry in users.values():
                bytes.fromhex(entry["salt"]), bytes.fromhex(entry["hash"])
                if entry["role"] not in USER_ROLES:
                    raise ValueError(f"unknown role {entry['role']!r}")
            return dict(users)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise UserStoreError(
                f"users file {self.path} is unreadable ({type(exc).__name__}); write it again "
                f"with scripts/make_users.py") from exc

    def usernames(self) -> list[str]:
        return sorted(self._users)

    def add(self, username: str, password: str, role: str, subject: str | None = None) -> None:
        """Add a user, or replace one's password, role and subject."""
        if not username or username != username.strip():
            raise ValueError("a username is required")
        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"the password must be at least {MIN_PASSWORD_LENGTH} characters")
        if role not in USER_ROLES:
            raise ValueError(f"role must be one of {USER_ROLES}, got {role!r}")
        if role == ROLE_DEVELOPER and not subject:
            raise ValueError("a developer needs the client id whose page they use")
        if role == ROLE_OPERATOR and subject:
            raise ValueError("an operator has no client id")
        salt = secrets.token_bytes(16)
        users = {**self._users, username: {
            "salt": salt.hex(), "hash": _derive(password, salt).hex(),
            "role": role, "subject": subject}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"scrypt": SCRYPT, "users": users}, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        self._users = users

    def verify(self, username: str, password: str) -> dict | None:
        """``{"role", "subject"}`` for the right password, else ``None``."""
        entry = self._users.get(username)
        if entry is None:
            _derive(password, self._decoy_salt)  # same work as for a real user
            return None
        derived = _derive(password, bytes.fromhex(entry["salt"]))
        if not hmac.compare_digest(derived, bytes.fromhex(entry["hash"])):
            return None
        return {"role": entry["role"], "subject": entry.get("subject")}


def jwt_config_from_env(env: Mapping[str, str] | None = None) -> tuple[JWTConfig, bool]:
    """The page's token settings, and whether the signing key was generated.

    The key is ``CLASP_JWT_SECRET`` or the file ``CLASP_JWT_SECRET_FILE`` names.
    With neither, a random key is made for this process: logins work, and end
    when the process stops. Lifetime: ``LOGIN_LIFETIME_MINUTES``.
    """
    secret = secret_from_env(JWT_SECRET_ENV, env)
    generated = secret is None
    if generated:
        secret = secrets.token_urlsafe(48)
    return JWTConfig(secret_key=secret, token_expiry_minutes=LOGIN_LIFETIME_MINUTES), generated


def issue_token(store: UserStore, username: str, password: str, config: JWTConfig) -> str | None:
    """A signed token for a correct login, else ``None``."""
    user = store.verify(username, password)
    if user is None:
        return None
    return create_token(username, config, {"role": user["role"], "subject": user["subject"]})


def bearer_claims(authorization_header: str | None, config: JWTConfig) -> dict:
    """The claims of ``Authorization: Bearer <token>``; ``TokenError`` when the
    header is missing or malformed, or the token is invalid or expired."""
    scheme, _, token = (authorization_header or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise TokenError("a bearer token is required")
    return verify_token(token.strip(), config)
