"""Service roles, read off the verified mTLS certificate name (U5, C10).

Services identify each other by certificate: the CN the handshake verified
(``security.mtls.peer``) maps to a role, and each route names the roles allowed
to call it with :func:`require_roles`.

The names are the ones ``scripts/make_dev_certs.sh`` issues: ``cluster``,
``demo-ui`` (the panel and the evaluator it runs), ``clasp-admin``, and one
certificate per edge named for its client id (``client-flask``, ...). The
table can be replaced by a JSON file named in ``CLASP_ROLE_TABLE``.

Fall-back (U6): without TLS there is no certificate name, the role is
``open`` and every route is allowed, exactly as before mTLS. A certificate
whose name has no role is ``unknown`` and is refused by every restricted route.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

from fastapi import HTTPException, Request

from security.mtls.peer import peer_identity

ROLE_CLUSTER = "svc:cluster"
ROLE_EVALUATION = "svc:evaluation"
ROLE_ADMIN = "admin"
ROLE_EDGE = "edge"
ROLE_OPEN = "open"
#: A CA-signed certificate whose name maps to nothing.
ROLE_UNKNOWN = "unknown"

#: Roles a route may name. ``open`` and ``unknown`` are outcomes, not grants.
GRANTABLE = frozenset({ROLE_CLUSTER, ROLE_EVALUATION, ROLE_ADMIN, ROLE_EDGE})

DEFAULT_ROLES: Mapping[str, str] = {
    "cluster": ROLE_CLUSTER,
    "demo-ui": ROLE_EVALUATION,
    "clasp-admin": ROLE_ADMIN,
}
ROLE_TABLE_ENV = "CLASP_ROLE_TABLE"
EDGE_PREFIX = "client-"


def load_role_table(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    """``DEFAULT_ROLES``, or the JSON object ``{certificate name: role}`` in the
    file ``CLASP_ROLE_TABLE`` names. A file that cannot be read as that is an
    error: a broken table must not quietly become the default one."""
    env = os.environ if env is None else env
    path = env.get(ROLE_TABLE_ENV)
    if not path:
        return DEFAULT_ROLES
    try:
        table = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{ROLE_TABLE_ENV}={path}: cannot read the role table ({exc})") from exc
    if not isinstance(table, dict) or not all(
            isinstance(k, str) and v in GRANTABLE for k, v in table.items()):
        raise ValueError(f"{ROLE_TABLE_ENV}={path}: must be an object mapping certificate "
                         f"names to one of {sorted(GRANTABLE)}")
    return table


def role_of(common_name: str | None, *, table: Mapping[str, str] = DEFAULT_ROLES) -> str:
    """The role of a verified certificate name.

    ``None`` (no TLS) is ``open``; a name in ``table`` has that role; any other
    name starting ``client-`` is an edge; everything else is ``unknown``.
    """
    if common_name is None:
        return ROLE_OPEN
    if common_name in table:
        return table[common_name]
    if common_name.startswith(EDGE_PREFIX) and len(common_name) > len(EDGE_PREFIX):
        return ROLE_EDGE
    return ROLE_UNKNOWN


def require_roles(*allowed: str,
                  identity: Callable[[Request], str | None] = peer_identity) -> Callable:
    """A FastAPI dependency that admits ``allowed`` roles and answers 403 to the rest.

    ``identity`` returns the caller's verified certificate name (by default the
    one ``PeerCertH11Protocol`` put in the request's scope). ``open`` always
    passes. The dependency returns the caller's role.
    """
    if not allowed or not set(allowed) <= GRANTABLE:
        raise ValueError(f"require_roles needs roles from {sorted(GRANTABLE)}, got {allowed!r}")
    granted = frozenset(allowed)

    def dependency(request: Request) -> str:
        name = identity(request)
        role = role_of(name, table=load_role_table())
        if role == ROLE_OPEN or role in granted:
            return role
        raise HTTPException(
            status_code=403,
            detail=(f"certificate {name!r} has role {role!r}; this route is for "
                    f"{', '.join(sorted(granted))}"),
        )

    return dependency
