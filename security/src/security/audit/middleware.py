"""Who called what: one middleware that records requests and writes the audit trail.

For every request outside ``exempt_paths`` it builds a row
``{"at", "identity", "role", "method", "path", "adapter", "version", "status"}``
and hands it to ``record`` (the registry keeps these in its ``access_log``
table). It also writes one line to the service's :class:`AuditLogger`:

* a refused authorisation (401 or 403) -> ``log_auth_failure``;
* an accepted mutating call (POST/PUT/PATCH/DELETE answered below 400) ->
  ``log_adapter_event``.

Reads are recorded but not audited. Only the caller's verified certificate
name, its role, the method, the path and the status are written: never a body,
a header, a token or a signature. A failure to record or to audit is logged and
never fails the request.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from fastapi import Request, Response
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from security.auth.roles import load_role_table, role_of

from .logger import AuditLogger, SecurityEvent

_log = logging.getLogger(__name__)
_ADAPTER_PATH = re.compile(r"^/adapters/([^/]+)(?:/versions/(\d+))?")
MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def event_for(path: str) -> SecurityEvent:
    """The audit event of an accepted mutating call, from its path."""
    if path.endswith("/login"):
        return SecurityEvent.TOKEN_ISSUED
    if path.endswith("/uploads") or path.endswith("/versions"):
        return SecurityEvent.ADAPTER_UPLOADED
    if path.endswith("/aggregate"):
        return SecurityEvent.AGGREGATION_COMPLETED
    return SecurityEvent.ADAPTER_VALIDATED


class AccessAuditMiddleware(BaseHTTPMiddleware):
    """See the module docstring.

    Args:
        audit: returns the service's ``AuditLogger`` (or ``None``: no audit file).
        identity: the caller's verified certificate name for a request, or ``None``.
        service: the service's name, written with each audit line.
        role: ``(request, identity) -> role``; by default the role of the
            certificate name (``security.auth.roles``). A page with logins
            passes its own (the role in the person's token).
        record: called with each row (in a worker thread), or ``None``.
        exempt_paths: paths that are neither recorded nor audited (liveness).
    """

    def __init__(self, app: Any, *, audit: Callable[[], AuditLogger | None],
                 identity: Callable[[Request], str | None], service: str,
                 record: Callable[[dict], None] | None = None,
                 exempt_paths: Iterable[str] = (),
                 role: Callable[[Request, str | None], str] | None = None) -> None:
        super().__init__(app)
        self.audit = audit
        self.identity = identity
        self.role = role
        self.service = service
        self.record = record
        self.exempt_paths = frozenset(exempt_paths)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        path = request.url.path
        if path in self.exempt_paths:
            return response
        try:
            await self._after(request, path, response.status_code)
        except Exception:  # noqa: BLE001 - bookkeeping must never fail the request
            _log.exception("could not record %s %s", request.method, path)
        return response

    async def _after(self, request: Request, path: str, status: int) -> None:
        try:
            name = self.identity(request)
            role = (self.role(request, name) if self.role is not None
                    else role_of(name, table=load_role_table()))
        except Exception:  # noqa: BLE001 - an unreadable identity is recorded as such
            name, role = None, "unknown"
        found = _ADAPTER_PATH.match(path)
        row = {
            "at": datetime.now(timezone.utc).isoformat(),
            "identity": name, "role": role, "method": request.method, "path": path,
            "adapter": found.group(1) if found else None,
            "version": int(found.group(2)) if found and found.group(2) else None,
            "status": status,
        }
        audit = self.audit()
        if audit is not None:
            actor = name or "anonymous"
            ip = request.client.host if request.client else None
            if status in (401, 403):
                audit.log_auth_failure(actor, "mtls-role" if name else "none",
                                       f"{request.method} {path} -> {status} (role {role})", ip)
            elif request.method in MUTATING and status < 400:
                audit.log_adapter_event(event_for(path), actor, {
                    "service": self.service, "role": role, "method": request.method,
                    "path": path, "status": status})
        if self.record is not None:
            await run_in_threadpool(self.record, row)
