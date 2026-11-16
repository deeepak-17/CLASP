"""The registry's audit trail (U4): an audit file and the ``access_log`` table.

* **Audit file.** One ``security.audit.AuditLogger`` writing JSON lines to
  ``CLASP_AUDIT_LOG`` (default ``<data root>/audit.jsonl``): every refused
  authorisation, every accepted mutating call, every recorded privacy row.
* **Access log.** Every request except ``/healthz`` is a row in the
  ``access_log`` table: who (the verified certificate name, or null over plain
  HTTP), with which role, the method and path, and the status. ``GET
  /access-log`` serves it to the ``admin`` role.

Both are written by ``security.audit.AccessAuditMiddleware``, installed here.
Neither holds a request body, a header, a token or a signature.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, Request
from security.audit import AccessAuditMiddleware, AuditLogger

from registry.config import data_root

AUDIT_ENV = "CLASP_AUDIT_LOG"
#: Liveness is polled every few seconds by the container runtime: not recorded.
EXEMPT_PATHS = ("/healthz",)

_lock = threading.Lock()
_current: dict = {"path": None, "logger": None}


def audit_path() -> Path:
    """``CLASP_AUDIT_LOG``, else ``audit.jsonl`` under the registry's data root."""
    return Path(os.environ.get(AUDIT_ENV) or data_root() / "audit.jsonl")


def audit_logger() -> AuditLogger:
    """The registry's audit logger, reopened when the configured path changes
    (the data root is read at request time, like the store's)."""
    path = audit_path()
    with _lock:
        if _current["path"] != path:
            if _current["logger"] is not None:
                _current["logger"].close()
            _current.update(path=path, logger=AuditLogger(
                name="clasp.audit.registry", log_file=path, stream=False))
        return _current["logger"]


def close_audit() -> None:
    """Flush and release the audit file (shutdown; tests on Windows)."""
    with _lock:
        if _current["logger"] is not None:
            _current["logger"].close()
        _current.update(path=None, logger=None)


def install(app: FastAPI, *, identity: Callable[[Request], str | None],
            record: Callable[[dict], None]) -> None:
    """Add the access/audit middleware. Call it before the other middleware is
    added, so it sits closest to the routes and sees their real answers."""
    app.add_middleware(AccessAuditMiddleware, audit=audit_logger, identity=identity,
                       service="registry", record=record, exempt_paths=EXEMPT_PATHS)
