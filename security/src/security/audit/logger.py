from __future__ import annotations

import itertools
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


#: Numbers the private (file-only) loggers; see ``AuditLogger.__init__``.
_private_names = itertools.count(1)


class SecurityEvent(str, Enum):
    """Enumeration of security events for audit logging."""
    AUTH_SUCCESS = "AUTH_SUCCESS"
    AUTH_FAILURE = "AUTH_FAILURE"
    TOKEN_ISSUED = "TOKEN_ISSUED"
    TOKEN_REVOKED = "TOKEN_REVOKED"
    CERT_GENERATED = "CERT_GENERATED"
    CERT_ROTATED = "CERT_ROTATED"
    CERT_EXPIRED = "CERT_EXPIRED"
    ADAPTER_UPLOADED = "ADAPTER_UPLOADED"
    ADAPTER_VALIDATED = "ADAPTER_VALIDATED"
    ADAPTER_REJECTED = "ADAPTER_REJECTED"
    AGGREGATION_STARTED = "AGGREGATION_STARTED"
    AGGREGATION_COMPLETED = "AGGREGATION_COMPLETED"
    RATE_LIMIT_HIT = "RATE_LIMIT_HIT"
    PRIVACY_BUDGET_RECORDED = "PRIVACY_BUDGET_RECORDED"
    PRIVACY_BUDGET_WARNING = "PRIVACY_BUDGET_WARNING"
    PRIVACY_BUDGET_EXHAUSTED = "PRIVACY_BUDGET_EXHAUSTED"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    ACCESS_DENIED = "ACCESS_DENIED"


@dataclass(frozen=True)
class AuditEntry:
    """Represents a single audit log entry."""
    timestamp: str
    event: SecurityEvent
    actor: str
    resource: str
    detail: str
    success: bool
    ip_address: str | None
    metadata: dict | None


class AuditLogger:
    """Structured security audit logging for CLASP."""

    def __init__(self, name: str = 'clasp.security.audit', log_file: str | Path | None = None,
                 stream: bool = True) -> None:
        """
        Initialize the audit logger.

        Args:
            name: The logger name.
            log_file: Optional path to a file to output logs to (its directory is created).
            stream: Also write each entry to the console. With ``False`` the
                logger writes to ``log_file`` only and is private to this
                instance, so two services (or two tests) never share handlers;
                :meth:`close` releases the file.
        """
        if not stream:
            # A name no other instance has or will have. Not the object's
            # address: Python reuses addresses, and loggers live for the whole
            # process, so a new instance would inherit a dropped one's handlers
            # and write into its file.
            name = f"{name}.{next(_private_names)}"
        self.logger = logging.getLogger(name)
        # Avoid adding handlers multiple times if the logger already has them
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

            class JsonFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
                    msg = record.msg
                    if isinstance(msg, dict):
                        return json.dumps(msg)
                    return json.dumps({"message": str(msg)})

            formatter = JsonFormatter()

            # Default stream handler
            if stream:
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                self.logger.addHandler(stream_handler)

            if log_file:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def close(self) -> None:
        """Flush and release this logger's handlers (and its log file)."""
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

    def log(
        self,
        event: SecurityEvent,
        actor: str,
        resource: str,
        detail: str = '',
        success: bool = True,
        ip_address: str | None = None,
        metadata: dict | None = None
    ) -> AuditEntry:
        """Log a security event and return the generated entry."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event=event,
            actor=actor,
            resource=resource,
            detail=detail,
            success=success,
            ip_address=ip_address,
            metadata=metadata
        )
        
        log_dict = {
            "timestamp": entry.timestamp,
            "event": entry.event.value,
            "actor": entry.actor,
            "resource": entry.resource,
            "detail": entry.detail,
            "success": entry.success,
            "ip_address": entry.ip_address,
            "metadata": entry.metadata,
        }
        
        # Log at INFO level
        self.logger.info(log_dict)
        return entry

    def log_auth_success(self, actor: str, method: str, ip: str | None = None) -> AuditEntry:
        """Convenience method to log an authentication success."""
        return self.log(
            event=SecurityEvent.AUTH_SUCCESS,
            actor=actor,
            resource="authentication",
            detail=f"Successful authentication via {method}",
            success=True,
            ip_address=ip
        )

    def log_auth_failure(self, actor: str, method: str, reason: str, ip: str | None = None) -> AuditEntry:
        """Convenience method to log an authentication failure."""
        return self.log(
            event=SecurityEvent.AUTH_FAILURE,
            actor=actor,
            resource="authentication",
            detail=f"Failed authentication via {method}: {reason}",
            success=False,
            ip_address=ip
        )

    def log_adapter_event(self, event: SecurityEvent, client_id: str, adapter_info: dict) -> AuditEntry:
        """Convenience method to log an adapter related event."""
        return self.log(
            event=event,
            actor=client_id,
            resource="adapter",
            detail="Adapter operation",
            success=event != SecurityEvent.ADAPTER_REJECTED,
            metadata=adapter_info
        )

    def log_privacy_event(self, event: SecurityEvent, client_id: str, epsilon_spent: float, epsilon_target: float) -> AuditEntry:
        """Convenience method to log a privacy budget event."""
        return self.log(
            event=event,
            actor=client_id,
            resource="privacy_budget",
            detail=f"Privacy budget event: {epsilon_spent}/{epsilon_target}",
            success=True,
            metadata={"epsilon_spent": epsilon_spent, "epsilon_target": epsilon_target}
        )

AUDIT_LOG_ENV = "CLASP_AUDIT_LOG"


class ServiceAudit:
    """One service's audit logger, following its configured file.

    The file is ``CLASP_AUDIT_LOG`` when set, else ``default_path()`` (a
    service with a data directory passes ``<data dir>/audit.jsonl``), else none:
    the entries then go to the console, where the process's logs go. The path
    is read on each use, and the file is reopened when it changes.
    """

    def __init__(self, service: str, default_path=None) -> None:
        self.service = service
        self._default_path = default_path
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._logger: AuditLogger | None = None

    def path(self) -> Path | None:
        configured = os.environ.get(AUDIT_LOG_ENV)
        if configured:
            return Path(configured)
        return Path(self._default_path()) if self._default_path is not None else None

    def logger(self) -> AuditLogger:
        path = self.path()
        with self._lock:
            if self._logger is None or self._path != path:
                if self._logger is not None:
                    self._logger.close()
                name = f"clasp.audit.{self.service}"
                self._logger = (AuditLogger(name=name, log_file=path, stream=False)
                                if path is not None else AuditLogger(name=name))
                self._path = path
            return self._logger

    def close(self) -> None:
        """Flush and release the audit file."""
        with self._lock:
            if self._logger is not None:
                self._logger.close()
            self._logger, self._path = None, None


_audit_logger_instance: AuditLogger | None = None

def get_audit_logger(name: str = 'clasp.security.audit') -> AuditLogger:
    """Module-level singleton factory for AuditLogger."""
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger(name=name)
    return _audit_logger_instance
