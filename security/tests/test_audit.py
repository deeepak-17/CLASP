"""Tests for security.audit — structured audit logging."""
from __future__ import annotations

from security.audit.logger import (
    AuditEntry,
    AuditLogger,
    SecurityEvent,
    get_audit_logger,
)


class TestSecurityEvent:
    def test_values_are_strings(self) -> None:
        assert SecurityEvent.AUTH_SUCCESS.value == "AUTH_SUCCESS"
        assert SecurityEvent.PRIVACY_BUDGET_EXHAUSTED.value == "PRIVACY_BUDGET_EXHAUSTED"


class TestAuditLogger:
    def test_log_returns_entry(self) -> None:
        logger = AuditLogger(name="test.audit.1")
        entry = logger.log(
            event=SecurityEvent.AUTH_SUCCESS,
            actor="edge-0",
            resource="auth",
        )
        assert isinstance(entry, AuditEntry)
        assert entry.event == SecurityEvent.AUTH_SUCCESS
        assert entry.actor == "edge-0"
        assert entry.success is True

    def test_log_auth_success(self) -> None:
        logger = AuditLogger(name="test.audit.2")
        entry = logger.log_auth_success("edge-0", "mTLS", ip="127.0.0.1")
        assert entry.event == SecurityEvent.AUTH_SUCCESS
        assert "mTLS" in entry.detail

    def test_log_auth_failure(self) -> None:
        logger = AuditLogger(name="test.audit.3")
        entry = logger.log_auth_failure("edge-0", "jwt", "expired")
        assert entry.event == SecurityEvent.AUTH_FAILURE
        assert entry.success is False

    def test_log_adapter_event(self) -> None:
        logger = AuditLogger(name="test.audit.4")
        entry = logger.log_adapter_event(
            SecurityEvent.ADAPTER_UPLOADED, "edge-0", {"size": 1024}
        )
        assert entry.event == SecurityEvent.ADAPTER_UPLOADED
        assert entry.metadata == {"size": 1024}

    def test_log_privacy_event(self) -> None:
        logger = AuditLogger(name="test.audit.5")
        entry = logger.log_privacy_event(
            SecurityEvent.PRIVACY_BUDGET_WARNING, "edge-0", 7.5, 8.0
        )
        assert entry.metadata["epsilon_spent"] == 7.5
        assert entry.metadata["epsilon_target"] == 8.0


class TestGetAuditLogger:
    def test_returns_singleton(self) -> None:
        a = get_audit_logger()
        b = get_audit_logger()
        assert a is b
