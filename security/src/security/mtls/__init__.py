from __future__ import annotations

from .ca import create_ca
from .certs import (
    generate_client_cert,
    generate_service_cert,
    setup_all_certs,
    verify_cert,
)
from .context import client_ssl_context, server_ssl_context
from .rotation import CertStatus, check_expiry

# Backward-compat alias: PR #17 (edge) used create_client_ssl_context
create_client_ssl_context = client_ssl_context

__all__ = [
    "CertStatus",
    "check_expiry",
    "client_ssl_context",
    "create_ca",
    "create_client_ssl_context",
    "generate_client_cert",
    "generate_service_cert",
    "server_ssl_context",
    "setup_all_certs",
    "verify_cert",
]
