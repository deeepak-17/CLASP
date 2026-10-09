from __future__ import annotations

from .ca import create_ca
from .certs import generate_service_cert, generate_client_cert, setup_all_certs, verify_cert
from .context import server_ssl_context, client_ssl_context
from .rotation import check_expiry, CertStatus

# Backward-compat alias: PR #17 (edge) used create_client_ssl_context
create_client_ssl_context = client_ssl_context

__all__ = [
    "create_ca",
    "generate_service_cert",
    "generate_client_cert",
    "setup_all_certs",
    "server_ssl_context",
    "client_ssl_context",
    "create_client_ssl_context",
    "verify_cert",
    "check_expiry",
    "CertStatus",
]
