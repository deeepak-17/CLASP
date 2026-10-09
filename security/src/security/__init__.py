"""CLASP Security (P3 — Kapilan).

A LIBRARY, not a service. Imported by the Edge and Cluster modules to apply
Opacus DP-SGD on gradients and mTLS on the edge<->cluster channel. It has no
standalone Dockerfile and is never deployed independently.

Public API
----------

Differential Privacy (DP-SGD):
    DPConfig           — configuration dataclass (on/off toggle, ε target, σ, C)
    make_private       — wraps (model, optimizer, loader) with Opacus DP-SGD
    PrivacyAccountant  — RDP-based ε tracking across steps and rounds
    BudgetExhaustedError — raised when cumulative ε exceeds target_epsilon
    get_privacy_engine — retrieves the Opacus PrivacyEngine from a wrapped model

mTLS:
    create_ca              — create a local RSA-4096 CA
    generate_service_cert  — issue a server cert (Edge, Cluster, Registry)
    generate_client_cert   — issue a client cert (edge nodes)
    setup_all_certs        — one-command cert generation for all services
    verify_cert            — validate a cert against the CA
    server_ssl_context     — SSL context for FastAPI/Uvicorn servers
    client_ssl_context     — SSL context for httpx/requests clients
    check_expiry           — cert expiry status
    CertStatus             — expiry check result

Threat Model:
    THREAT_MODEL           — list of identified threats
    render_markdown        — render threat model as markdown
"""
__version__ = "0.1.0"

# ── DP-SGD ──────────────────────────────────────────────────────────────────
from security.dp import (
    BudgetExhaustedError,
    DPConfig,
    EpsilonTracker,
    PrivacyAccountant,
    get_privacy_engine,
    make_private,
)

# ── mTLS ────────────────────────────────────────────────────────────────────
from security.mtls import (
    CertStatus,
    check_expiry,
    client_ssl_context,
    create_ca,
    create_client_ssl_context,
    generate_client_cert,
    generate_service_cert,
    server_ssl_context,
    setup_all_certs,
    verify_cert,
)

# ── Threat Model ────────────────────────────────────────────────────────────
from security.threat_model import THREAT_MODEL, render_markdown

__all__ = [
    # Threat Model
    "THREAT_MODEL",
    "BudgetExhaustedError",
    "CertStatus",
    # DP
    "DPConfig",
    "EpsilonTracker",
    "PrivacyAccountant",
    "check_expiry",
    "client_ssl_context",
    # mTLS
    "create_ca",
    "create_client_ssl_context",
    "generate_client_cert",
    "generate_service_cert",
    "get_privacy_engine",
    "make_private",
    "render_markdown",
    "server_ssl_context",
    "setup_all_certs",
    "verify_cert",
]
