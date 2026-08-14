"""CLASP Security (P3 — Kapilan).

A LIBRARY, not a service. Imported by the Edge and Cluster modules to apply
Opacus DP-SGD on gradients and mTLS on the edge<->cluster channel. It has no
standalone Dockerfile and is never deployed independently.

Public API
----------
DP-SGD (differential privacy):
    DPConfig             — training-time privacy configuration
    make_private         — wrap model/optimizer/loader with Opacus DP-SGD
    make_private_lora    — specialised wrapper for PEFT/LoRA on NF4 bases
    EpsilonTracker       — standalone RDP privacy accounting
    get_current_epsilon  — query ε from a live PrivacyEngine
    check_budget_exceeded — check if ε > max

mTLS (transport security):
    CertificateAuthority       — local ECDSA P-256 CA
    create_server_ssl_context  — mTLS context for the server (cluster)
    create_client_ssl_context  — mTLS context for the client (edge)
"""
import importlib

__version__ = "0.1.0"

# -- mTLS (always available — only needs cryptography) --------------------
from security.ca import CertificateAuthority
from security.tls_config import create_server_ssl_context, create_client_ssl_context

# -- DP-SGD (lazy-loaded to avoid torch/opacus dependency at import time) --
_DP_SGD_LAZY = {
    "DPConfig": "security.dp_config",
    "make_private": "security.dp_engine",
    "make_private_lora": "security.dp_engine",
    "EpsilonTracker": "security.epsilon_tracker",
    "get_current_epsilon": "security.privacy_accountant",
    "check_budget_exceeded": "security.privacy_accountant",
}

def __getattr__(name: str):
    if name in _DP_SGD_LAZY:
        module = importlib.import_module(_DP_SGD_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # DP-SGD
    "DPConfig",
    "make_private",
    "make_private_lora",
    "EpsilonTracker",
    "get_current_epsilon",
    "check_budget_exceeded",
    # mTLS
    "CertificateAuthority",
    "create_server_ssl_context",
    "create_client_ssl_context",
]
