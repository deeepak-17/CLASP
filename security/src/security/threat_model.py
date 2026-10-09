from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ThreatActor(Enum):
    MALICIOUS_CLIENT = "malicious_client"        # A compromised or adversarial edge node
    COMPROMISED_SERVER = "compromised_server"    # Attacker with access to cluster server
    NETWORK_EAVESDROPPER = "network_eavesdropper" # Passive MITM on network
    INSIDER = "insider"                           # Authorized user with elevated access

class Asset(Enum):
    SOURCE_CODE = "source_code"           # Proprietary code in training data
    GRADIENT_UPDATES = "gradient_updates" # Per-sample gradients during training
    ADAPTER_WEIGHTS = "adapter_weights"   # Trained LoRA adapter parameters
    MODEL_WEIGHTS = "model_weights"       # Base model weights
    METADATA = "metadata"                 # Training metadata, configs, seeds

class AttackSurface(Enum):
    GRADIENT_INVERSION = "gradient_inversion"     # Reconstruct training data from gradients
    MEMBERSHIP_INFERENCE = "membership_inference" # Determine if a sample was in training set
    MITM_UPLOAD = "mitm_upload"                   # Intercept adapter uploads
    ADAPTER_POISONING = "adapter_poisoning"       # Upload malicious adapter to degrade model
    MODEL_EXTRACTION = "model_extraction"         # Steal the model via API queries
    EAVESDROPPING = "eavesdropping"               # Passive interception of network traffic

@dataclass(frozen=True)
class Threat:
    actor: ThreatActor
    asset: Asset
    attack: AttackSurface
    severity: str  # "critical", "high", "medium", "low"
    mitigation: str
    dp_relevant: bool  # True if DP-SGD mitigates this threat
    mtls_relevant: bool  # True if mTLS mitigates this threat

THREAT_MODEL: list[Threat] = [
    Threat(ThreatActor.COMPROMISED_SERVER, Asset.GRADIENT_UPDATES, AttackSurface.GRADIENT_INVERSION, "critical", "DP-SGD adds noise and clips gradients, making individual records indistinguishable.", True, False),
    Threat(ThreatActor.COMPROMISED_SERVER, Asset.SOURCE_CODE, AttackSurface.MEMBERSHIP_INFERENCE, "high", "DP-SGD prevents confident membership inference via theoretical privacy bounds.", True, False),
    Threat(ThreatActor.NETWORK_EAVESDROPPER, Asset.ADAPTER_WEIGHTS, AttackSurface.EAVESDROPPING, "high", "mTLS encrypts all traffic between edge and cluster nodes.", False, True),
    Threat(ThreatActor.NETWORK_EAVESDROPPER, Asset.METADATA, AttackSurface.MITM_UPLOAD, "high", "mTLS mutually authenticates both ends to prevent MITM attacks.", False, True),
    Threat(ThreatActor.MALICIOUS_CLIENT, Asset.MODEL_WEIGHTS, AttackSurface.MODEL_EXTRACTION, "medium", "Rate limiting and DP on the global model limit information extraction.", True, False),
    Threat(ThreatActor.MALICIOUS_CLIENT, Asset.ADAPTER_WEIGHTS, AttackSurface.ADAPTER_POISONING, "high", "Gradient clipping strictly bounds the influence of any single client update.", True, False),
    Threat(ThreatActor.INSIDER, Asset.GRADIENT_UPDATES, AttackSurface.MEMBERSHIP_INFERENCE, "medium", "DP ensures even authorized insiders cannot infer training data membership.", True, False),
    Threat(ThreatActor.NETWORK_EAVESDROPPER, Asset.SOURCE_CODE, AttackSurface.EAVESDROPPING, "critical", "mTLS encrypts data in transit, ensuring source code remains confidential.", False, True)
]

DP_UNIT_JUSTIFICATION: str = """In CLASP, each client (edge node) represents one developer's private codebase.
The privacy unit is the entire client's participation in a federated round — i.e., whether a given developer participated at all, not whether a specific code snippet was in the training set.
This is because:
(1) the adapter upload is the atomic unit of information leaving the client,
(2) the cluster server sees one update per client per round,
(3) client-level DP matches the threat model where the adversary wants to determine if a specific developer's code influenced the model."""

SYSTEM_BOUNDARY: str = """The system boundary encompasses the Edge Node (Client) and the Cluster Server.
The base model is considered public and untrusted. The training data on the client is private and trusted.
The cluster server is semi-honest: it follows protocol but may attempt to infer client data from updates.
The network is fully untrusted, vulnerable to MITM and eavesdropping."""

def render_markdown() -> str:
    md = "# CLASP Programmatic Threat Model\n\n"
    md += "## DP Unit Justification\n"
    md += DP_UNIT_JUSTIFICATION + "\n\n"
    md += "## System Boundary\n"
    md += SYSTEM_BOUNDARY + "\n\n"
    md += "## Threats\n"
    md += "| Actor | Asset | Attack | Severity | Mitigation | DP Relevant | mTLS Relevant |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for t in THREAT_MODEL:
        md += f"| {t.actor.value} | {t.asset.value} | {t.attack.value} | {t.severity} | {t.mitigation} | {t.dp_relevant} | {t.mtls_relevant} |\n"
    return md
