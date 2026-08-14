# CLASP Threat Model v1

**Owner:** Kapilan V (P3 — Security & QA Lead)
**Date:** August 2026
**Framework:** STRIDE-AI (OWASP GenAI LLM Top 10, 2026 edition)

---

## 1. System Overview

CLASP is an on-premise, hierarchical federated code-generation system.
No source code leaves the organization. Personalization is composed from
three weight layers:

```
W_new = W_base + α·ΔW_cluster + β·ΔW_client
```

- **W_base** — frozen DeepSeek-Coder (1.3B dev / 6.7B target), NF4 quantised
- **ΔW_cluster** — federated LoRA adapter (aggregated from multiple developers)
- **ΔW_client** — per-developer LoRA adapter (trained locally)

---

## 2. Actors

| Actor | Trust Level | Description |
|-------|------------|-------------|
| **Developer (Edge)** | Honest-but-curious | Trains local LoRA, uploads to cluster. May try to infer other developers' data. |
| **Cluster Admin** | Trusted | Operates the aggregation server. Has access to all uploaded LoRA updates. |
| **External Attacker** | Untrusted | No direct system access. May intercept network traffic or attempt to join as a rogue edge. |
| **Compromised Edge** | Malicious | A developer machine infected with malware. May send poisoned LoRA updates. |

---

## 3. Assets

| Asset | Location | Sensitivity |
|-------|----------|-------------|
| Training data (source code) | Edge PC only | **HIGH** — proprietary code, never leaves edge |
| Client LoRA weights (ΔW_client) | Edge PC | MEDIUM — may encode patterns of private code |
| Gradient updates | Edge → Cluster (transit) | **HIGH** — can leak training data via DLG attacks |
| Cluster LoRA weights (ΔW_cluster) | Cluster aggregator | MEDIUM — aggregated from multiple developers |
| Base model (W_base) | Edge + Cluster | LOW — publicly available DeepSeek-Coder |
| Version Registry metadata | Registry service | LOW — adapter versions, Pass@k scores |
| TLS certificates | Edge + Cluster | **HIGH** — compromise enables MITM attacks |

---

## 4. Attack Surface (STRIDE-AI)

### 4.1 Spoofing — Impersonating a legitimate edge device
- **Threat:** Rogue device joins the federated training by presenting forged credentials.
- **Impact:** Model poisoning via malicious LoRA updates.
- **Mitigation:** mTLS with private CA. Each edge must present a cert signed by the CLASP root CA.
- **CLASP component:** `security.ca`, `security.tls_config`

### 4.2 Tampering — Modifying LoRA updates in transit
- **Threat:** MITM attack modifies gradient/LoRA updates between edge and cluster.
- **Impact:** Corrupted cluster LoRA, degraded code generation quality.
- **Mitigation:** TLS 1.3 encryption (integrity + confidentiality). mTLS ensures both endpoints are authenticated.
- **CLASP component:** `security.tls_config`, `security.mtls_service`

### 4.3 Repudiation — Developer denies submitting a poisoned update
- **Threat:** A malicious developer submits a backdoored LoRA update, then denies it.
- **Impact:** Inability to trace the source of model degradation.
- **Mitigation:** mTLS client certs provide non-repudiation. Each upload is tied to a specific client certificate CN.

### 4.4 Information Disclosure — Gradient leakage
- **Threat:** An honest-but-curious cluster admin reconstructs private training data from uploaded LoRA updates using DLG (Deep Leakage from Gradients) or analytics-based attacks.
- **Impact:** Exposure of proprietary source code.
- **Mitigation:** DP-SGD with ghost clipping. Noise is injected into LoRA gradients before upload, providing (ε, δ)-DP guarantee.
- **CLASP component:** `security.dp_engine`, `security.epsilon_tracker`

### 4.5 Denial of Service — Overloading the aggregation server
- **Threat:** A compromised edge floods the cluster with oversized or malformed LoRA uploads.
- **Impact:** Cluster becomes unresponsive; legitimate training stalls.
- **Mitigation:** Rate limiting on the cluster FastAPI endpoint. Input validation (LoRA tensor shape/size checks).

### 4.6 Elevation of Privilege — Cluster admin accesses raw updates
- **Threat:** The cluster admin extracts individual developer contributions from the aggregated model.
- **Impact:** Privacy violation — individual code patterns exposed.
- **Mitigation:** DP-SGD ensures each individual contribution is masked by calibrated noise. FedASK (NeurIPS 2025) sketching further protects individual updates during aggregation.

---

## 5. Cross-Cluster Leakage Scenarios

| # | Scenario | Attack Vector | Mitigation |
|---|----------|--------------|------------|
| 1 | **Gradient reconstruction (DLG)** | Cluster admin applies DLG to reconstruct code snippets from uploaded LoRA deltas | DP-SGD noise (ε ≤ 8.0) makes reconstruction infeasible |
| 2 | **Model poisoning** | Compromised edge sends adversarial LoRA updates that degrade the cluster model | FedProx regularization + mTLS identity binding + contribution monitoring |
| 3 | **Membership inference** | Attacker queries the cluster model to determine if a specific code sample was in the training set | DP-SGD provides formal membership privacy guarantees |
| 4 | **Cluster-swapping** | Attacker injects LoRA weights from Cluster A into Cluster B | mTLS cert CN includes cluster_id; registry validates cluster affinity |
| 5 | **Eavesdropping** | Network-level interception of LoRA uploads between edge and cluster | TLS 1.3 encryption; mTLS prevents MITM even if network is compromised |

---

## 6. Security Controls Summary

| Control | Algorithm / Tool | Paper Reference |
|---------|-----------------|-----------------|
| Transport encryption | TLS 1.3 + ECDSA P-256 | NIST SP 1800-35 (2025) |
| Mutual authentication | mTLS with private CA | NIST SP 1800-35 (2025) |
| Gradient privacy | DP-SGD + Ghost Clipping | FlashDP (NeurIPS 2025) |
| Privacy accounting | RDP Moments Accountant | Time-Adaptive Privacy (ICLR 2025) |
| Privacy auditing | Hidden-state auditing | Cebere et al. (ICLR 2025) |
| Federated aggregation privacy | Double sketching | FedASK (NeurIPS 2025) |
| LoRA noise mitigation | Alternating A/B updates | LA-LoRA (ICLR 2026) |
| Threat modeling | STRIDE-AI + ASTRIDE | OWASP GenAI LLM Top 10 (2026) |

---

## 7. Residual Risks

1. **Insider threat (cluster admin with DB access):** DP-SGD mitigates but does not eliminate this risk at very low ε budgets.
2. **Supply-chain attack on base model:** W_base is a public model — we trust DeepSeek's integrity. Not in scope for P3.
3. **Side-channel attacks (timing, power):** Not addressed; requires hardware-level mitigations beyond this project's scope.
