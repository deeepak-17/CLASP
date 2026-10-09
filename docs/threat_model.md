# CLASP Threat Model v1

## System Overview
CLASP is a federated code-generation system with per-developer personalization. It leverages LoRA adapters to customize large language models for individual developers, while using Federated Learning to share improvements globally without centralizing private source code.

## System Boundary Diagram
```text
[Edge Node (Developer Client)] 
       | (Private Code)
       |
    (mTLS) -- Adapter Updates / Global Models
       |
[Cluster Server (Aggregator)]
       |
[Registry & Eval Services]
```

## Trust Assumptions
- **Base Model**: Public, potentially untrusted.
- **Training Data**: Private to the developer, fully trusted locally, highly sensitive.
- **Cluster Server**: Semi-honest ("honest-but-curious") - it follows the protocol but might try to infer training data from uploaded adapters.
- **Network**: Untrusted, assumed susceptible to eavesdropping and Man-in-the-Middle (MITM) attacks.

## Actor Definitions
- **Malicious Client**: An adversarial edge node or a compromised client instance aiming to poison the model or extract data.
- **Compromised Server**: An attacker with access to the cluster server attempting to reverse-engineer client updates.
- **Network Eavesdropper**: Passive attacker on the network trying to intercept model updates or training data.
- **Insider**: Authorized user with elevated access who might abuse privileges to infer sensitive data.

## Asset Inventory
- **Source Code**: Proprietary code in training data. (Highly Sensitive)
- **Gradient Updates**: Per-sample gradients during training. (Sensitive)
- **Adapter Weights**: Trained LoRA adapter parameters. (Sensitive)
- **Model Weights**: Base model weights. (Public/Low Sensitivity)
- **Metadata**: Training metadata, configs, seeds. (Medium Sensitivity)

## Attack Surface Analysis
- **Gradient Inversion**: Reconstruct training data from gradients. (Severity: Critical, Likelihood: Medium)
- **Membership Inference**: Determine if a sample was in the training set. (Severity: High, Likelihood: High)
- **MITM Upload**: Intercept adapter uploads. (Severity: High, Likelihood: Medium)
- **Adapter Poisoning**: Upload malicious adapter to degrade the model. (Severity: High, Likelihood: Medium)
- **Model Extraction**: Steal the model via API queries. (Severity: Medium, Likelihood: Low)
- **Eavesdropping**: Passive interception of network traffic. (Severity: Critical, Likelihood: High)

## DP Unit Justification (Client-Level)
In CLASP, each client (edge node) represents one developer's private codebase. The privacy unit is the entire client's participation in a federated round — i.e., whether a given developer participated at all, not whether a specific code snippet was in the training set. 

This is because:
1. The adapter upload is the atomic unit of information leaving the client.
2. The cluster server sees one update per client per round.
3. Client-level DP matches the threat model where the adversary wants to determine if a specific developer's code influenced the model.

## Mitigation Matrix
| Threat | Mitigation | Mechanism |
|--------|------------|-----------|
| Eavesdropping | mTLS | Encrypts all traffic between edge and cluster |
| MITM Upload | mTLS | Mutual authentication prevents impersonation |
| Gradient Inversion | DP-SGD | Noise injection prevents exact data reconstruction |
| Membership Inference | DP-SGD | Bounded influence limits confidence of inference attacks |
| Adapter Poisoning | DP-SGD | Gradient clipping strictly bounds update magnitude |

## Residual Risks
- Data poisoning by malicious clients within the bounds of gradient clipping.
- Side-channel attacks on the client node itself.
- Compromise of the CA infrastructure used for mTLS.
