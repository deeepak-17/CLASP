# P2 Cluster — integration boundaries

Scope rule: this document describes what **Cluster expects** from the other modules, what **exists in
the repository today**, whether the interfaces **match**, what P2 can **mock**, and what stays
**blocked**. Nothing here implements another team's work; the only code added on P2's side is
`cluster/tls.py`, `cluster/integration.py` and the identity / sink hooks in `cluster/server.py`
(all optional, off by default). Repository facts below were read from the working tree on
2026-10-06; `security/`, `services/registry` and `services/evaluation` contain only an empty package
`__init__.py` (234-282 bytes each).

## Summary

| Seam | Cluster side | Other side | Status |
|---|---|---|---|
| Edge -> Cluster adapter upload (A) | `POST /clusters/{id}/uploads`, `POST /uploads` — **done, tested** | no upload client in `services/edge` | **BLOCKED BY P1** (live) — mockable, mocked in tests |
| LoRA hyper-parameters | defaults r=16, alpha=16, q/k/v/o — **aligned in Week 6** | Edge pins r=16, alpha=16, q/k/v/o | **MATCH** |
| Cluster -> Edge adapter retrieval | `GET /clusters/{id}/adapters/active` (PEFT keys + `peft_config`) — **done, tested** | no download client in Edge | **BLOCKED BY P1** (live) |
| Client identity | self-asserted `client_id`; `configure_identity()` hook — **done, tested with a stub** | none | **BLOCKED BY P3** (real identity) |
| mTLS (G1) | `cluster/tls.py`, tested against an *ephemeral test CA* | `security/` is empty | **BLOCKED BY P3** |
| DP mu tuning (W9) | toy non-DP mu sweep only | `security/` is empty | **BLOCKED BY P3** |
| Cluster -> Registry snapshot | `SnapshotSink` + `AdapterRef` mapping + in-memory stub — **done, tested** | `services/registry` is empty | **BLOCKED BY P4** (live) |
| Evaluation | none needed by Cluster (see below) | `services/evaluation` is empty | **BLOCKED BY P5** (eval of cluster adapters) |

## P1 Edge

* **Cluster expects**: an upload per client per round (`AdapterUpload`, `cluster/schemas/messages.py`):
  `client_id`, `round_id` (must equal the cluster's current round), `rank`, `alpha`,
  `target_modules`, `num_layers`, `num_examples`, optional `base_cluster_id`, and
  `2 * len(target_modules) * num_layers` tensors named either `layers.<i>.<module>.lora_{A,B}.weight`
  or a full PEFT key `base_model.model.model.layers.<i>.self_attn.<module>.lora_{A,B}.weight`.
  Tensors are base64 + dtype + shape (`TensorPayload`).
* **Exists**: `edge/lora_init.py` (`attach_lora`; contract comment r=16, lora_alpha=16, q/k/v/o,
  base model `deepseek-ai/deepseek-coder-1.3b-base`; Edge `config.py` also defines a `target`
  profile `deepseek-coder-6.7b-base`) and `edge/toy_training_loop.py` (`attach_lora(model, r=16, ...)`,
  i.e. alpha 16 via the default). No adapter-upload or adapter-download client; `edge.merge`
  (referenced by `tests/test_peft_interop.py`, whose constants were copied from the `origin/services/edge`
  branch) is **not** in this working tree.
* **Match**: yes for rank, alpha, target modules and PEFT key layout (`test_peft_interop.py`).
  **The previous mismatch (Edge alpha 16 vs Cluster default 32) is fixed on the Cluster side**
  (`DEFAULT_ALPHA = 16.0`, see `WEEK6_REVIEW.md` #1). Aggregation now *rejects* a mixed-alpha round
  instead of averaging it (422 at upload time over HTTP). `base_model_name_or_path` in the broadcast
  config defaults to the 1.3B dev model; pass the real one if Edge trains the 6.7B `target` profile.
* **Mockable**: yes — every Cluster test uploads synthetic adapters through the real HTTP handlers
  (`tests/test_server_*.py`) or drives real FedProx toy clients in-process (`federation.py`).
* **Blocked**: a real Edge client posting a real trained adapter and pulling the cluster adapter back.
  Not verified: that a real Edge-trained adapter loads through `from_state_dict` (only PEFT-shaped
  synthetic adapters and the key convention were tested).
* **Reported to P1, not changed**: stray first line in `edge/lora_init.py`
  (`from peft.tuners.lora.corda import target_modules`).

## P3 Security

* **Cluster expects** (design intent from the root README: "security is a library imported by the
  edge and cluster layers"): TLS material for a mutually authenticated channel (CA, server cert/key,
  client cert/key), a way to map a verified client certificate to a `client_id`, and, for DP, an
  accountant that fills `ClusterAdapterBroadcast.epsilon`.
* **Exists**: nothing — `security/src/security/__init__.py` is an empty package. `security/README.md`
  and `pyproject.toml` exist.
* **P2-side infrastructure built** (and the only thing that can honestly be built):
  * `cluster/tls.py` — `MTLSFiles`, `uvicorn_ssl_kwargs()`, `client_ssl_context()`,
    `flower_certificates()`, `common_name_from_peercert()`.
  * `server.configure_identity(provider, require=...)` — an upload whose `client_id` differs from the
    authenticated identity is refused (403), a missing identity is refused (401) when required, a
    failing provider fails closed (401); a member of cluster A may not read cluster B's adapter (403).
  * `tests/test_tls_p2_side.py` — starts the real HTTP app under uvicorn with `uvicorn_ssl_kwargs`
    using certificates from an **ephemeral CA generated by the test**: a client with a certificate from
    that CA is served; a client with no certificate, and a client whose certificate comes from another CA,
    are refused. `tests/test_server_identity.py` — the identity hook with a stub provider.

### G1 (Edge <-> Cluster live over mTLS): **BLOCKED BY P3** (and P1)

G1 has **not** passed and is not claimed. What was verified is only that the Cluster HTTP server enforces
client-certificate TLS when it is *given* certificates. To close G1:

1. P3 provides the CA / certificate issuance (and the `security` API Cluster should import) — Cluster has
   nothing to import today.
2. Decide how a verified certificate becomes a `client_id`. uvicorn verifies the client certificate during
   the handshake but **does not expose the peer certificate to the ASGI app**, so either a TLS-terminating
   proxy relays the verified subject (install a provider with `configure_identity`) or the service moves to
   a server that exposes it (`common_name_from_peercert` is ready for that case).
3. P1 provides an upload/download client that presents its certificate.
4. Run the Edge -> Cluster -> Edge round trip (up, aggregate, down) over that channel, plus the
   bad-certificate rejection, and record it.
5. Flower gRPC: `flower_certificates()` only builds the tuple Flower's `certificates=` expects.
   `start_grpc_server` still starts an **insecure** channel; a Flower mTLS round has never been run here.

### DP mu tuning (W9 Wed, "FedProx mu tuning for DP stability"): **BLOCKED BY P3**

There is no DP-SGD code in the repository, so tuning mu *for DP stability* cannot be done and no result
is claimed. What exists, clearly separate: a **toy, non-DP** FedProx mu sweep in
`python -m cluster.evidence` (section `mu_sweep`, results in `demo_runs/phase2_cluster_evidence.md`). It
shows only that on that toy loss rises as mu grows (more drag toward the global adapter) and that the
proximal step diverges once `lr * mu >= 2`. It says nothing about the optimum under DP noise/clipping.
Once P3's clipped + noised gradients exist: rerun the sweep through the same `evidence.mu_sweep` harness with
DP on, and record the chosen mu with its epsilon.

## P4 Registry

* **Cluster expects**: somewhere to publish each aggregated cluster adapter after a round: a
  versioned adapter named per cluster, with the PEFT tensors and `adapter_config` (and later epsilon).
* **Exists**: `contracts.types.AdapterRef(name, version, kind, cluster_id)` and `AdapterKind.CLUSTER`;
  `services/registry` is an empty package (docker-compose declares a registry service on :8004).
* **Interface**: Cluster defines `cluster.integration.SnapshotSink.publish(ref, broadcast)` and maps
  round `r` of cluster `c` to `AdapterRef(name=c, version=r+1, kind=CLUSTER, cluster_id=c)`. **This naming is
  Cluster's proposal** — P4 has not specified one. It is wired into `MultiClusterFederation(sink=...)` and
  `server.configure_snapshot_sink(...)`; a failing sink is reported (`publish_error`) and never loses a round.
* **Mockable**: yes — `InMemorySnapshotSink` (a test double, not a registry), `tests/test_integration_stubs.py`.
* **Blocked**: publishing to a real registry; the wire format of the snapshot (the registry may want
  safetensors + metadata rather than `ClusterAdapterBroadcast`); the Registry -> Evaluation extended metadata
  list (`rank, target_modules, alpha, beta, epsilon, seed, timestamp`) is only partly covered by the
  broadcast (no beta/seed/timestamp mapping to registry metadata exists to test against).
* Observation: the wire messages (`AdapterUpload`, `ClusterAdapterBroadcast`) live in the cluster
  package, not in the shared `contracts` package, which only has `AdapterRef`, `AdapterKind`, `EvalResult`.

## P5 Evaluation

* **Cluster expects**: nothing at runtime. Cluster does not call an evaluation endpoint; Evaluation is
  meant to consume adapters from the registry (and drive promote/rollback). There is therefore no
  "evaluation endpoint" on the Cluster side to implement or mock.
* **Exists**: `contracts.types.EvalResult(adapter: AdapterRef, benchmark, pass_at_k)`;
  `services/evaluation` is an empty package.
* **What Cluster offers an evaluator** today: `GET /clusters/{id}/adapters/active` (PEFT-keyed tensors +
  a ready `adapter_config`) and `GET /clusters/{id}/aggregate/manifest`; the same adapter is what the
  registry sink receives.
* **Blocked**: evaluating a cluster adapter (pass@k), and anything that feeds an eval result back to
  Cluster (Cluster has no such input).

## What Cluster does *not* depend on
`contracts` is used only for `AdapterRef`/`AdapterKind` in `cluster/integration.py`. Nothing in Cluster
imports Edge, Security, Registry or Evaluation code.
