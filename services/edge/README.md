# Edge Layer — PEFT training, 4-bit inference, composite merger

**Owner:** Adithyaa (P1)

Depends on the shared `contracts` package; keep cross-module interaction
interface-driven.

```bash
pip install -e contracts -e services/evaluation -e services/edge
pytest services/edge/tests
```

## What the edge owns, and what it imports

The edge trains client adapters, composes `base + α·cluster + β·client` (D6),
serves and benchmarks the composite, and talks to the other services. It does
**not** carry its own copy of another module's logic; it imports it:

| Needs | Comes from | Used by |
|---|---|---|
| D2 SVD aggregation | `cluster.aggregation` (P2) | `edge.aggregate` (PEFT ⇄ `LoRAAdapter` bridge only) |
| D5 promotion rule | `registry.promotion.decide` (P4) | `edge.promote.predict_decision` (manifest prediction) |
| DP-SGD, ε accounting | `security.DPConfig`, `security.make_private`, `security.get_privacy_engine`, `security.PrivacyAccountant` (P3) | `edge.train_client --dp` |
| mTLS client context | `security.client_ssl_context` (P3; the edge raises the minimum to TLS 1.3) | `edge.transport.mtls_session` |
| completion metric | `evaluation.completion` (P5) | `edge.completion_eval` |
| wire types | `contracts` | `edge.wire.privacy_block`, `edge.promote` |

`cluster`, `registry` and `security` are imported lazily, inside the code path
that needs them, so the edge installs and its CPU test leg runs without them;
the tests that need one skip themselves when it is absent. Install them all
with the `integration` extra:

```bash
pip install -e services/cluster -e services/registry -e security -e "services/edge[integration]"
```

## Integration modules (four-seam loop)

| Module | Seam | What it does |
|---|---|---|
| `edge.wire` | A | PEFT ⇄ cluster key translation (both ways), fp32 wire contract, the `AdapterUpload` envelope incl. its `privacy` block, safetensors (de)serialization |
| `edge.transport` | A, C1, C2 | plain or mTLS (`security`) `requests` session; `post_upload` |
| `edge.registry_client` | C1 / C2 | pull `/active` + `/file`, verify sha256, materialize a loadable PEFT directory, `POST /promote` |
| `edge.completion_eval` | — | greedy next-line completion against a real model; scoring lives in `evaluation.completion` |
| `edge.promote` | C2 | `EvalResult` assembly, HumanEval guard resolution, the D5 call |

```bash
# pull a cluster adapter out of the registry and write a PEFT directory
python -m edge.registry_client pull cluster-web ./pulled/cluster-web

# in-project completion metric for one client
python -m edge.completion_eval --client web/client-flask --adapter artifacts/round1/client-flask/adapter
```

## Training

```bash
# base-only (round 1)
python -m edge.train_client --client web/client-flask --budget-plan budget_plan.json --lr 1e-3

# D3: train on the FROZEN base + alpha*cluster (cluster adapter pulled from the registry)
python -m edge.train_client --client web/client-flask --budget-plan budget_plan.json --lr 1e-3 \
    --cluster-adapter artifacts/round2_cluster/cluster-web --alpha 0.5

# D7: DP-SGD through P3's security library (needs the security branch integrated).
# Gradient checkpointing is off under DP (Opacus hooks), so use seq 512 on a 4 GB card.
python -m edge.train_client --client web/client-requests --dp --seq-len 512 \
    --dp-epsilon 8 --dp-delta 1e-5 --dp-batch-size 8
```

Evaluate a D3 round against the same cluster adapters the clients trained on:

```bash
python -m edge.round --adapters artifacts/round2_d3 --out artifacts/round2_d3_out \
    --cluster-adapters web=artifacts/round2_cluster/cluster-web \
                       scientific=artifacts/round2_cluster/cluster-scientific
```

The whole four-seam round is one command — see `docs/integration-sprint.md`:

```bash
python scripts/demo_round.py --round 1
```
