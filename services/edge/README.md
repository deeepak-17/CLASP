# Edge Layer — PEFT training, 4-bit inference, dynamic merger

**Owner:** Adithyaa (P1)

Skeleton module. See the team's internal conventions notes (local). Depends on
the shared `contracts` package; keep cross-module interaction interface-driven.

```bash
pip install -e contracts -e services/evaluation -e services/edge
pytest services/edge/tests
```

## Integration modules (four-seam loop)

| Module | Seam | What it does |
|---|---|---|
| `edge.wire` | A | PEFT ⇄ cluster key translation (both ways), fp32 wire contract, the `AdapterUpload` envelope, safetensors (de)serialization |
| `edge.registry_client` | C1 / C2 | pull `/active` + `/file`, verify sha256, materialize a loadable PEFT directory, `POST /promote` |
| `edge.completion_eval` | — | greedy next-line completion against a real model; scoring lives in `evaluation.completion` |
| `edge.promote` | C2 | `EvalResult` assembly, HumanEval guard resolution, the D5 call |

```bash
# pull the live cluster adapter out of the registry and write a PEFT directory
python -m edge.registry_client pull cluster-web ./pulled/cluster-web

# in-project completion metric for one client
python -m edge.completion_eval --client web/client-flask     --adapter artifacts/round1/client-flask/adapter
```

The whole round is one command — see `docs/integration-sprint.md`:

```bash
python scripts/demo_round.py --round 1
```
