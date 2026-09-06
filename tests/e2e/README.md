# Cross-service E2E loop (P4)

One test, `test_full_loop.py`, chaining Cluster's and the State Registry's real
FastAPI apps together over their own `TestClient`s — no real network, no GPU,
no trained weights. It proves the three services agree with each other over
the actual HTTP contract, which no single service's own test suite can see on
its own:

```
Edge (synthetic)  --A--> Cluster  --B--> Registry  --C1--> (pulled back)
                                            \--C2--> promote / rollback
```

- **A** `POST /uploads` — three tiny synthetic client adapters
  (`cluster.adapter_format.random_adapter`) upload to Cluster.
- **B** `POST /adapters/{name}/versions` — Cluster's real `aggregate_svd`
  output, serialized to a real safetensors blob, saved to the Registry.
- **C1** `GET .../active` + `.../file` — pulled back out and round-tripped;
  also reassembles a loadable PEFT `adapter_config.json` from just the two
  responses (Integration Sprint Defect 4).
- **C2** `POST .../promote` — both PROMOTE and ROLLBACK exercised over the
  real seam, not just asserted against `registry.promotion.decide` in
  isolation.

Edge itself isn't imported (it needs torch/transformers/peft); a synthetic
adapter stands in, the same way Cluster's own HTTP tests already do. Real
aggregation math and real promotion logic run unchanged — nothing here is
reimplemented, only wired together.

Run: `pytest tests/e2e -q` (needs `contracts`, `services/cluster`,
`services/registry` installed, plus `safetensors` — see the `e2e` job in
`.github/workflows/ci.yml`).
