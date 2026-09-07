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
  also checks how much of a loadable PEFT `adapter_config.json` the
  registry's stored metadata can reassemble, against cluster's own real
  `peft_config` (Integration Sprint Defect 4). **Partially open, not
  closed**: `contracts.LoRAHyperParams` has no field for PEFT's structural
  config (`peft_type`, `use_rslora`, `use_dora`, `fan_in_fan_out`,
  `lora_bias`) or `base_model_name_or_path`, and `lora_dropout` doesn't
  round-trip either — the test asserts these gaps explicitly rather than
  hiding them behind a hand-rolled config that happens to look complete.
- **C2** `POST .../promote` — both PROMOTE and ROLLBACK exercised over the
  real seam, and ROLLBACK is checked against the actual restored bytes (a
  distinguishable v2), not only the reported active version number.

Edge itself isn't imported (it needs torch/transformers/peft); a synthetic
adapter stands in, the same way Cluster's own HTTP tests already do. Real
aggregation math and real promotion logic run unchanged — nothing here is
reimplemented, only wired together.

Run: `pytest tests/e2e -q` (needs `contracts`, `services/cluster`,
`services/registry` installed, plus `safetensors` — see the `e2e` job in
`.github/workflows/ci.yml`).
