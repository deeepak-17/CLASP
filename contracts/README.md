# contracts — CLASP shared interfaces

The integration seam of the monorepo. Every service installs this package and
communicates through the types defined here, rather than importing each other
directly. Changes here are high-impact: flag them and keep them
backward-compatible where possible.

```bash
pip install -e contracts
```

## Versions

- **1.0.0** — the three wire seams (`AdapterUpload`, `ClusterSnapshot`,
  `EvalResult`), `AdapterMetadata`, `RunManifest`, LoRA/privacy value objects.
- **1.1.0** (additive, backward-compatible) — `AdapterKind.COMPOSITE` and
  `CompositeProvenance` on `AdapterMetadata.composed_from` for pre-merged
  composites (D6); `to_json()` / `from_json()` on every value object, accepting
  v1.0 payloads unchanged; `GuardMetrics` normalizes pass@k keys to `int` and
  validates them, so `pass_at_k[1]` survives a JSON hop.

Use `to_json` / `from_json` on both sides of an HTTP hop rather than
`dataclasses.asdict` + `json.loads`: JSON has no integer keys and no tuples.

