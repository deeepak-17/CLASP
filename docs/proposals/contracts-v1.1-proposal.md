# Proposal: contracts v1.0.0 → v1.1.0

> **Status:** DRAFT — awaiting all-hands sign-off (MASTER_PLAN D9: post-freeze
> contract changes need a semver bump + sign-off).
> **Author:** Deepak (P4). **Raised by:** W3 registry review.
> **Type:** additive, backward-compatible (minor bump). No existing field changes
> type or is removed, so v1.0 producers/consumers keep working unchanged.

## Why now

Two issues surfaced reviewing the W3 registry against the frozen v1.0 seam.
Both are cheaper to fix while the freeze is still young than after W8 forces a
breaking bump mid-phase.

1. **D6 composite adapters have no representation.** `AdapterKind` is only
   `{CLIENT, CLUSTER}`, and `AdapterMetadata` has no provenance linking a
   promoted composite back to the exact `(cluster_version, client_version, α, β)`
   it was built from. D6 requires storing `base ⊕ α·cluster ⊕ β·client` as one
   composite per developer at **promotion time (W8)**. Without a contract slot
   for it, W8 will force a post-freeze change to the seam — the exact event the
   freeze exists to avoid. This is a **known, scheduled** deliverable owned by
   P4, so we should reserve the fields now.

2. **`GuardMetrics.pass_at_k` breaks across JSON.** The field is typed
   `dict[int, float]`, but JSON has no integer keys: `{1: 0.31}` deserializes to
   `{"1": 0.31}`. The D5 promotion rule keys on **pass@1**, so a consumer doing
   `pass_at_k[1]` on a payload that crossed the wire raises `KeyError`. The
   current in-memory contract test never exercises the JSON path, so it passes
   while the real seam is unsafe.

## Proposed changes

### 1. Composite adapter kind + provenance (D6)

Add a third `AdapterKind` and a provenance value object; add one optional field
to `AdapterMetadata`. All additive.

```python
class AdapterKind(str, Enum):
    CLIENT = "client"
    CLUSTER = "cluster"
    COMPOSITE = "composite"   # NEW: pre-merged base ⊕ α·cluster ⊕ β·client (D6)


@dataclass(frozen=True)
class CompositeProvenance:
    """Exactly what a promoted composite was built from (D6, W8)."""
    cluster_name: str
    cluster_version: int
    client_name: str
    client_version: int
    alpha: float              # composition coefficient used at merge time
    beta: float
    base_model: str = "deepseek-ai/deepseek-coder-6.7b-base"


@dataclass(frozen=True)
class AdapterMetadata:
    ...                       # all existing fields unchanged
    composed_from: CompositeProvenance | None = None   # NEW: set iff kind == COMPOSITE
```

Rationale for pinning α/β *in provenance* as well as in `hparams`: the composite
is merged offline at promotion (D6), so the coefficients actually used must be
recorded on the immutable artifact for reproducibility (D9) even if a later
experiment config changes the defaults.

### 2. Safe pass@k serialization (D5)

Keep the ergonomic `dict[int, float]` in memory, but make crossing the wire
lossless with explicit (de)serializers and normalization. No field type change,
so this stays backward-compatible.

```python
@dataclass(frozen=True)
class GuardMetrics:
    benchmark: str
    pass_at_k: dict[int, float]

    def to_json(self) -> dict:
        return {"benchmark": self.benchmark,
                "pass_at_k": {str(k): v for k, v in self.pass_at_k.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "GuardMetrics":
        return cls(benchmark=d["benchmark"],
                   pass_at_k={int(k): v for k, v in d["pass_at_k"].items()})

    def pass_at(self, k: int) -> float | None:
        return self.pass_at_k.get(k)
```

The registry's promotion rule (W5) then reads `guard.pass_at(1)` rather than
indexing a raw dict, so the D5 "HumanEval pass@1 drop ≤ 2 pts" check is safe
regardless of how the payload was serialized.

## Required accompanying tests (land with the bump)

- Round-trip `to_json`/`from_json` for **all three wire seams**
  (`AdapterUpload`, `ClusterSnapshot`, `EvalResult`/`GuardMetrics`) — the review
  found these have zero serialization coverage today.
- `pass_at_k` survives a `json.dumps` → `json.loads` → `from_json` cycle with
  integer-key access intact.
- `AdapterMetadata` round-trips `composed_from` for a `COMPOSITE` adapter.

## Compatibility / migration

- **Backward compatible.** New enum value + new optional fields; v1.0 payloads
  deserialize unchanged (`composed_from` defaults to `None`).
- Registry storage already tolerates unknown/None metadata fields; the W3
  `_metadata_from_dict` needs a small additive branch for `composed_from`.
- Bump `CONTRACTS_VERSION` and package version to `1.1.0`; update the freeze
  test (`test_contracts.py`) to assert `1.1.0`.
- No change required in edge/cluster/eval until they actually emit composites
  (W8) or read pass@k over the wire (W5).

## Owners impacted (sign-off needed)

| Module | Impact | Sign-off |
|---|---|---|
| P4 Registry | implements COMPOSITE storage + provenance (W8), consumes pass@k in promotion (W5) | ☐ Deepak |
| P1 Edge | builds the composite at promotion; emits `CompositeProvenance` | ☐ Adithyaa |
| P5 Eval | emits `GuardMetrics` via `to_json`; is the pass@k producer | ☐ Aditya |
| P2 Cluster | no change (cluster snapshots already covered) | ☐ Prasanth |
| P3 Security | no change (ε already in `PrivacySpec`) | ☐ Kapilan |

## Decision

- [ ] Approved as v1.1.0 — P4 implements before W8 (composite) / W5 (pass@k).
- [ ] Deferred — accept a breaking bump at W8 instead (not recommended).
- [ ] Rejected — record rationale here.
