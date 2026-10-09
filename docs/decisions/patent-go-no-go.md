# Decision record — patent go / no-go (D10)

| | |
|---|---|
| **Status** | **OPEN — awaiting the team and the guide** |
| **Decides** | whether to file a disclosure with the university TTO |
| **If GO** | disclosure starts at the beginning of Phase III, before any paper submission or public repo release |
| **If NO-GO** | nothing to file; the Phase III submission and public release proceed |
| **Recorded by** | Deepak (P4) |

Publication (paper, public repo, public demo) counts as prior art against a later
patent filing. That is why this record must close before the venue gate.

## Candidate claims — what is plausibly novel, stated narrowly

1. **Three-layer composite personalization served as a single adapter.** The
   `α·ΔW_cluster + β·ΔW_client` composite is merged exactly by rank
   concatenation and stored with immutable provenance. It is built at
   promotion time under a two-sided gate, so serving never stacks adapters at
   runtime. (`services/registry/src/registry/composite.py`, `composition.py`)
2. **Two-sided promotion with automatic rollback for federated adapters.** An
   adapter is promoted only if in-project quality beats a measured noise band
   *and* a general-benchmark guard holds; otherwise the registry repoints to
   the previous version. Everything is recorded in an append-only audit trail.
   (`promotion.py`)
3. **Dynamic re-clustering of federated clients on the cosine of their LoRA
   updates**, with exact low-rank SVD aggregation per cluster. (P2's cluster layer)

## Questions the decision has to answer

- [ ] Prior-art search (Google Patents, Lens) for each claim above — owner, date.
- [ ] Is any claim more than a combination of known techniques (federated LoRA,
      FedProx, SVD aggregation, model registries with rollback)?
- [ ] What has been disclosed publicly already? (Panel reviews are internal; the
      GitHub repo's visibility must be checked.)
- [ ] TTO process, timeline and cost, as confirmed with the guide.

## Decision

- [ ] GO — claims: ______ · disclosure owner: ______ · date: ______
- [ ] NO-GO — reason: ______ · date: ______

Signed off by: ☐ Adithyaa ☐ Prasanth ☐ Kapilan ☐ Deepak ☐ Aditya ☐ Dr. Swapna T R
