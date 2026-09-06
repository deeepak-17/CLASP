# CLASP-P5 · Week 4 Completion Checklist

**Scope:** P5 Week-4 daily targets. The Week-2 project-level partition remains fully intact — Week 4 *adds* a sibling strategy and sibling config, touching zero Week-2 output paths (see `PartitionOutputConfig.report_suffix` for how the shared build/validate scripts avoid clobbering Week-2's reports when reused for Week 4).

## Monday: Build per-developer (individual-style) partition logic
- [x] `partitions/strategies.py::PerDeveloperStrategy` — real implementation (previously a Week-2 placeholder that raised on purpose)
- [x] **What "developer" means here, stated plainly**: this corpus's git history is a depth-1 shallow clone (`configs/dataset.yaml`), verified directly — every file in every one of D1's 6 projects resolves to the *same single commit*. There is no real per-file authorship signal. No developer identity is fabricated.
- [x] Basis 1 — **module_path**: where a project has real subpackages (Flask: `json/`, `sansio/`, root; Werkzeug: `routing/`, `debug/`, `middleware/`, `datastructures/`, `sansio/`, root), each subpackage is one developer's slice — module ownership is a legitimate, common proxy for contribution boundaries, and it is directly present in the corpus (`relative_path`), unlike authorship.
- [x] Basis 2 — **deterministic_hash_chunk**: for flat projects (Requests, Click, Colorama, Jinja2 — no subpackages at all in D1), files are ordered by `sha256(seed:cluster_id:file_id)` and cut into contiguous, roughly-equal chunks. Deterministic and seed-reproducible; explicitly not a real-identity claim.
- [x] Basis 3 — **single_developer_whole_project**: falls back to one developer (== the project-level shard) when a project is too small to split further even once (Colorama's 5 files, in D1).
- [x] `configs/partition_per_developer.yaml` — sibling config; `partition.per_developer` block documents the above
- [x] Generated (real data): **20 clients across 6 clusters** — module_path: 10 shards, deterministic_hash_chunk: 9 shards, single_developer_whole_project: 1 shard (Colorama)

## Tuesday: Validate per-developer partitions don't leak across projects
- [x] `scripts/validate_partitions.py --config configs/partition_per_developer.yaml` — the identical Week-2 `PartitionValidator` (disjointness, content-disjointness, project purity, coverage, min shard size, balance), reused unmodified because its checks are manifest-wide and therefore already prove no-cross-project-leakage regardless of how many clients one project is split into
- [x] Result: **PASS, 9/9 checks, 0 errors, 0 warnings** — 135/135 files covered exactly once, disjoint by file_id, no identical content shared, project purity 1.0
- [x] Machine-readable output: `reports/partition_validation_report_per_developer.json` (added for both configs — Week-2's `partition_validation_report.json` now also exists)

## Wednesday: Cross-check both partition types against P2's client count
- [x] `scripts/cross_check_client_counts.py` — reuses `interfaces.cluster_client.MockClusterClient` (the same admission rules Week 2's contract-compliance gate already applies)
- [x] Result, both reported as **warnings** (matching the Week-2 precedent for this exact situation), not failures:
  - project_level: **6 actual vs 3 P2-expected**
  - per_developer: **20 actual vs 3 P2-expected**
- [x] `reports/client_count_cross_check.md` — explicit "what needs to be decided during integration" section; P5's strategy was **not** changed to force agreement

## Thursday: Report comparing project-level vs per-dev splits
- [x] `scripts/generate_partition_comparison.py` — reads the *actual* generated manifests, metadata and validation JSON for both strategies (no invented statistics; a missing artefact is reported as "not available", not silently skipped)
- [x] `reports/project_vs_developer_comparison.md` — headline numbers, coverage/leakage results, per-cluster project distribution, developer-basis breakdown, advantages/limitations, reproducibility, integration implications

## Friday: Hand off partitions to Adithyaa/Prasanth for Week-7 use
- [x] Both manifests are contract-valid (`interfaces/schemas/partition_manifest.schema.json`) and sit at stable, documented paths:
  - `datasets/partitions/manifest.json` (project-level, Week 2, unchanged)
  - `datasets/partitions/per_developer/manifest.json` (per-developer, Week 4)
- [x] Both are reproducible from `datasets/processed/corpus.jsonl` with a committed seed (`20260616`) via `python scripts/build_partitions.py --config <config>`
- [x] README/docs updated so P1/P2 can find both without asking

## Cross-cutting Week-4 deliverables
- [x] `partitions/models.py::PerDeveloperConfig` — new config section, additive (`PartitionConfig.per_developer` defaults so Week-2 configs need no change)
- [x] `partitions/strategies.py::build_strategy` — signature extended with an optional `per_developer` parameter; every existing call site and test still compiles unchanged
- [x] `partitions/partitioner.py::PartitionResult.assignments` — new optional field carrying the strategy's own `ClientAssignment` objects (with `developer_basis`/`developer_label`) for traceability, without touching the frozen `PartitionShard`/`PartitionManifest` contract
- [x] `tests/test_partitions_per_developer.py` — 25 new tests: module-path grouping, hash-chunk fallback (determinism across runs, difference across seeds, not-alphabetic-slicing), single-developer fallback, min-size enforcement, no-cross-project-leakage, config validation, end-to-end build+validate
- [x] Two Week-2 placeholder tests updated (`PerDeveloperStrategy` "not implemented", `build_strategy` registry) — same pattern as the Week-3 seam-test updates, documented inline
- [x] Full suite passing (see final report for the exact count)

**Week 4 status: COMPLETE.**

**Genuine limitation, stated once more for the record:** neither `module_path` nor `deterministic_hash_chunk` is a measurement of real individual developer contribution. They are the most defensible splits the corpus's actual metadata supports, chosen specifically because the alternative — treating the shallow clone's single commit author as "the developer" — would have been a fabricated identity dressed up as a real one.
