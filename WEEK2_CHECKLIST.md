# CLASP-P5 · Week 2 Completion Checklist

## Monday: Partition Strategy
- [x] `partitions/strategies.py` — PartitionStrategy base + ProjectLevelStrategy
- [x] Non-IID by project: one shard per source repository
- [x] Client naming convention: `client-{cluster_id}`
- [x] Deterministic, canonical ordering (by file_id)
- [x] PerDeveloperStrategy declared (not implemented until Week 4)

## Tuesday: Partition Build
- [x] `partitions/models.py` — PartitionConfig schema
- [x] `partitions/partitioner.py` — Partitioner
- [x] `configs/partition.yaml` — partition configuration
- [x] `scripts/build_partitions.py` — CLI
- [x] Write one JSONL shard per client
- [x] Write contract manifest: `datasets/partitions/manifest.json`
- [x] Shard content hashes (stable, deterministic)
- [x] Dry-run mode

## Wednesday: Partition Validation
- [x] `partitions/validation.py` — PartitionValidator
- [x] `scripts/validate_partitions.py` — CLI
- [x] Hard constraints (errors):
  - [x] Disjointness (no file_id in two shards)
  - [x] Content disjointness (no identical file content across shards)
  - [x] Full coverage (every corpus file assigned exactly once)
  - [x] Project purity (every file in a shard shares its cluster_id)
  - [x] Minimum shard size (no near-empty Flower clients)
- [x] Balance metrics (warnings):
  - [x] Imbalance ratio (max/min files)
  - [x] Gini coefficient
  - [x] Coefficient of variation
- [x] `fail_on_warning` config for CI enforcement
- [x] Report: `reports/partition_validation_report.md`

## Thursday: Partition Metadata
- [x] `partitions/metadata.py` — descriptive metadata builder
- [x] Per-cluster aggregates (files, lines, bytes, SLOC/file)
- [x] Licence roll-up
- [x] Corpus share percentages
- [x] Source provenance (refs, commit SHAs)
- [x] Artefact: `datasets/partitions/partition_metadata.json`
- [x] Report: `reports/partition_metadata.md`

## Friday: Contract Compliance
- [x] `interfaces/cluster_client.py` — P2 protocol + MockClusterClient
- [x] `interfaces/registry_client.py` — P4 protocol + MockRegistryClient
- [x] `interfaces/validation.py` — JSON Schema + semantic validation
- [x] `interfaces/schemas/*.json` — contract schemas (4 total)
- [x] `scripts/check_contract_compliance.py` — five-check gate
  - [x] Check 1: partition manifest schema validation
  - [x] Check 2: provenance (corpus SHA-256 match, shards present)
  - [x] Check 3: P2 acceptance (MockClusterClient admission rules)
  - [x] Check 4: P4 read path (snapshot round-trip)
  - [x] Check 5: P5 → P4 (EvalResult schema validation)
- [x] Report: `reports/contract_compliance_report.md`
- [x] **Integration finding**: surfaced P2 client-count coordination issue (warning)

## Cross-cutting Week-2 Deliverables
- [x] Tests: partitions, interfaces (163 additional tests)
- [x] Full pipeline reproducibility (synthetic mode + real git mode)
- [x] Re-serialisation stability checks (contract round-trips)
- [x] Total test suite: 256 passing

**Week 2 status: COMPLETE**

**Explicitly deferred to Week 3+:**
- Pass@k scoring (`evaluation.scoring`)
- Sandboxed execution policy (requires P3 sign-off)
- Wiring to P1's real Edge Layer (merged model)
- Dashboard (React + Recharts)
- Per-developer partitioning (Week 4)
