# CLASP-P5 · Partitioning (Project-Level and Per-Developer)

Reference doc for Week 2 (project-level, frozen) and Week 4 (per-developer). See `partitions/strategies.py` for the implementation and full rationale docstrings; this is the shorter operational summary.

## 1. Two strategies, two configs, one pipeline

| | Project-level (Week 2) | Per-developer (Week 4) |
|---|---|---|
| Config | `configs/partition.yaml` | `configs/partition_per_developer.yaml` |
| Manifest | `datasets/partitions/manifest.json` | `datasets/partitions/per_developer/manifest.json` |
| Clients (real D1 corpus) | 6 (one per project) | 20 |
| Client naming | `client-{cluster_id}` | `client-{cluster_id}-dev{dev_index}` |

Same `Partitioner`, `PartitionValidator`, `scripts/build_partitions.py` and `scripts/validate_partitions.py` — only `--config` differs. Both are pure functions of `(datasets/processed/corpus.jsonl, seed=20260616, strategy)`, so both are exactly reproducible and directly comparable.

## 2. What "developer" means here

D1's source repositories are shallow-cloned (`configs/dataset.yaml`'s `acquisition.git_depth: 1`) for reproducible, size-bounded acquisition. That has a consequence worth stating plainly: **every file in every project resolves to the same single commit**, verified directly —

```
$ cd datasets/raw/flask && git log --oneline | wc -l
1
$ git log -1 --format='%an' -- src/flask/app.py
David Lord
```

— so there is no real per-file authorship signal anywhere in this corpus to split on. Per the Week-4 instruction not to fabricate developer identities, `PerDeveloperStrategy` uses only structure the corpus genuinely carries, with three bases, tried in order and falling back when the previous one can't clear `per_developer.min_files_per_developer`:

1. **`module_path`** — a project's real subpackages (Flask: `json/`, `sansio/`, root; Werkzeug: `routing/`, `debug/`, `middleware/`, `datastructures/`, `sansio/`, root) each become one developer's slice. Module ownership is a legitimate, commonly used proxy for contribution boundaries in real engineering orgs, and unlike authorship it is directly present in `CorpusRecord.relative_path`.
2. **`deterministic_hash_chunk`** — for projects with no subpackages at all (Requests, Click, Colorama, and — despite its size — Jinja2 are all flat file lists in D1), files are ordered by `sha256(seed:cluster_id:file_id)` and cut into contiguous, roughly-equal chunks. Hashing before chunking avoids grouping files by alphabetic prefix, which would be no more meaningful a boundary than the flat structure it's replacing.
3. **`single_developer_whole_project`** — when a project is too small to split even once (Colorama's 5 files), the honest answer is one developer, identical to the project-level shard.

`datasets/partitions/per_developer/developer_basis.json` records which basis (and, for `module_path`, which subpackage) produced every client shard — traceability that deliberately lives outside the frozen `PartitionManifest` contract rather than inside it.

## 3. Leakage validation

`PartitionValidator` (`partitions/validation.py`) is unmodified for Week 4 — its checks (disjointness by `file_id`, content-disjointness, project purity, coverage, minimum shard size) are manifest-wide, so they already prove no-cross-project-leakage regardless of how many clients one project is split into. Running it against the per-developer manifest is the actual Week-4-Tuesday leakage check:

```bash
python scripts/validate_partitions.py --config configs/partition_per_developer.yaml
```

Result on the real corpus: **PASS, 9/9 checks, 0 errors, 0 warnings.** A machine-readable sibling (`reports/partition_validation_report_per_developer.json`) is written alongside the Markdown report for both configs.

## 4. Client-count cross-check and comparison

```bash
python scripts/cross_check_client_counts.py         # -> reports/client_count_cross_check.md
python scripts/generate_partition_comparison.py      # -> reports/project_vs_developer_comparison.md
```

Neither script changes P5's partition strategy to make counts agree with P2's current simulator width — a mismatch is reported as a coordination finding (matching the Week-2 precedent already in this repository's README) for integration week to resolve, not something P5 decides unilaterally.

## Reproducing everything in this document

```bash
python scripts/build_partitions.py                                          # Week-2, unchanged
python scripts/build_partitions.py --config configs/partition_per_developer.yaml
python scripts/validate_partitions.py
python scripts/validate_partitions.py --config configs/partition_per_developer.yaml
python scripts/cross_check_client_counts.py
python scripts/generate_partition_comparison.py
pytest tests/test_partitions.py tests/test_partitions_per_developer.py -v
```
