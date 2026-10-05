# DevOps, orchestration and reproducibility

**Owner:** Deepak (P4). How CLASP is built, tested, deployed, secured and
reproduced — and the measured evidence behind each claim.

## 1. Build and deploy: one compose file, three profiles

| Command | Brings up |
|---|---|
| `docker compose up -d` | `registry` (:8004, stateful) + `cluster` (:8002). The cluster waits for a **healthy** registry. |
| `docker compose --profile demo up` | + `demo-ui` (:8010) and `demo-seed`, a one-shot job that drives the registry end to end and writes `registry-demo.json` to the `demo-reports` volume |
| `docker compose --profile train up edge` | + the edge federated round (`edge.round`) on the NVIDIA runtime, with trained adapters and corpus mounted |
| `-f docker-compose.yml -f docker-compose.mtls.yml` | the registry behind mutual TLS (§4) |

`security/` is a library, never a container. Every image is `python:3.11-slim`
with `contracts` installed first: the contracts package is the only thing
services share.

**State (D9).** Only the registry is stateful. Its named volume `registry-data`
survives `docker compose down` and `up`. Verified by saving four adapters, cycling
the stack, and listing the same four afterwards. A version is a directory
published by one atomic rename. Blobs are never overwritten. The `active` pointer
is replaced atomically. Decisions and GC runs go to append-only JSONL.

## 2. Continuous integration

`.github/workflows/ci.yml` runs on every push and PR to `main` and to
`integration/**`, which is where team PRs land first:

| Job | What it proves |
|---|---|
| `lint` | `ruff check .` over the whole repo |
| `test` (6-package matrix) | contracts, security, edge, cluster, registry, evaluation each install and pass alone; the registry enforces **≥ 90 % coverage** |
| `e2e` | cluster → registry loop over the real apps |
| `integration` | all four seams (edge → cluster → registry → edge → registry), CPU only, plus registry-vs-edge composite equivalence |
| `experiments` | every `experiments/*/config.yaml` has a seed, a GPU-hours estimate, stays in the D8 caps and fits its budget |
| `compose` | every profile and the mTLS overlay parse; the real registry image + volume come up and `demo-seed` passes every step |

**Fresh machine.** `scripts/fresh_machine_check.sh [ref]` clones the commit into an
empty directory, so nothing untracked exists. It then installs and tests inside a
bare `python:3.11-slim`: contracts, the registry suite with its coverage floor,
the config check, and building and verifying a repro pack. Finally it builds and
runs the compose demo from that clone. It passes at the commit that introduced it.

## 3. Registry lifecycle (what `demo-seed` asserts)

1. Save a cluster adapter and a client adapter: immutable, sha256 per payload, ε recorded.
2. Compose them into one pre-merged composite. Exact, rank r_c + r_l, provenance stored (D6).
3. Save a better client v2 and promote it: D5 says PROMOTE, and the composite is
   rebuilt from v2 in the same call.
4. Save a worse client v3 and promote it: D5 says ROLLBACK (edit-similarity delta
   0.01 below the 0.02 band; pass@1 dropped 0.10), so `active` returns to v2.
5. Run a restore drill on a 24-layer, rank-16 adapter, timed (§5).
6. Read the lineage (composite v1 ← cluster@v1 + client@v1; composite v2 ←
   cluster@v1 + client@v2), the audit trail (`promote`, `rollback`) and a GC dry run.

## 4. Security of the registry endpoint (D7)

`python -m registry.serve` serves plain HTTP unless `CLASP_TLS_CERT`,
`CLASP_TLS_KEY` and `CLASP_TLS_CA` are all set. With them, every connection must
present a client certificate signed by the CLASP CA and negotiate TLS 1.3. That
is the same policy as the security library's server context. A partial TLS
config refuses to start; it never falls back to HTTP silently.

Verified in Docker with the overlay and `scripts/make_dev_certs.sh`:

| Client | Result |
|---|---|
| CA-signed client cert | `200 {"status":"ok"}` |
| no client cert | connection refused (curl exit 52) |
| plain HTTP | refused (curl exit 52) |
| TLS 1.2 forced | handshake refused (curl exit 35) |

The container healthcheck keeps reporting `healthy` under mTLS. **Scope:** this
hardens the registry endpoint. Its callers must present client certificates
before they can use it under the overlay. Wiring those clients is the security
library's job, so the plain demo round runs without the overlay.

## 5. Non-functional requirements (D11)

| NFR | Budget | Measured | Where |
|---|---|---|---|
| Registry restore-to-previous | ≤ 10 s | **0.019–0.026 s** in Docker (24 MB adapter: 24 layers × q/k/v/o, r = 16, hidden 2048), restore + fetch metadata + fetch payload; 0.24 s in-process in CI | `registry.demo` step `restore_nfr`; `tests/test_restore.py` |
| One federated round | ≤ 30 min | **14.0–17.9 min** (four seams, real adapters, 3 runs); **27.8 min** for the full 6-client round excluding training — training adds 55.8 min | `experiments/w12-integration/RESULTS.md`; `experiments/w4w5-g2-g3-round/SUMMARY.md` |
| Adapter swap | ≤ 2 s | 0.501 s median (p95 1.387 s) | edge, `experiments/w3-edge-lora-composite/RESULTS.md` |
| Composite TTFT overhead | ≤ 200 ms | +36.7 ms worst | edge, `docs/integration-sprint.md` §9 |

The round NFR passes only if D11 excludes client training. Including training it
is 83.6 min on the development GPU (RTX 2050, 4 GB), which is not the reference
GPU. Read it as "inside 30 min for aggregate + compose + evaluate", not as a
blanket pass.

## 6. Experiments and reproducibility (D8, D9)

`python -m registry.runs` turns a config into runs:

- **Before a run.** `check` / `plan` reject a config without a seed or a per-run
  `gpu_hours_estimate`. They also reject one that exceeds a D8 cap (6 clients,
  rank 16, seq len 1024, 200 steps, 5 rounds) without a written
  `caps_override.justification`, or one whose sweep exceeds `budget.gpu_hours_total`.
- **Running.** `run <config> -- <command with {rank} {outputs_dir} ...>` expands
  the sweep. It runs each point without a shell and archives it under
  `runs/<experiment>/<run_id>/` in the result store (`CLASP_RESULT_STORE`).
- **Manifest v2.** The contracts `RunManifest` (seed, config hash, adapter
  versions, GPU-hours estimate and actual), plus status, sweep point, command,
  exit code, timestamps, git commit, environment, and a sha256 per output.
- **Resume.** `run_id` is a hash of config and point. Re-running a sweep skips
  completed points and re-runs failed or interrupted ones. Completed runs are
  immutable, and `freeze --tag` makes an experiment final.

`python -m registry.repro pack [--registry-url URL]` bundles into one tarball:
configs and write-ups, every run manifest, installed package versions, a registry
metadata snapshot (versions, lineage, audit trail), `REPRODUCE.md`, and a
`MANIFEST.json` holding the git commit and a sha256 per file. `verify` re-hashes
everything, re-validates every config, and requires every run to be complete.

## 7. Known limits

- The registry is single-process by design: its write lock is per process, so
  it runs one uvicorn worker. That is ample for a 6-client federation.
- Tensor blobs are not in the repro pack. They are re-derivable from configs and
  seeds and are identified by sha256 in the registry snapshot.
- Composite ε uses basic composition. That is an upper bound, conservative, and
  correct, but looser than an accountant-level bound.
