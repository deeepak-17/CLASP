# Multi-laptop demo — runbook

The build for *CLASP Multi-Laptop Demo Plan v3* (2026-10-09). Laptop A runs the
server side; laptops B and C are edges that upload over a phone hotspot. The guide
watches the live round in `demo_ui` and the results in the evaluation dashboard.

```
 laptop B (edge)             laptop A                              laptop C (edge)
 edge.webui :8020 ──upload──► cluster  :8002 ──publish──► registry :8004 ◄── ...
   client-flask              demo_ui  :8010 (live panel)            edge.webui :8020
                             dashboard :8005 (results)              client-numpy
```

## What each plan item became

| # | Item | Where | State |
|---|------|-------|-------|
| 1 | Post-merge smoke test | this runbook, "Verified" below | Run on one machine over the LAN address, plain and mTLS |
| 2 | `edge.upload` | `services/edge/src/edge/upload.py` | Done, tested against the real cluster |
| 3 | Edge web page | `python -m edge.webui` (:8020) | Done |
| 4 | Edge setup bundle | `scripts/make_edge_bundle.py` | Done |
| 5 | Bind `0.0.0.0` outside Docker | `scripts/run_services.py` | Done (`--ui`, `--mtls-dir`, prints LAN URLs) |
| 6 | Uploaded client IDs in `GET /clusters` | `uploaded_clients`, `aggregated_clients` | Done |
| 7 | Live panel on per-cluster endpoints | `demo_ui` `/` (`static/live.html`, `/api/live/*`) | Done |
| 8 | Evaluation → Promote | `scripts/evaluate_candidate.py`, `demo_ui/evidence.py` | Done; the candidate itself is evaluated (2026-10-14) |
| 9 | Cluster TLS switched on | `python -m cluster.serve` | Done; uploads bound to the client cert's CN |
| 10 | LAN certificates | `scripts/make_dev_certs.sh` (`CLASP_LAN_IPS`, `CLASP_EDGE_CLIENTS`) | Done |
| 11 | Round into the React dashboard | `scripts/feed_dashboard.py`, `results/rounds.json` | Done for round 1 |
| 12 | `docker compose --profile demo up` on laptop A | compose files updated | **Not run** (no docker daemon on the build machine) |
| 13 | Network rehearsal on 3 laptops | steps below | **To do, all** |
| 14 | Fallback | `scripts/demo_local.py` | Done; the video is still to record |

Defaults taken for the open questions (change them if the team decides otherwise):
Q3 plain HTTP on the hotspot, with mTLS one flag away (and tested); Q4 pre-trained
adapters, live fine-tune optional from the edge page; Q5 Docker on laptop A, with
`run_services.py` as the fallback; Q6 `demo_ui` for the live run and the dashboard
for results. Q1/Q2: an edge that only uploads needs **no GPU and no model weights**,
just the bundle (`numpy`, `safetensors`, `requests`, `contracts`).

## Before demo day

On laptop A (the one with the trained adapters):

```powershell
# 1. laptop A's address on the hotspot (ipconfig) — say 192.168.43.10.
#    With the WireGuard tunnel of docs/remote-access.md use its fixed address
#    (10.77.0.1) instead: it does not change between sessions.
# 2. certificates and keys for that address (in Git Bash; everything is mTLS by default)
#    $ CLASP_LAN_IPS=192.168.43.10 CLASP_EDGE_CLIENTS="client-flask client-numpy" scripts/make_dev_certs.sh
# 3. who may log in: the panel's operator, and each edge page's developer
python scripts/make_users.py --out .runtime/users/panel.json --add operator:operator
python scripts/make_users.py --out .runtime/users/client-flask.json --add asha:developer:client-flask
# 4. one bundle per edge laptop: adapters, manifest, that edge's certificate,
#    upload key and users file, and its run steps
python scripts/make_edge_bundle.py client-flask client-numpy --cluster-url https://192.168.43.10:8002
# 5. refresh the dashboard's data (bakes services/evaluation/results in)
python scripts/feed_dashboard.py --build
```

Hand each `.runtime/edge-bundles/edge-<client>.zip` to its laptop. The zip holds
`EDGE_<client>.md` with that laptop's steps. A bundle contains that edge's private
key and upload key, so hand it over directly.

Hotspot addresses change between sessions. If laptop A's IP changes, remake the
certificates (step 2), restart the services and remake the bundles (the URL and
the certificates are inside them). An edge that still has the old ones is told
so: its upload fails with the address it tried, the addresses the certificate
lists, and the command above.

With the WireGuard tunnel (`docs/remote-access.md`) skip the address hunt: laptop A is
always `10.77.0.1`, the certificates are issued for it once, and the firewall rule is the
tunnel-only one from that document instead of the one below.

Windows Firewall on laptop A (PowerShell as administrator), without the tunnel:

```powershell
New-NetFirewallRule -DisplayName "CLASP demo" -Direction Inbound -Protocol TCP `
  -LocalPort 8002,8004,8005,8010 -Action Allow -Profile Any
```

Some hotspots isolate clients from each other. Test from each edge laptop during
the rehearsal (item 13), with the certificate from its bundle:
`curl --cacert edge-certs/<client>/ca.pem --cert edge-certs/<client>/client.pem --key edge-certs/<client>/client.key https://<A>:8002/healthz`.

## On the day

Laptop A:

```powershell
docker compose --profile demo up -d                     # registry, cluster, demo-ui, dashboard: mTLS + HTTPS
# without docker:
python scripts/run_services.py --ui                     # the same, on the host
#   plain HTTP, for debugging only: --insecure, or -f docker-compose.insecure.yml
```

Open `https://localhost:8010` (import `.runtime/certs/ca.pem` into the browser
once, or accept its warning), log in as the operator, and press
**Register edge laptops**.

Laptops B and C: `.\start_edge_<client>.ps1`, open `https://127.0.0.1:8020` and log in.

One round, end to end (2026-10-14 build):

1. **Train** on B and C ("Train a new adapter here first"): DP-SGD (ε ≤ 8 over the run's
   two rounds together) with the
   FedProx proximal term (μ 0.01, `--fedprox-mu`), 512-token blocks. In round 0 there
   is no merged adapter yet, so it trains on the bare base.
2. **Send**. Laptop A's panel shows each laptop arrive. Every upload passes P3's
   integrity check (`security.validation`); a round of 3+ is screened for poisoned
   updates.
3. **Merge** per cluster on laptop A. Round 0 averages the full adapters. From round 1
   on, each upload names the merged adapter it trained on (sha256) and the cluster adds
   the round's updates onto it — FedProx's server step,
   `C_{N+1} = C_N + (1/α)·Σ w_k R_k` — refusing anything trained on another adapter
   (409 stale_adapter). The panel says which rule ran and the merge's ε: the largest
   cumulative ε any laptop in it has spent over its run.
   The merge is the cluster's **candidate**: the edges keep pulling the adapter the
   cluster already serves, and the next merge waits until this one is decided. The edge
   page's step 4 reads "Awaiting evaluation".
4. **Publish** the candidate to the registry (`cluster-<id>`, a new immutable version,
   inactive: nothing is served from it yet).
5. **Evaluate** on laptop A: `scripts/evaluate_candidate.py` scores the candidate (the
   newest version above the active one) and the active version, each as base + α·cluster
   — or, when nothing is active yet, the bare base model with no adapter: in-project
   completion on the noise-band client's held-out files and the line-completion guard
   (300 frozen examples). Several minutes; results are cached by sha256 (the bare base
   under `base`). It runs in demo_ui's own environment, so start laptop A with
   `run_services.py` from the `.venv` (torch + CUDA); the docker demo-ui image has no GPU
   stack.
6. **Promote**: the registry's D5 rule on those measurements of that exact version. The
   panel then tells the cluster: on PROMOTE `candidate/promote` (the registry's sha256 for
   that version), on REJECT `candidate/reject`. The panel shows "Promoted" or
   "Rejected".
7. **Back on the edges**: after a promote, each page pulls the merged adapter laptop A now
   serves (it pulls only when `active_sha256` changes) and the next training starts from
   it; step 4 reads "Merged and promoted". After a reject nothing is pulled: the edges
   keep the adapter they have, step 4 reads "Rejected, previous adapter kept", and the
   next round trains on it.
8. **After round 2: Re-cluster (dry run)**. Once both clusters have merged round 2 the
   panel's "Re-cluster (dry run)" button is enabled. It shows the grouping the cluster
   would choose from round 2's updates beside the current one, the laptops that would
   move, and the similarity matrix. Nothing is moved: applying a new grouping starts a
   new run with a new privacy budget. Save the response as
   `experiments/clustering-ablation/results/live-dry-run-<date>.json` (the browser's
   network tab, or `POST /api/live/recluster` with the operator's token).

A run is two rounds per laptop: the privacy budget (ε ≤ 8 over the run) is charged when
an adapter is sent, and a third send in the same run is refused on the laptop. The edge
page shows what has been spent; a new run is a new `--run-id`.

The results are on the dashboard at `https://<A>:8005`.

## Promote: what the guide will see

Promote sends only numbers measured on the version it judges, and the panel prints where
each came from (version, sha256, client, examples). A version nobody evaluated is not
judged: the panel asks for Evaluate first. `CLASP_EVAL_EVIDENCE` still lets an operator
supply an evaluation file explicitly (format in `demo_ui/evidence.py`). The noise band
is P5's `noise_report.json` (0.0, deterministic decoding).

Evaluate and Promote are enabled while a cluster has a candidate. The first candidate is
compared against the bare base model; later ones against the version the registry
serves. A rejected candidate is never served, and an operator restore never returns to
a refused version (a REJECT, or a D5 `rollback` recorded before the candidate flow): the
registry's `/restore` answers 409, and the cluster's
`/clusters/{id}/adapters/active/restore` reads the registry's `rejected` list first and
answers 409 `rejected_version` without fetching anything.

## Verified (2026-10-09, one Windows laptop with an RTX 2050, LAN address 10.214.204.34)

Plain HTTP:
- `run_services.py --ui` binds 0.0.0.0.
- Register, then four separate `edge.upload` processes posted to `http://10.214.204.34:8002`, two of them at the same time.
- Aggregate → publish v1 for both clusters. Promote refused, because a single version has nothing to compare against.
- Round 2: one upload through `edge.webui`, the DP adapter (ε 7.99) and two more over the CLI. Aggregate → publish v2, then Promote: ROLLBACK as above.
- The aggregate's ε is reported as unknown because `client-werkzeug` trained without DP (D7 rule).

mTLS (then an option, `--mtls-dir`; now the default), certs with the LAN IP:
- A client without a certificate is refused at the handshake.
- Per-edge uploads are accepted.
- An upload as `client-werkzeug` with `client-flask`'s certificate gets 403.
- The cluster published to the registry over mTLS. The sha256 was identical to the plain run (D9).

`demo_local.py --mtls`: two edge pages on one laptop, upload accepted.

Tests: cluster 234 passed (2 skipped), edge 156 passed (1 skipped), including the new upload, webui and serve tests.

Not verified: Docker (item 12), and two physical laptops on a hotspot (item 13).
