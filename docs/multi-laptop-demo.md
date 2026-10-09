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
| 8 | Evaluation → Promote | `demo_ui/evidence.py` | Done; see "Promote" below |
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
# 1. laptop A's address on the hotspot (ipconfig) — say 192.168.43.10
# 2. one bundle per edge laptop (adapters + manifest + run steps)
python scripts/make_edge_bundle.py client-flask client-numpy --cluster-url http://192.168.43.10:8002
#    for mTLS: certs first (in Git Bash), then bundles that carry each edge's cert
#    $ CLASP_LAN_IPS=192.168.43.10 CLASP_EDGE_CLIENTS="client-flask client-numpy" scripts/make_dev_certs.sh
python scripts/make_edge_bundle.py client-flask client-numpy --cluster-url https://192.168.43.10:8002 --certs .runtime/certs
# 3. refresh the dashboard's data (bakes services/evaluation/results in)
python scripts/feed_dashboard.py --build
```

Hand each `.runtime/edge-bundles/edge-<client>.zip` to its laptop. The zip holds
`EDGE_<client>.md` with that laptop's steps. A bundle made with `--certs` contains
that edge's private key, so hand it over directly.

Hotspot addresses change between sessions. If laptop A's IP changes, remake the
bundles (the URL is inside them) and, for mTLS, the certificates.

Windows Firewall on laptop A (PowerShell as administrator):

```powershell
New-NetFirewallRule -DisplayName "CLASP demo" -Direction Inbound -Protocol TCP `
  -LocalPort 8002,8004,8005,8010 -Action Allow -Profile Any
```

Some hotspots isolate clients from each other. Test with `curl http://<A>:8002/healthz`
from each edge laptop during the rehearsal (item 13).

## On the day

Laptop A:

```powershell
docker compose --profile demo up -d                     # registry, cluster, demo-ui, dashboard
#   mTLS: docker compose -f docker-compose.yml -f docker-compose.mtls.yml --profile demo up -d
# without docker:
python scripts/run_services.py --ui                     # add --mtls-dir .runtime/certs for mTLS
```

Open `http://localhost:8010` and press **1 · Register clusters and members**.

Laptops B and C: `.\start_edge_<client>.ps1`, open `http://127.0.0.1:8020`, pick the
adapter, press **Upload to cluster**. Laptop A's panel shows each client arrive.

Then, per cluster on laptop A: **Aggregate (SVD)** → **Publish** → **Promote**. The
results are on the dashboard at `http://<A>:8005`.

## Promote: what the guide will see

Promote sends numbers evaluation measured, never constants, and the panel prints
where each came from:

- In-project edit similarity for candidate and baseline (`evaluation.completion`,
  60 held-out examples, greedy), from the newest `scripts/demo_round.py` round
  manifest. Those were measured on round 1's aggregate (2026-09-06), not on the one
  built live, because scoring the 1.3B model needs a GPU. The panel says so.
- HumanEval guard: evaluation's measured base pass@1 (0.50 on 20 problems). No
  candidate has been scored, so the candidate guard is missing.
- Noise band: P5's `noise_report.json` (0.0, deterministic decoding).

D5 treats a missing guard as a failed one, so **the registry currently answers
ROLLBACK for both clusters**, even though edit similarity improved (+0.0133 web,
+0.0023 scientific). That is the honest answer. Two ways to change it:

- Evaluation scores a candidate HumanEval anchor
  (`services/evaluation/results/humaneval_guard/cluster-<id>_anchor.json`).
- A GPU laptop re-scores the live aggregate and writes a file that
  `CLASP_EVAL_EVIDENCE` points at (format in `demo_ui/evidence.py`).

D5 needs a previous version to roll back to, so Promote is enabled once a cluster
has two versions. Run one round before the guide arrives, or use the registry
volume from the rehearsal.

## Verified (2026-10-09, one Windows laptop with an RTX 2050, LAN address 10.214.204.34)

Plain HTTP:
- `run_services.py --ui` binds 0.0.0.0.
- Register, then four separate `edge.upload` processes posted to `http://10.214.204.34:8002`, two of them at the same time.
- Aggregate → publish v1 for both clusters. Promote refused, because a single version has nothing to compare against.
- Round 2: one upload through `edge.webui`, the DP adapter (ε 7.99) and two more over the CLI. Aggregate → publish v2, then Promote: ROLLBACK as above.
- The aggregate's ε is reported as unknown because `client-werkzeug` trained without DP (D7 rule).

mTLS (`--mtls-dir`, certs with the LAN IP):
- A client without a certificate is refused at the handshake.
- Per-edge uploads are accepted.
- An upload as `client-werkzeug` with `client-flask`'s certificate gets 403.
- The cluster published to the registry over mTLS. The sha256 was identical to the plain run (D9).

`demo_local.py --mtls`: two edge pages on one laptop, upload accepted.

Tests: cluster 234 passed (2 skipped), edge 156 passed (1 skipped), including the new upload, webui and serve tests.

Not verified: Docker (item 12), and two physical laptops on a hotspot (item 13).
