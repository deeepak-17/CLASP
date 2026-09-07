# CLASP Panel Demo UI

**Not a CLASP service.** This is presentation scaffolding for the Phase II
Panel Review (P4/Deepak), isolated under `demo_ui/` on purpose — it is never
imported by `services/`, never in the Docker Compose stack, and breaks nothing
if deleted after the panel.

## What it actually does

Three static pages, one thin FastAPI backend (`server.py`) that:

1. **Reads real, already-committed artifacts** for anything that can't run
   live in a browser demo — Edge's real training/composition/eval numbers
   need a GPU + the base model, neither available at a panel:
   `services/edge/artifacts/round1/*/manifest.json`,
   `services/edge/artifacts/round1_out/round_manifest.json`,
   `services/edge/artifacts/clusters/*/aggregate_manifest.json`.
2. **Proxies straight through** to the real Cluster (`:8002`) and Registry
   (`:8004`) FastAPI apps for everything that *can* run live. No aggregation
   or promotion logic is reimplemented — every proxied call hits the actual
   service code.
3. **Orchestrates** the multi-step demo actions (send-to-cluster,
   publish-to-registry, promote) by sequencing the same real HTTP calls a
   real Edge client would make. Business logic stays in `cluster`/`registry`;
   this layer only sequences requests and reads files.

## Architecture

```
Browser (3 static pages, vanilla JS)
        │  fetch("/api/...")
        ▼
demo_ui/server.py  (:8010 — this demo layer, not a CLASP service)
        │
        ├── /api/edge/*       → reads real committed JSON artifacts (no live Edge)
        ├── /api/cluster/*    → proxies, unchanged, to cluster.server:app   (:8002)
        ├── /api/registry/*   → proxies, unchanged, to registry.app:app    (:8004)
        └── /api/demo/*       → orchestrates real calls to :8002 / :8004
```

## Known, load-bearing limitation

`cluster.server`'s real implementation holds **one active aggregate at a
time** (one process, one cluster, one upload buffer — by its own design, see
`_ClusterState`'s docstring). This demo layer works within that: it verifies
(via `GET /aggregate/manifest`) that the aggregate you're about to publish
actually came from the cluster you're publishing it as, and refuses (409) if
not. **Always finish one cluster fully — Send → Publish → Promote — before
starting the other.**

## Setup

```bash
# from repo root, one venv is enough for all three processes
python3 -m venv .venv-demo
.venv-demo/bin/pip install -e contracts -e services/cluster -e services/registry
.venv-demo/bin/pip install -r demo_ui/requirements.txt
```

## Run (3 terminals + the browser)

```bash
# Terminal 1 — Registry (real service, stateful)
CLASP_REGISTRY_DATA=./_demo_data/registry .venv-demo/bin/uvicorn registry.app:app --port 8004

# Terminal 2 — Cluster (real service)
.venv-demo/bin/uvicorn cluster.server:app --port 8002

# Terminal 3 — Demo UI
.venv-demo/bin/uvicorn demo_ui.server:app --port 8010
```

Open `http://localhost:8010/` (redirects to Page 1). See `RUNBOOK.md` for the
full pre-demo checklist and presentation script.
