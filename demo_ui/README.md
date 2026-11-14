# CLASP live round panel

**Not a CLASP service.** The page laptop A shows during the multi-laptop demo
(P4/Deepak), isolated under `demo_ui/` on purpose — it is never imported by
`services/` and breaks nothing if deleted.

## What it actually does

One page (`static/live.html`) and a thin FastAPI backend (`server.py`):

1. **Drives the real services** through their own HTTP APIs (`/api/live/*`):
   register each cluster's members, aggregate, publish to the registry,
   promote. Every number and tensor comes from them or from the edge laptops,
   which upload their own trained adapters (`edge.webui`).
2. **Promote sends measured evaluation** (`evidence.py`), never constants, and
   says where each number came from. On a ROLLBACK it tells the cluster to
   serve the restored version again, so the edges' next round builds on it.
3. **Proxies** `/api/cluster/*` and `/api/registry/*` straight through, for
   inspection.

The earlier panel pages and their `/api/demo/*` routes (synthetic tensors,
hardcoded promote metrics, a staged regression) were removed.

```
edge laptops (edge.webui) ──upload──► cluster :8002 ◄──┐
                                                        │ /api/live/*
browser ──► demo_ui/server.py :8010 ────────────────────┤
                                                        │
                                       registry :8004 ◄─┘
```

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

Open `http://localhost:8010/` for the live multi-laptop round; the edges upload from
their own laptops (see `docs/multi-laptop-demo.md`). `python scripts/run_services.py --ui`
starts all three in one go.
