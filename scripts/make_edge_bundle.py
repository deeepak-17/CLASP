"""Build the setup bundle for one edge laptop (demo plan v3 item 4).

    python scripts/make_edge_bundle.py client-flask --cluster-url http://192.168.43.10:8002
    python scripts/make_edge_bundle.py client-numpy --cluster-url https://192.168.43.10:8002 \\
        --certs .runtime/certs --with-corpus

The edge laptop clones the repo as usual; this zip carries what git does not
(trained weights and corpora are gitignored), laid out at the paths the edge
commands already use, so it unzips straight into the checkout:

    services/edge/artifacts/<set>/<client>/adapter/      the pre-trained adapter(s)
    services/edge/artifacts/<set>/<client>/manifest.json FedAvg weight + DP privacy block
    datasets/materialized/<cluster>/<client>/            (--with-corpus) for a live fine-tune
    edge-certs/<client>/                                 (--certs) ca.pem client.pem client.key
    EDGE_<client>.md, start_edge_<client>.ps1            what to run on that laptop

Uploading needs no GPU and no model: ``edge.upload`` only reads the adapter
(numpy + safetensors + requests). The 1.3B base weights matter only for a live
fine-tune (Q4), and the run steps say how to fetch them.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "services" / "edge" / "artifacts"
CORPUS = REPO_ROOT / "datasets" / "materialized"
DEFAULT_SETS = ("round2_d3", "dp_smoke", "round1")
MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"


def cluster_of(client: str) -> str:
    for cluster_dir in CORPUS.glob("*/"):
        if (cluster_dir / client).is_dir():
            return cluster_dir.name
    for s in DEFAULT_SETS:
        m = ARTIFACTS / s / client / "manifest.json"
        if m.exists():
            cid = json.loads(m.read_text(encoding="utf-8")).get("client_id", "")
            if "/" in cid:
                return cid.split("/", 1)[0]
    raise SystemExit(f"cannot tell which cluster {client} belongs to")


def run_steps(client: str, cluster: str, cluster_url: str, sets, certs: bool, corpus: bool) -> str:
    mtls = f" --mtls-dir edge-certs/{client}" if certs else ""
    adapter = f"services/edge/artifacts/{sets[0]}/{client}/adapter"
    train = (f"""
## Optional: a short live fine-tune (needs an NVIDIA GPU with >= 4 GB)

```powershell
pip install -e services/evaluation -e services/edge    # torch, transformers, peft, opacus (use a CUDA build of torch)
huggingface-cli download {MODEL_ID}        # several GB; do this before demo day
python -m edge.train_client --client {cluster}/{client} --max-steps 50 --skip-base-eval --out-dir services/edge/artifacts/live
```

Or use the "Short fine-tune" panel on the edge page. The new adapter shows up under `live/`.
""" if corpus else "")
    return f"""# Edge laptop: {client} (cluster `{cluster}`)

Laptop A runs the cluster at `{cluster_url}`. This laptop uploads `{client}`'s adapter to it.

## Once, before demo day

```powershell
git clone https://github.com/deeepak-17/CLASP.git; cd CLASP
# unzip this bundle into the CLASP folder (it adds files under services/, datasets/, edge-certs/)
python -m venv .venv; .venv\\Scripts\\activate
pip install -e contracts -e services/edge --no-deps
pip install numpy safetensors requests{" -e security" if certs else ""}
```

Uploading needs no GPU and no model weights.

## On the day

1. Join the same hotspot as laptop A. Check: `curl {cluster_url}/healthz`{" (with --cacert/--cert/--key from edge-certs)" if certs else ""}.
2. Wait until laptop A has pressed **Register clusters** in its demo panel.
3. Start this laptop's page and open http://127.0.0.1:8020:

   ```powershell
   .\\start_edge_{client}.ps1
   ```

   Pick the adapter, press **Upload to cluster**. Laptop A's panel shows `{client}` arrive.

Command-line equivalent:

```powershell
python -m edge.upload --cluster-url {cluster_url}{mtls} --adapter {adapter}
```

If the cluster answers 409 (it aggregated in between), run the upload again: it reads the new round itself.
{train}
Adapters in this bundle: {", ".join(f"`{s}`" for s in sets)}.
"""


def start_script(client: str, cluster_url: str, certs: bool) -> str:
    mtls = f" --mtls-dir edge-certs/{client}" if certs else ""
    return (f"# Start {client}'s edge page (http://127.0.0.1:8020) against laptop A's cluster.\n"
            f"$here = Split-Path -Parent $MyInvocation.MyCommand.Path\n"
            f"Set-Location $here\n"
            f"if (Test-Path .venv\\Scripts\\Activate.ps1) {{ . .venv\\Scripts\\Activate.ps1 }}\n"
            f"python -m edge.webui --cluster-url {cluster_url} --client-id {client}{mtls}\n")


def build(client: str, cluster_url: str, out: Path, sets, certs_dir, with_corpus: bool) -> Path:
    cluster = cluster_of(client)
    found = [s for s in sets if (ARTIFACTS / s / client / "adapter" / "adapter_model.safetensors").exists()]
    if not found:
        raise SystemExit(f"no trained adapter for {client} under {ARTIFACTS} in sets {list(sets)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        def add(path: Path, arc: str) -> None:
            nonlocal added
            z.write(path, arc)
            added += path.stat().st_size

        for s in found:
            root = ARTIFACTS / s / client
            for f in sorted(p for p in root.rglob("*") if p.is_file()):
                add(f, f.relative_to(REPO_ROOT).as_posix())
        if with_corpus:
            corpus = CORPUS / cluster / client
            if not corpus.is_dir():
                raise SystemExit(f"--with-corpus: {corpus} does not exist")
            for f in sorted(p for p in corpus.rglob("*") if p.is_file()):
                add(f, f.relative_to(REPO_ROOT).as_posix())
        if certs_dir:
            edge_certs = Path(certs_dir) / "edges" / client
            missing = [n for n in ("ca.pem", "client.pem", "client.key") if not (edge_certs / n).exists()]
            if missing:
                raise SystemExit(f"{edge_certs} lacks {missing}: run "
                                 f"CLASP_EDGE_CLIENTS={client} scripts/make_dev_certs.sh first")
            for n in ("ca.pem", "client.pem", "client.key"):
                add(edge_certs / n, f"edge-certs/{client}/{n}")
        z.writestr(f"EDGE_{client}.md",
                   run_steps(client, cluster, cluster_url, found, bool(certs_dir), with_corpus))
        z.writestr(f"start_edge_{client}.ps1", start_script(client, cluster_url, bool(certs_dir)))
    print(f"{out}  ({added / 1e6:.1f} MB before compression)")
    print(f"  client {client}, cluster {cluster}, adapters {found}"
          f"{', corpus' if with_corpus else ''}{', certs' if certs_dir else ''}")
    if certs_dir:
        print("  contains this edge's private key: hand it over directly, do not post it anywhere")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="zip what one edge laptop needs for the demo")
    ap.add_argument("clients", nargs="+", help="client ids, e.g. client-flask client-numpy")
    ap.add_argument("--cluster-url", required=True, help="laptop A's cluster, e.g. http://192.168.43.10:8002")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / ".runtime" / "edge-bundles")
    ap.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS),
                    help=f"artifact sets to include when present (default {' '.join(DEFAULT_SETS)})")
    ap.add_argument("--certs", type=Path, default=None,
                    help="make_dev_certs.sh output dir; adds edges/<client>/ for mTLS")
    ap.add_argument("--with-corpus", action="store_true", help="add the client's corpus for a live fine-tune")
    args = ap.parse_args(argv)
    for client in args.clients:
        build(client, args.cluster_url, args.out_dir / f"edge-{client}.zip", args.sets,
              args.certs, args.with_corpus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
