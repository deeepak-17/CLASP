"""One-command integration round across all four seams (Integration Sprint · B7).

    A   Edge  -> Cluster    POST /uploads                 (6 real client adapters)
    B   Cluster -> Registry POST /versions                (via /publish)
    C1  Registry -> Edge    GET  /active + /file          (sha256 verified)
    C2  Edge  -> Registry   POST /promote                 (D5 decision)

Run it:

    docker compose up -d registry cluster        # or scripts/run_services.py
    python scripts/demo_round.py --round 1

What actually happens, per cluster (web, scientific):

  1. Its three real trained LoRA adapters are uploaded over HTTP. The FedAvg
     sample weight is each client's block count, recomputed here by packing the
     client's corpus — not copied out of an old manifest.
  2. The cluster aggregates the SAME uploads twice: once with the naive
     per-factor average (D2's ablation baseline) and once with the exact
     SVD path. Both are published to the registry as immutable versions, v1 and
     v2, so the promotion rule has a baseline to judge the candidate against —
     which it must, because D5 rolls back when there is nothing to compare to.
  3. Both versions are pulled back out of the registry and materialized into
     PEFT directories, sha256-verified against what the registry recorded.
  4. For one representative client per cluster, a composite
     `base + alpha*cluster + beta*client` is built from each version and scored
     with the in-project next-line completion metric on that client's held-out
     files.
  5. The candidate (v2) goes to POST /promote with the baseline's metrics. The
     registry applies D5 and answers PROMOTE or ROLLBACK.

Nothing here is stubbed and nothing is back-filled from a previous run. If a
step cannot run — no GPU, no HumanEval guard — the manifest says so in that
step's own record and the affected numbers are absent, not invented.

Results land in ``experiments/w12-integration/results/``.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "services" / "edge" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "services" / "edge" / "src"))

import requests  # noqa: E402

from edge import wire  # noqa: E402
from edge.promote import promote_candidate, resolve_guard  # noqa: E402
from edge.registry_client import RegistryClient  # noqa: E402

DEFAULT_CLUSTER_URL = "http://localhost:8002"
DEFAULT_REGISTRY_URL = "http://localhost:8004"
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "w12-integration" / "results"
DEFAULT_ADAPTERS = REPO_ROOT / "services" / "edge" / "artifacts" / "round1"
DEFAULT_CORPUS = REPO_ROOT / "datasets" / "materialized"

#: D1's static clusters. Client ids are the directory names under the corpus root.
CLUSTERS: Dict[str, List[str]] = {
    "web": ["client-flask", "client-requests", "client-werkzeug"],
    "scientific": ["client-numpy", "client-pandas", "client-scikit-learn"],
}
#: Which client's held-out files stand for the cluster in the D5 evaluation.
#: The largest held-out split gives the least noisy edit-similarity estimate.
REPRESENTATIVE = {"web": "client-werkzeug", "scientific": "client-scikit-learn"}

NFR_ROUND_MINUTES = 30.0


def log(msg: str = "") -> None:
    print(msg, flush=True)


def banner(seam: str, text: str) -> None:
    log(f"\n{'=' * 74}\n{seam}  {text}\n{'=' * 74}")


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def preflight(cluster_url: str, registry_url: str) -> Dict:
    out = {}
    for name, url in (("cluster", f"{cluster_url}/healthz"),
                      ("registry", f"{registry_url}/healthz")):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            out[name] = r.json()
            log(f"  {name:9s} {url}  ->  ok")
        except Exception as exc:  # noqa: BLE001 - report and stop, do not guess
            raise SystemExit(
                f"{name} is not answering at {url} ({exc}).\n"
                f"Start the stack first:  docker compose up -d registry cluster\n"
                f"or, without docker:     python scripts/run_services.py") from exc
    return out


# --------------------------------------------------------------------------- #
# corpus packing (real sample weights + held-out splits)
# --------------------------------------------------------------------------- #
def pack_corpora(corpus_root: Path, clients: List[str], seq_len: int) -> Tuple[Dict[str, Dict], object]:
    """Block counts and held-out splits for every client, via edge.chunking.

    Needs the tokenizer only — no model, no GPU. The block count is the FedAvg
    weight the aggregation uses, so it is recomputed rather than trusted from a
    stale manifest.
    """
    from edge.chunking import discover_clients, pack_client
    from edge.config import PROFILES
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(PROFILES["dev"].model_id)
    tokenizer.pad_token = tokenizer.eos_token
    found = discover_clients(corpus_root)
    packs: Dict[str, Dict] = {}
    for cid, cdir in sorted(found.items()):
        short = cdir.name
        if short not in clients:
            continue
        train, held = pack_client(cdir, tokenizer, seq_len=seq_len)
        packs[short] = {
            "client_key": cid, "dir": cdir,
            "n_train_files": train["n_files"], "n_train_blocks": train["n_chunks"],
            "n_held_out_files": held["n_files"], "n_held_out_blocks": held["n_chunks"],
            "held": held,
        }
        log(f"  {short:22s} train {train['n_chunks']:5d} blocks / {train['n_files']:4d} files"
            f"   held-out {held['n_chunks']:4d} blocks / {held['n_files']:3d} files")
    missing = [c for c in clients if c not in packs]
    if missing:
        raise SystemExit(f"no materialized corpus for {missing} under {corpus_root}")
    return packs, tokenizer


# --------------------------------------------------------------------------- #
# seam A + B
# --------------------------------------------------------------------------- #
def run_seam_a_b(cluster_url: str, registry_url: str, cluster_id: str,
                 members: List[str], adapters_root: Path, packs: Dict[str, Dict],
                 round_no: int, seed: int) -> Dict:
    """Upload three clients, aggregate twice (naive then svd), publish both."""
    registry_name = f"cluster-{cluster_id}"
    health = requests.get(f"{cluster_url}/healthz", timeout=10).json()
    cluster_round = health["clusters"].get(cluster_id, {}).get("round_id", 0)

    uploads = []
    for short in members:
        adapter_dir = adapters_root / short / "adapter"
        if not (adapter_dir / "adapter_model.safetensors").exists():
            raise SystemExit(
                f"{adapter_dir} has no adapter_model.safetensors. The trained "
                f"weights are gitignored; point --adapters at the directory that "
                f"holds them.")
        payload = wire.upload_payload_from_dir(
            adapter_dir, client_id=short, cluster_id=cluster_id,
            round_id=cluster_round, num_examples=packs[short]["n_train_blocks"],
            seed=seed)
        t0 = time.time()
        r = requests.post(f"{cluster_url}/uploads", json=payload, timeout=600)
        if r.status_code != 201:
            raise SystemExit(f"A: upload of {short} failed {r.status_code}: {r.text[:400]}")
        body = r.json()
        dt = round(time.time() - t0, 2)
        uploads.append({"client": short, "status": r.status_code, "seconds": dt, **body})
        log(f"  A  {short:22s} {body['tensors_received']:3d} tensors, "
            f"{payload['num_layers']} layers  ->  201  ({dt}s)")

    versions = {}
    for label, method, retain in (("baseline", "naive", True), ("candidate", "svd", False)):
        t0 = time.time()
        r = requests.post(f"{cluster_url}/aggregate", json={
            "cluster_id": cluster_id, "aggregation": method,
            "include_manifest": True, "retain_uploads": retain,
            "exact_lowrank": True}, timeout=1800)
        if r.status_code != 200:
            raise SystemExit(f"A: aggregate({method}) failed {r.status_code}: {r.text[:400]}")
        broadcast = r.json()
        manifest = requests.get(f"{cluster_url}/adapters/{cluster_id}/manifest",
                                timeout=60).json()
        agg_s = round(time.time() - t0, 2)
        err = manifest.get("svd_reconstruction_error", {})
        log(f"  A  aggregate {method:6s} -> rank {broadcast['rank']}, "
            f"{broadcast['num_layers']} layers, {len(broadcast['tensors'])} tensors  "
            f"({agg_s}s)")
        if err:
            log(f"       vs exact weighted average: mean {err.get('mean', float('nan')):.6f}  "
                f"max {err.get('max', float('nan')):.6f}  over {err.get('n_modules')} modules")

        t0 = time.time()
        r = requests.post(f"{cluster_url}/adapters/{cluster_id}/publish", json={
            "registry_url": registry_url, "adapter_name": registry_name,
            "round": round_no, "seed": seed, "set_active": True}, timeout=900)
        if r.status_code != 201:
            raise SystemExit(f"B: publish({method}) failed {r.status_code}: {r.text[:400]}")
        version_meta = r.json()["version"]
        log(f"  B  publish {method:6s} -> {registry_name} v{version_meta['ref']['version']}  "
            f"sha256 {version_meta['sha256'][:16]}...  "
            f"clients {version_meta['source_clients']}  ({round(time.time()-t0,2)}s)")
        versions[label] = {
            "aggregation": method, "seconds": agg_s,
            "aggregation_manifest": manifest, "registry_version": version_meta,
        }
    return {"registry_name": registry_name, "cluster_round_id": cluster_round,
            "uploads": uploads, "versions": versions}


# --------------------------------------------------------------------------- #
# seam C1
# --------------------------------------------------------------------------- #
def run_seam_c1(rc: RegistryClient, registry_name: str, versions: Dict,
                out_root: Path) -> Dict:
    pulled = {}
    for label, rec in versions.items():
        v = rec["registry_version"]["ref"]["version"]
        dest = out_root / registry_name / f"v{v}"
        t0 = time.time()
        manifest = rc.materialize(registry_name, dest, version=v)
        log(f"  C1 pull {registry_name} v{v} ({label:9s}) -> {dest.relative_to(REPO_ROOT)}  "
            f"sha256 verified {manifest['sha256_verified'][:16]}...  "
            f"({round(time.time()-t0,2)}s)")
        manifest["seconds"] = round(time.time() - t0, 2)
        pulled[label] = manifest
    return pulled


# --------------------------------------------------------------------------- #
# composite + in-project evaluation
# --------------------------------------------------------------------------- #
def build_composite(cluster_dir: Path, client_dir: Path, alpha: float, beta: float,
                    out_dir: Path) -> Dict:
    """base + alpha*cluster + beta*client via the existing edge.merge machinery."""
    from edge.merge import (
        CONTRACT_HYPERPARAMS,
        compose,
        load_adapter,
        merge_max_error,
        save_adapter,
        validate_compatibility,
    )

    cluster_sd, cluster_cfg = load_adapter(cluster_dir)
    client_sd, client_cfg = load_adapter(client_dir)
    info = validate_compatibility([cluster_cfg, client_cfg], ["cluster", "client"],
                                  contract=CONTRACT_HYPERPARAMS)
    parts = [(cluster_sd, cluster_cfg, alpha), (client_sd, client_cfg, beta)]
    sd, cfg = compose(parts)
    err = merge_max_error(parts, sd, cfg)
    save_adapter(out_dir, sd, cfg)
    return {"path": str(out_dir), "rank": cfg["r"], "alpha": alpha, "beta": beta,
            "merge_self_check": err, "compatibility": info}


def evaluate_composite(model_bundle, composite_dir: Path, pack: Dict, examples,
                       adapter_name: str, batch_size: int, max_new_tokens: int,
                       prompt_tokens: int, ppl_max_blocks: Optional[int]) -> Dict:
    """Load a composite as a PEFT adapter and measure perplexity + completion.

    ``ppl_max_blocks`` caps the held-out perplexity pass the same way
    ``edge.round``'s ``--grid-max-blocks`` caps its alpha sweep: scikit-learn's
    244 blocks at micro-batch 1 dominate the round's wall clock, and the
    promotion rule reads edit_similarity, not perplexity. The cap is recorded
    in the manifest so no capped number is mistaken for a full-split one.
    """
    from edge.completion_eval import evaluate_completion
    from edge.train_client import evaluate as ppl_evaluate

    model, tokenizer, holder = model_bundle
    model = holder["model"]
    safe = adapter_name.replace("/", "_").replace(".", "_").replace("-", "_")
    if holder["anchored"]:
        model.load_adapter(str(composite_dir), adapter_name=safe)
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(composite_dir), adapter_name=safe)
        holder["model"] = model
        holder["anchored"] = True
    model.set_adapter(safe)

    chunks = pack["held"]["chunks"]
    capped = chunks[:ppl_max_blocks] if ppl_max_blocks else chunks
    ppl = ppl_evaluate(model, capped, tokenizer.pad_token_id)
    result = evaluate_completion(model, tokenizer, examples, ppl["perplexity"],
                                 batch_size=batch_size, max_new_tokens=max_new_tokens,
                                 prompt_tokens=prompt_tokens)
    result["perplexity_detail"] = {
        **ppl,
        "blocks_scored": len(capped),
        "blocks_available": len(chunks),
        "capped": len(capped) < len(chunks),
    }
    result["adapter_name"] = safe
    if len(getattr(model, "peft_config", {})) > 1:
        model.delete_adapter(safe)
        model.set_adapter(next(iter(model.peft_config)))
    return result


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="CLASP four-seam integration round (B7)")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--cluster-url", default=DEFAULT_CLUSTER_URL)
    ap.add_argument("--registry-url", default=DEFAULT_REGISTRY_URL)
    ap.add_argument("--adapters", default=str(DEFAULT_ADAPTERS),
                    help="root holding <client>/adapter directories")
    ap.add_argument("--corpus-root", default=str(DEFAULT_CORPUS))
    ap.add_argument("--out", default=str(DEFAULT_RESULTS))
    ap.add_argument("--only", nargs="+", choices=sorted(CLUSTERS),
                    help="restrict to these clusters")
    ap.add_argument("--alpha", type=float, default=0.5, help="cluster coefficient")
    ap.add_argument("--beta", type=float, default=1.0, help="client coefficient")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=60,
                    help="in-project completion examples per evaluation")
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--ppl-max-blocks", type=int, default=0,
                    help="cap the held-out perplexity pass (0 = full split, the "
                         "default: the round measured 14.33 min against a 30 min "
                         "NFR uncapped). Cap it on a slower box; the cap is "
                         "recorded per number in the manifest.")
    ap.add_argument("--noise-band", type=float, default=0.0,
                    help="edit-similarity improvement required (P5's measured band)")
    ap.add_argument("--candidate-anchor", help="HumanEval anchor.json for the candidate")
    ap.add_argument("--baseline-anchor", help="HumanEval anchor.json for the baseline")
    ap.add_argument("--skip-eval", action="store_true",
                    help="skip the model-backed evaluation (no GPU): seams A/B/C1 "
                         "still run and C2 goes out with the metric marked unmeasured")
    args = ap.parse_args()

    t_round = time.time()
    started = datetime.now(timezone.utc).isoformat()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pulled_root = out_dir / "pulled"
    composite_root = out_dir / "composites"
    selected = args.only or sorted(CLUSTERS)

    banner("0", "preflight — services answering")
    health = preflight(args.cluster_url, args.registry_url)

    guard = resolve_guard(args.candidate_anchor, args.baseline_anchor)
    log(f"\n  HumanEval guard: {'AVAILABLE' if guard.available else 'UNAVAILABLE'}")
    log(f"    {guard.reason}")

    banner("0", "packing client corpora (real FedAvg weights + held-out splits)")
    members = [c for cl in selected for c in CLUSTERS[cl]]
    # pack_corpora loads its own tokenizer (no model needed); the eval path
    # uses the one that comes back with the model. Same model id either way.
    packs, _tokenizer = pack_corpora(Path(args.corpus_root), members, args.seq_len)

    model_bundle = None
    eval_note = None
    if args.skip_eval:
        eval_note = ("--skip-eval: no model was loaded, so no in-project metric was "
                     "measured this run")
        log(f"\n  {eval_note}")
    else:
        banner("0", "loading the base model (NF4, dev profile)")
        try:
            from edge.model_loader import load_model
            from edge.train_client import set_determinism

            set_determinism(args.seed)
            model, tok, profile = load_model("dev")
            model.eval()
            model.config.use_cache = True
            model_bundle = (model, tok, {"model": model, "anchored": False})
            log(f"  loaded {profile.model_id} on {model.device}")
        except Exception as exc:  # noqa: BLE001
            eval_note = (f"base model could not be loaded ({type(exc).__name__}: {exc}); "
                         f"the in-project metric was NOT measured this run")
            log(f"  !! {eval_note}")

    rc = RegistryClient(args.registry_url)
    results: Dict[str, Dict] = {}
    counters = {"uploads": 0, "aggregations": 0, "snapshots": 0,
                "composites": 0, "decisions": 0}

    for cluster_id in selected:
        members = CLUSTERS[cluster_id]
        banner("A/B", f"cluster-{cluster_id}: {len(members)} clients -> aggregate -> registry")
        ab = run_seam_a_b(args.cluster_url, args.registry_url, cluster_id, members,
                          Path(args.adapters), packs, args.round, args.seed)
        counters["uploads"] += len(ab["uploads"])
        counters["aggregations"] += len(ab["versions"])
        counters["snapshots"] += len(ab["versions"])

        banner("C1", f"cluster-{cluster_id}: pull both versions back, verify sha256")
        pulled = run_seam_c1(rc, ab["registry_name"], ab["versions"], pulled_root)

        rep = REPRESENTATIVE[cluster_id]
        client_dir = Path(args.adapters) / rep / "adapter"
        evals: Dict[str, Optional[Dict]] = {}
        composites: Dict[str, Dict] = {}
        examples = []
        if model_bundle is not None:
            from edge.completion_eval import client_examples

            examples = client_examples(packs[rep]["dir"], args.max_examples, args.stride)
            log(f"\n  representative client for D5: {rep} "
                f"({len(examples)} held-out completion examples)")
        for label, rec in pulled.items():
            comp_dir = composite_root / ab["registry_name"] / label
            composites[label] = build_composite(
                Path(rec["path"]), client_dir, args.alpha, args.beta, comp_dir)
            counters["composites"] += 1
            log(f"  D6 composite {label:9s} rank {composites[label]['rank']}  "
                f"a={args.alpha} b={args.beta}  "
                f"merge self-check rel {composites[label]['merge_self_check']['max_rel_err']:.2e}")
            if model_bundle is not None and examples:
                t0 = time.time()
                ev = evaluate_composite(model_bundle, comp_dir, packs[rep], examples,
                                        f"{ab['registry_name']}_{label}",
                                        args.batch_size, args.max_new_tokens,
                                        args.prompt_tokens, args.ppl_max_blocks or None)
                m = ev["in_project"]
                log(f"     eval {label:9s} edit_sim {m['edit_similarity']:.4f}  "
                    f"exact {m['exact_match']:.4f}  ppl {m['perplexity']:.4f}  "
                    f"({round(time.time()-t0,1)}s)")
                evals[label] = ev
            else:
                evals[label] = None

        banner("C2", f"cluster-{cluster_id}: D5 promotion decision")
        cand_v = ab["versions"]["candidate"]["registry_version"]["ref"]["version"]
        cand_metrics = evals.get("candidate")
        base_metrics = evals.get("baseline")
        if cand_metrics is None:
            log("  !! in-project metric unmeasured this run — the D5 rule is being "
                "given no edit_similarity, which is a ROLLBACK by construction (F3).")
        promotion = promote_candidate(
            rc, ab["registry_name"], version=cand_v, kind="cluster",
            cluster_id=cluster_id,
            in_project=(cand_metrics["in_project"] if cand_metrics else None),
            baseline_in_project=(base_metrics["in_project"] if base_metrics else None),
            guard=guard, noise_band=args.noise_band, seed=args.seed)
        counters["decisions"] += 1
        d = promotion["decision"]
        log(f"  C2 {d['action'].upper()}  active v{d['active_version_after']}")
        log(f"     reason: {d['reason']}")
        log(f"     {promotion['authority_note']}")

        results[cluster_id] = {
            "seam_a_b": ab, "seam_c1": pulled, "composites": composites,
            "evaluation": evals, "representative_client": rep,
            "examples_scored": len(examples), "seam_c2": promotion,
            "clients": {c: {k: v for k, v in packs[c].items()
                            if k not in ("held", "dir")} for c in members},
        }

    banner("=", "round summary")
    total_minutes = round((time.time() - t_round) / 60.0, 2)
    audit = {}
    for cluster_id in selected:
        name = f"cluster-{cluster_id}"
        audit[name] = rc.list_promotions(name)
        versions = rc.list_versions(name)
        log(f"\n  {name}: {len(versions['versions'])} versions, active v{versions['active']}")
        for v in versions["versions"]:
            log(f"    v{v['ref']['version']}  {v['aggregation']:10s}  "
                f"sha256 {v['sha256'][:16]}...  clients {v['source_clients']}")
        for dec in audit[name]:
            log(f"    decision: {dec['action'].upper()} -> active v{dec['active_version_after']}")

    manifest = {
        "task": "Integration Sprint B7 — four-seam round (A: edge->cluster, "
                "B: cluster->registry, C1: registry->edge, C2: edge->registry)",
        "round": args.round,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_minutes": total_minutes,
        "nfr_round_minutes": NFR_ROUND_MINUTES,
        "nfr_met": total_minutes <= NFR_ROUND_MINUTES,
        "services": {"cluster": args.cluster_url, "registry": args.registry_url,
                     "healthz": health},
        "config": {
            "alpha": args.alpha, "beta": args.beta, "seed": args.seed,
            "seq_len": args.seq_len, "max_examples": args.max_examples,
            "stride": args.stride, "max_new_tokens": args.max_new_tokens,
            "prompt_tokens": args.prompt_tokens, "noise_band": args.noise_band,
            "ppl_max_blocks": args.ppl_max_blocks,
            "clusters": {c: CLUSTERS[c] for c in selected},
            "representative_clients": {c: REPRESENTATIVE[c] for c in selected},
            "adapters_root": str(args.adapters), "corpus_root": str(args.corpus_root),
        },
        "counters": counters,
        "humaneval_guard": guard.as_dict(),
        "evaluation_note": eval_note,
        "clusters": results,
        "promotion_audit_trail": audit,
        "versions": {"python": platform.python_version()},
        "known_deviations": {
            "d3_composition_order": (
                "clients were trained against the frozen base, not against frozen "
                "(base + alpha*cluster) — no cluster adapter existed at training "
                "time. Not closed by this sprint; needs a second training pass."),
            "d5_noise_band": (
                f"band = {args.noise_band}; the measured band is still P5's "
                f"outstanding deliverable. At 0.0, 'improved' means 'improved by "
                f"any amount'."),
            "d5_regression_guard": guard.reason,
            "d1_web_cluster": (
                "on disk the web cluster is {flask, requests, werkzeug}; D1 "
                "specifies {django, flask, requests}"),
            "seam_security": (
                "no DP and no mTLS on this channel — P3's modules are a library "
                "this sprint, per plan (W7/W10)"),
        },
    }
    try:
        import torch

        manifest["versions"]["torch"] = torch.__version__
        manifest["hardware"] = {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                             if torch.cuda.is_available() else None),
        }
    except Exception:  # noqa: BLE001 - torch is optional for the A/B/C1 path
        pass

    path = out_dir / f"round{args.round}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    shutil.rmtree(composite_root, ignore_errors=True)
    rc.close()

    log(f"\n  counters: {counters}")
    log(f"  wall clock: {total_minutes:.2f} min against a {NFR_ROUND_MINUTES:g} min NFR "
        f"({'MET' if manifest['nfr_met'] else 'MISSED'})")
    log(f"  manifest: {path}")


if __name__ == "__main__":
    main()
