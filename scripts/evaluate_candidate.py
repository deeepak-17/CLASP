"""Evaluate the candidate the registry is about to judge, on this laptop's GPU (D5).

    python scripts/evaluate_candidate.py --cluster web --registry-url http://localhost:8004

Promote (D5) judges the registry's pending candidate — the newest version above
the active one, uploaded inactive — against the version the registry is
serving. This measures exactly those two, as the edges use them — ``base +
alpha * cluster`` (D3/D6) — so the decision is about the candidate itself, not
about an earlier round's adapter. When nothing is active yet (the first
candidate), the baseline is the bare base model with no adapter loaded (C7):

* **in-project** (D5's primary metric): next-line completion (edit similarity,
  exact match) on the held-out files of the client the noise band was
  measured on (``demo_ui.evidence.noise_band``), plus held-out perplexity;
  ``edge.completion_eval``, greedy, the same examples for both versions;
* **line-completion guard** (D5's regression side): next-line completion on the
  frozen 300-example guard set (``evaluation.eval_harness.line_guard``; click,
  jinja and colorama, which no cluster trains on), the same
  ``edge.completion_eval.evaluate_completion`` call, greedy, the same examples
  for both versions. The per-example edit similarities are kept so the paired
  noise of the candidate-minus-baseline difference can be reported.

Each version is scored once: results are cached by the payload's sha256 and
the guard set's fingerprint (the bare base model under ``base``), so a promoted
candidate is the next round's baseline for free and a score taken on another
guard set is never reused. The output,
``<out-dir>/<cluster>-v<N>.json``, is the evidence file ``demo_ui`` hands
Promote — only for that version and that sha256.

Needs the 1.3B model and a CUDA GPU (laptop A's RTX 2050), the client's corpus
under ``datasets/materialized`` and, under mTLS, ``CLASP_TLS_CLIENT_CERT`` /
``CLASP_TLS_CLIENT_KEY`` / ``CLASP_TLS_CA`` (demo_ui's certificate).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # demo_ui.evidence, for the noise band

DEFAULT_OUT = REPO_ROOT / ".runtime" / "live-eval"


def log(msg: str) -> None:
    print(msg, flush=True)


def pick_versions(listing: Dict, active_version: int | None) -> Tuple[Dict, Dict | None]:
    """(candidate, baseline): the candidate is the newest version above the
    active one; the baseline is the active version's metadata, or ``None`` when
    nothing is active (the candidate is then compared with the bare base
    model). A rejected version in between is never the baseline: it was never
    active.

    A listing that carries ``candidate`` (the registry's own rule: newest
    undecided above active, never a refused version) is followed as is; an
    older registry's listing falls back to the newest version above active."""
    versions = sorted(listing["versions"], key=lambda v: v["ref"]["version"])
    floor = active_version if active_version is not None else 0
    if "candidate" in listing:
        pending = [v for v in versions if v["ref"]["version"] == listing["candidate"]]
    else:
        pending = [v for v in versions if v["ref"]["version"] > floor]
    if not pending:
        raise SystemExit("the registry has no candidate above the active version for this "
                         "cluster: aggregate and publish first")
    if active_version is None:
        return pending[-1], None
    base = next((v for v in versions if v["ref"]["version"] == active_version), None)
    if base is None:
        raise SystemExit(f"the registry reports v{active_version} active but does not list it")
    return pending[-1], base


def evidence_path(out_dir: Path | str, cluster_id: str, version: int) -> Path:
    return Path(out_dir) / f"{cluster_id}-v{version}.json"


def _guard(metrics: Dict) -> List[Dict]:
    """One version's guard as ``GuardMetrics.to_json`` carries it (contracts 1.2.0)."""
    return [{"benchmark": metrics["benchmark"],
             "edit_similarity": float(metrics["edit_similarity"]),
             "exact_match": float(metrics["exact_match"]),
             "n_examples": int(metrics["n_examples"])}]


def promote_body(cluster_id: str, cand: Dict, cand_scores: Dict, base_scores: Dict,
                 *, band: Dict) -> Dict:
    """What ``POST /adapters/{name}/promote`` is sent for this candidate, signed.

    The evaluator is the one that measured, so it is the one that signs (U4):
    an HMAC over the adapter name, the version, the sha256 of the bytes that
    were scored, and every number. The panel forwards this block as it is; the
    registry verifies it, so a metric changed on the way, or a result attached
    to other bytes, is refused. The secret is ``CLASP_EVAL_HMAC_SECRET`` or the
    file ``CLASP_EVAL_HMAC_SECRET_FILE`` names; with neither, ``eval_signature``
    is None and a registry without the secret accepts the result with a warning.
    """
    from security.auth.hmac_signer import secret_from_env, sign_evaluation

    name, version, sha = cand["ref"]["name"], cand["ref"]["version"], cand["sha256"]
    result = {
        "adapter": {"name": name, "version": version, "kind": "cluster", "cluster_id": cluster_id},
        "in_project": cand_scores["in_project"],
        "guard": _guard(cand_scores["guard"]),
        "baseline_in_project": base_scores["in_project"],
        "baseline_noise_band": float(band["value"]),
        "seed": 0,
    }
    baseline_guard = _guard(base_scores["guard"])
    secret = secret_from_env("CLASP_EVAL_HMAC_SECRET")
    signature = (sign_evaluation(adapter=name, version=version, candidate_sha256=sha,
                                 eval_result=result, baseline_guard=baseline_guard, secret=secret)
                 if secret else None)
    return {"eval": result, "baseline_guard": baseline_guard, "candidate_sha256": sha,
            "eval_signature": signature}


def evidence_document(cluster_id: str, cand: Dict, base: Dict | None, cand_scores: Dict,
                      base_scores: Dict, *, band: Dict, details: Dict) -> Dict:
    """The file demo_ui.evidence reads (the CLASP_EVAL_EVIDENCE shape, plus
    which exact versions were measured, plus the signed promote body). ``base``
    is None when the baseline was the bare base model."""
    from evaluation.eval_harness.line_guard import paired_noise

    guard_noise = float(paired_noise(base_scores["guard_rows"], cand_scores["guard_rows"]))
    guard_sha = cand_scores.get("guard_examples_sha256")
    return {
        "promote": promote_body(cluster_id, cand, cand_scores, base_scores, band=band),
        "source": (f"scripts/evaluate_candidate.py on {platform.node()}, "
                   f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}"),
        "candidate_version": cand["ref"]["version"], "candidate_sha256": cand["sha256"],
        "baseline_version": base["ref"]["version"] if base else None,
        "baseline_sha256": base["sha256"] if base else None,
        "clusters": {cluster_id: {
            "in_project": cand_scores["in_project"],
            "baseline_in_project": base_scores["in_project"],
            "guard": _guard(cand_scores["guard"]),
            "baseline_guard": _guard(base_scores["guard"]),
            "guard_noise": guard_noise,
            "guard_examples_sha256": guard_sha,
            "noise_band": float(band["value"]),
        }},
        "noise_band": band,
        "guard_noise": guard_noise,
        "guard_examples_sha256": guard_sha,
        "details": details,
    }


def write_evidence(out_dir: Path | str, cluster_id: str, doc: Dict) -> Path:
    path = evidence_path(out_dir, cluster_id, doc["candidate_version"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)  # the panel never reads a half-written file
    return path


# --------------------------------------------------------------------------- #
# the GPU part
# --------------------------------------------------------------------------- #
def _registry(url: str):
    from edge.registry_client import RegistryClient

    session = None
    if url.startswith("https"):
        from edge import transport

        env = os.environ
        session = transport.mtls_session(client_cert=env["CLASP_TLS_CLIENT_CERT"],
                                         client_key=env["CLASP_TLS_CLIENT_KEY"],
                                         ca_cert=env["CLASP_TLS_CA"])
    return RegistryClient(base_url=url, session=session)


def _composite(rc, name: str, version: int, alpha: float, work: Path) -> Path:
    """base + alpha * cluster vN as one PEFT adapter (what the edges train on)."""
    from edge.merge import compose, load_adapter, save_adapter

    raw = work / f"{name}-v{version}"
    rc.materialize(name, raw, version=version)
    sd, cfg = load_adapter(raw)
    csd, ccfg = compose([(sd, cfg, alpha)])
    out = work / f"{name}-v{version}-alpha{alpha:g}"
    save_adapter(out, csd, ccfg)
    return out


def score_version(state: Dict, rc, name: str, meta: Dict | None, args, work: Path) -> Dict:
    """In-project + line-completion guard for one version, cached by its sha256
    and the guard set's fingerprint. ``meta=None`` scores the bare base model
    (no adapter loaded), cached under the key ``"base"`` (C7)."""
    from edge.completion_eval import evaluate_completion
    from edge.train_client import evaluate as perplexity
    from evaluation.completion import examples_fingerprint
    from evaluation.eval_harness.line_guard import guard_metrics

    bare = meta is None
    version, sha = (None, "base") if bare else (meta["ref"]["version"], meta["sha256"])
    label = "base model" if bare else f"v{version}"
    guard_examples = state["guard_examples"]
    guard_sha = examples_fingerprint(guard_examples)
    key = "base" if bare else f"{sha}-alpha{args.alpha:g}"
    cache = work / "cache" / (f"{key}-guard{guard_sha[:12]}-"
                              f"{state['client'].replace('/', '_')}.json")
    if cache.exists():
        log(f"  {label} ({sha[:12]}) already scored: {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))
    t0 = time.time()

    def measure(model):
        held = state["held_chunks"][: args.ppl_max_blocks]
        ppl = perplexity(model, held, state["tokenizer"].pad_token_id)
        log(f"  {label}: held-out perplexity {ppl['perplexity']} over {len(held)} blocks")
        completion = evaluate_completion(model, state["tokenizer"], state["examples"],
                                         ppl["perplexity"])
        log(f"  {label}: edit similarity {completion['in_project']['edit_similarity']:.4f}, "
            f"exact match {completion['in_project']['exact_match']:.4f} "
            f"({len(state['examples'])} examples, {completion['wall_seconds']} s)")
        scored = evaluate_completion(model, state["tokenizer"], guard_examples, ppl["perplexity"])
        guard = guard_metrics(scored["in_project"], len(guard_examples))
        log(f"  {label}: line-completion guard edit similarity {guard['edit_similarity']:.4f}, "
            f"exact match {guard['exact_match']:.4f} "
            f"({guard['n_examples']} examples, {scored['wall_seconds']} s)")
        return held, completion, scored, guard

    if bare:
        if state["peft"] is None:
            model = state["model"]
            model.eval()
            held, completion, scored, guard = measure(model)
        else:  # adapters are already injected: switch them off for the base model
            with state["peft"].disable_adapter():
                held, completion, scored, guard = measure(state["peft"])
    else:
        adapter_dir = _composite(rc, name, version, args.alpha, work)
        model = state["model"]
        if state["peft"] is None:
            from peft import PeftModel

            model = state["peft"] = PeftModel.from_pretrained(model, str(adapter_dir),
                                                              adapter_name=label)
        else:
            model = state["peft"]
            model.load_adapter(str(adapter_dir), adapter_name=label)
        model.set_adapter(label)
        model.eval()
        held, completion, scored, guard = measure(model)
        if len(model.peft_config) > 1:
            model.delete_adapter(label)
    result = {"version": version, "sha256": sha, "in_project": completion["in_project"],
              "guard": guard, "guard_rows": [r["edit_similarity"] for r in scored["rows"]],
              "guard_examples_sha256": guard_sha, "perplexity_blocks": len(held),
              "examples_sha256": completion["examples_sha256"],
              "wall_seconds": round(time.time() - t0, 1)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="score the registry's candidate and its baseline (D5)")
    ap.add_argument("--cluster", required=True, help="cluster id, e.g. web")
    ap.add_argument("--registry-url", default=os.environ.get("CLASP_REGISTRY_URL",
                                                             "http://localhost:8004"))
    ap.add_argument("--registry-name", default=None, help="default cluster-<cluster>")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--alpha", type=float, default=0.5, help="D3 composition coefficient")
    ap.add_argument("--client", default=None,
                    help="'<cluster>/<client>' whose held-out files to score; default: the one "
                         "the D5 noise band was measured on")
    ap.add_argument("--max-examples", type=int, default=60)
    ap.add_argument("--ppl-max-blocks", type=int, default=16)
    ap.add_argument("--profile", default="dev")
    args = ap.parse_args(argv)
    name = args.registry_name or f"cluster-{args.cluster}"
    t0 = time.time()

    rc = _registry(args.registry_url)
    listing = rc.list_versions(name)
    cand, base = pick_versions(listing, listing.get("active"))
    against = (f"baseline v{base['ref']['version']} ({base['sha256'][:12]})" if base
               else "the bare base model (nothing active yet)")
    log(f"{name}: candidate v{cand['ref']['version']} ({cand['sha256'][:12]}) against "
        f"{against}, alpha {args.alpha:g}")

    from demo_ui.evidence import noise_band
    band = noise_band(args.cluster)
    client = args.client or band.get("client")
    if not client:
        raise SystemExit(f"no noise-band client for {args.cluster!r}: pass --client")

    from edge.chunking import DEFAULT_CORPUS_ROOT, DEFAULT_SEQ_LEN, pack_client
    from edge.completion_eval import client_examples
    from edge.model_loader import load_model
    from edge.train_client import set_determinism
    from evaluation.eval_harness.line_guard import load_guard_set

    client_dir = DEFAULT_CORPUS_ROOT / client
    examples = client_examples(client_dir, max_examples=args.max_examples)
    if not examples:
        raise SystemExit(f"no held-out examples under {client_dir}/held_out")
    guard_examples = load_guard_set()
    log(f"in-project: {client}'s held-out files, {len(examples)} examples; "
        f"guard: line completion, {len(guard_examples)} frozen examples")

    set_determinism(0)
    model, tokenizer, _ = load_model(args.profile)
    _, held = pack_client(client_dir, tokenizer, seq_len=DEFAULT_SEQ_LEN)
    state = {"model": model, "tokenizer": tokenizer, "peft": None, "client": client,
             "examples": examples, "guard_examples": guard_examples,
             "held_chunks": held["chunks"]}
    work = args.out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    scores = {}
    for role, meta in (("baseline", base), ("candidate", cand)):
        log(f"scoring the {role}, " + (f"v{meta['ref']['version']}" if meta else "bare base model"))
        scores[role] = score_version(state, rc, name, meta, args, work)

    doc = evidence_document(args.cluster, cand, base, scores["candidate"], scores["baseline"],
                            band=band, details={
                                "alpha": args.alpha, "client": client,
                                "composition": "base + alpha * cluster (D3/D6), as the edges train",
                                "candidate": scores["candidate"], "baseline": scores["baseline"],
                                "wall_seconds": round(time.time() - t0, 1)})
    path = write_evidence(args.out_dir, args.cluster, doc)
    c, b = scores["candidate"], scores["baseline"]
    log(f"edit similarity {c['in_project']['edit_similarity']:.4f} vs "
        f"{b['in_project']['edit_similarity']:.4f}, line-completion guard "
        f"{c['guard']['edit_similarity']:.4f} vs {b['guard']['edit_similarity']:.4f} "
        f"(paired noise {doc['guard_noise']:.4f}), noise band {band['value']} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
