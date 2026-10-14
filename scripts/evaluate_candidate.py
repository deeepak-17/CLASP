"""Evaluate the candidate the registry is about to judge, on this laptop's GPU (D5).

    python scripts/evaluate_candidate.py --cluster web --registry-url http://localhost:8004

Promote (D5) compares the version the registry is serving with the one before
it. This measures exactly those two, as the edges use them — ``base + alpha *
cluster`` (D3/D6) — so the decision is about the candidate itself, not about an
earlier round's adapter:

* **in-project** (D5's primary metric): next-line completion (edit similarity,
  exact match) on the held-out files of the client the noise band was
  measured on (``demo_ui.evidence.noise_band``), plus held-out perplexity;
  ``edge.completion_eval``, greedy, the same examples for both versions;
* **HumanEval guard**: the frozen first-20 subset, greedy, generated with
  ``edge.humaneval_baseline.generate_completions`` and scored on P5's executor
  against the original HumanEval tests (``evaluation.eval_harness.guard_anchor``).

Each version is scored once: results are cached by the payload's sha256, so
the previous candidate is the next round's baseline for free. The output,
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
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # demo_ui.evidence, for the noise band

DEFAULT_OUT = REPO_ROOT / ".runtime" / "live-eval"
GUARD_BENCHMARK = "HumanEval"
GUARD_N = 20  # the frozen subset base_anchor.json was scored on


def log(msg: str) -> None:
    print(msg, flush=True)


def pick_versions(listing: Dict) -> Tuple[Dict, Dict]:
    """(candidate, baseline): the registry's active version and the one before it."""
    active = listing.get("active")
    if active is None:
        raise SystemExit("the registry has no active version for this cluster: publish first")
    versions = sorted(listing["versions"], key=lambda v: v["ref"]["version"])
    cand = next(v for v in versions if v["ref"]["version"] == active)
    earlier = [v for v in versions if v["ref"]["version"] < active]
    if not earlier:
        raise SystemExit(f"v{active} has no previous version to compare with (D5): publish a "
                         f"second round first")
    return cand, earlier[-1]


def evidence_path(out_dir: Path | str, cluster_id: str, version: int) -> Path:
    return Path(out_dir) / f"{cluster_id}-v{version}.json"


def _guard(pass_at_1: float):
    return [{"benchmark": GUARD_BENCHMARK, "pass_at_k": {"1": float(pass_at_1)}}]


def evidence_document(cluster_id: str, cand: Dict, base: Dict, cand_scores: Dict,
                      base_scores: Dict, *, band: Dict, details: Dict) -> Dict:
    """The file demo_ui.evidence reads (the CLASP_EVAL_EVIDENCE shape, plus
    which exact versions were measured)."""
    return {
        "source": (f"scripts/evaluate_candidate.py on {platform.node()}, "
                   f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}"),
        "candidate_version": cand["ref"]["version"], "candidate_sha256": cand["sha256"],
        "baseline_version": base["ref"]["version"], "baseline_sha256": base["sha256"],
        "clusters": {cluster_id: {
            "in_project": cand_scores["in_project"],
            "baseline_in_project": base_scores["in_project"],
            "guard": _guard(cand_scores["pass_at_1"]),
            "baseline_guard": _guard(base_scores["pass_at_1"]),
            "noise_band": float(band["value"]),
        }},
        "noise_band": band,
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


def score_version(state: Dict, rc, name: str, meta: Dict, args, work: Path) -> Dict:
    """In-project + HumanEval for one version, cached by its sha256."""
    from edge.completion_eval import evaluate_completion
    from edge.humaneval_baseline import generate_completions
    from edge.train_client import evaluate as perplexity
    from evaluation.eval_harness.guard_anchor import build_anchor, score_samples

    version, sha = meta["ref"]["version"], meta["sha256"]
    cache = work / "cache" / f"{sha}-alpha{args.alpha:g}-{state['client'].replace('/', '_')}.json"
    if cache.exists():
        log(f"  v{version} ({sha[:12]}) already scored: {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))
    t0 = time.time()
    adapter_dir = _composite(rc, name, version, args.alpha, work)
    model = state["model"]
    label = f"v{version}"
    if state["peft"] is None:
        from peft import PeftModel

        model = state["peft"] = PeftModel.from_pretrained(model, str(adapter_dir), adapter_name=label)
    else:
        model = state["peft"]
        model.load_adapter(str(adapter_dir), adapter_name=label)
    model.set_adapter(label)
    model.eval()

    held = state["held_chunks"][: args.ppl_max_blocks]
    ppl = perplexity(model, held, state["tokenizer"].pad_token_id)
    log(f"  v{version}: held-out perplexity {ppl['perplexity']} over {len(held)} blocks")
    completion = evaluate_completion(model, state["tokenizer"], state["examples"], ppl["perplexity"])
    log(f"  v{version}: edit similarity {completion['in_project']['edit_similarity']:.4f}, "
        f"exact match {completion['in_project']['exact_match']:.4f} "
        f"({len(state['examples'])} examples, {completion['wall_seconds']} s)")

    problems = {t: {"prompt": state["tasks"][t].prompt} for t in state["task_ids"]}
    samples = generate_completions(model, state["tokenizer"], problems, state["task_ids"],
                                   max_new_tokens=args.max_new_tokens)
    samples_path = work / f"{sha}-humaneval.jsonl"
    samples_path.write_text("".join(json.dumps(s) + "\n" for s in samples), encoding="utf-8")
    outcomes = score_samples(samples, state["tasks"], timeout_seconds=10.0, sandbox="auto")
    anchor = build_anchor(outcomes, samples_path=samples_path,
                          generation_manifest={"subset": {"n": len(state["task_ids"]),
                                                          "task_ids": state["task_ids"]}})
    log(f"  v{version}: HumanEval pass@1 {anchor['base_pass_at_1']} "
        f"({anchor['base_solved']}/{anchor['n_problems']})")
    if len(model.peft_config) > 1:
        model.delete_adapter(label)
    result = {"version": version, "sha256": sha, "in_project": completion["in_project"],
              "pass_at_1": anchor["base_pass_at_1"], "humaneval_solved": anchor["base_solved"],
              "humaneval_n": anchor["n_problems"], "perplexity_blocks": len(held),
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
    ap.add_argument("--max-new-tokens", type=int, default=384, help="HumanEval generation cap")
    ap.add_argument("--profile", default="dev")
    args = ap.parse_args(argv)
    name = args.registry_name or f"cluster-{args.cluster}"
    t0 = time.time()

    rc = _registry(args.registry_url)
    cand, base = pick_versions(rc.list_versions(name))
    log(f"{name}: candidate v{cand['ref']['version']} ({cand['sha256'][:12]}) against "
        f"baseline v{base['ref']['version']} ({base['sha256'][:12]}), alpha {args.alpha:g}")

    from demo_ui.evidence import noise_band
    band = noise_band(args.cluster)
    client = args.client or band.get("client")
    if not client:
        raise SystemExit(f"no noise-band client for {args.cluster!r}: pass --client")

    from edge.chunking import DEFAULT_CORPUS_ROOT, DEFAULT_SEQ_LEN, pack_client
    from edge.completion_eval import client_examples
    from edge.model_loader import load_model
    from edge.train_client import set_determinism
    from evaluation.eval_harness.harness import load_evaluation_config
    from evaluation.eval_harness.registry import build_adapter
    from evaluation.interfaces.contracts import BenchmarkName

    client_dir = DEFAULT_CORPUS_ROOT / client
    examples = client_examples(client_dir, max_examples=args.max_examples)
    if not examples:
        raise SystemExit(f"no held-out examples under {client_dir}/held_out")
    tasks = {t.task_id: t for t in build_adapter(BenchmarkName.HUMANEVAL,
                                                 load_evaluation_config(None)).load_tasks()}
    task_ids = sorted(tasks, key=lambda t: int(t.split("/")[1]))[:GUARD_N]
    log(f"in-project: {client}'s held-out files, {len(examples)} examples; "
        f"guard: HumanEval/0..{GUARD_N - 1}")

    set_determinism(0)
    model, tokenizer, _ = load_model(args.profile)
    _, held = pack_client(client_dir, tokenizer, seq_len=DEFAULT_SEQ_LEN)
    state = {"model": model, "tokenizer": tokenizer, "peft": None, "client": client,
             "examples": examples, "held_chunks": held["chunks"], "tasks": tasks,
             "task_ids": task_ids}
    work = args.out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    scores = {}
    for role, meta in (("baseline", base), ("candidate", cand)):
        log(f"scoring the {role}, v{meta['ref']['version']}")
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
        f"{b['in_project']['edit_similarity']:.4f}, HumanEval {c['pass_at_1']} vs {b['pass_at_1']}, "
        f"noise band {band['value']} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
