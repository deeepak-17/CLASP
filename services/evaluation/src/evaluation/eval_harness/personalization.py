"""Personalization results — edge round manifests -> ``personalization.json``.

The edge lane (``edge.round``) writes one ``round_manifest.json`` per federated
round: for every client, held-out perplexity of the frozen base, of the
client-only adapter, and of the three-layer composite
``base + alpha*cluster + beta*client`` (D6), all on that client's own held-out
files. That is the personalization measurement. This module reads one or more
of those manifests and produces the compact, dashboard- and report-facing
view:

* per client and round: base / client-only / composite perplexity, the
  personalization delta (composite - base), the cluster layer's contribution
  at the reference alpha, the alpha the sweep selected, and the token count
  behind the number;
* per round: how many clients improved, mean / min / max delta, and the
  cluster layer's mean contribution;
* across rounds: the per-client change from one round to the next (the D3
  composition-order comparison is exactly round 1 -> round 2).

Every number is read from the manifest, never recomputed or filled in — a
missing key is an error naming its path, so a change on the producer's side
fails here instead of rendering a wrong chart. Perplexity is lower-is-better:
a negative delta is an improvement.
"""

from __future__ import annotations

from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from evaluation.utils.errors import EvaluationError
from evaluation.utils.timing import utc_timestamp

#: Version of the ``personalization.json`` shape (``dashboard/src/lib/personalization.ts`` mirrors it).
FEED_VERSION = "1.0.0"


def _get(data: Mapping[str, Any], *path: str) -> Any:
    node: Any = data
    for i, key in enumerate(path):
        if not isinstance(node, Mapping) or key not in node:
            raise EvaluationError(f"edge round manifest is missing '{'.'.join(path[: i + 1])}'")
        node = node[key]
    return node


def _client_row(client_id: str, block: Mapping[str, Any]) -> dict[str, Any]:
    def split(key: str) -> Any:
        return _get(block, "full_split", key)

    base = float(split("base_ppl"))
    client_only = float(split("client_only_ppl"))
    composite = float(split("composite_ppl"))
    promotion = block.get("promotion") or {}
    return {
        "client_id": client_id,
        "cluster": str(_get(block, "cluster")),
        "n_train_files": int(_get(block, "n_train_files")),
        "n_held_out_blocks": int(_get(block, "n_held_out_blocks")),
        "n_tokens": int(split("n_tokens")),
        "base_ppl": base,
        "client_only_ppl": client_only,
        "composite_ppl": composite,
        "best_alpha": float(_get(block, "best_alpha")),
        "alpha_ref": float(split("alpha_ref")),
        "alpha_ref_ppl": float(split("alpha_ref_ppl")),
        "personalization_delta_ppl": float(_get(block, "personalization_delta_ppl")),
        "cluster_contribution_at_alpha_ref": float(_get(block, "cluster_contribution_at_alpha_ref")),
        "improved": composite < base,
        "decision": promotion.get("decision"),
        "decision_is_authoritative": bool(promotion.get("decision_is_authoritative", False)),
    }


def summarize_round(manifest: Mapping[str, Any], *, label: str | None = None) -> dict[str, Any]:
    """One edge round manifest -> one ``personalization.json`` round entry."""
    round_no = int(_get(manifest, "round"))
    clients = [_client_row(cid, block) for cid, block in sorted(_get(manifest, "clients").items())]
    if not clients:
        raise EvaluationError(f"edge round manifest for round {round_no} has no clients")
    deltas = [c["personalization_delta_ppl"] for c in clients]
    contributions = [c["cluster_contribution_at_alpha_ref"] for c in clients]
    timings = manifest.get("timings") or {}
    return {
        "round": round_no,
        "label": label or f"round {round_no}",
        "utc": manifest.get("utc"),
        "model_id": manifest.get("model_id"),
        "profile": manifest.get("profile"),
        "seed": manifest.get("seed"),
        "alpha_grid": [float(a) for a in manifest.get("alpha_grid") or []],
        "d3_composition_order": (manifest.get("known_deviations") or {}).get("d3_composition_order"),
        "eval_wall_minutes": timings.get("total_minutes"),
        "summary": {
            "n_clients": len(clients),
            "n_improved": sum(1 for c in clients if c["improved"]),
            "mean_delta_ppl": round(fmean(deltas), 4),
            "min_delta_ppl": min(deltas),
            "max_delta_ppl": max(deltas),
            "mean_cluster_contribution_ppl": round(fmean(contributions), 4),
            "n_cluster_helps": sum(1 for v in contributions if v < 0),
            "n_alpha_zero": sum(1 for c in clients if c["best_alpha"] == 0.0),
        },
        "clients": clients,
    }


def compare_rounds(earlier: Mapping[str, Any], later: Mapping[str, Any]) -> dict[str, Any]:
    """Per-client change between two summarized rounds (clients present in both)."""
    before = {c["client_id"]: c for c in earlier["clients"]}
    rows = []
    for client in later["clients"]:
        prev = before.get(client["client_id"])
        if prev is None:
            continue
        rows.append(
            {
                "client_id": client["client_id"],
                "composite_ppl_change": round(client["composite_ppl"] - prev["composite_ppl"], 4),
                "cluster_contribution_before": prev["cluster_contribution_at_alpha_ref"],
                "cluster_contribution_after": client["cluster_contribution_at_alpha_ref"],
                "best_alpha_before": prev["best_alpha"],
                "best_alpha_after": client["best_alpha"],
            }
        )
    return {
        "from_round": earlier["round"],
        "to_round": later["round"],
        "n_clients": len(rows),
        "n_composite_better": sum(1 for r in rows if r["composite_ppl_change"] < 0),
        "clients": rows,
    }


def build_feed(
    manifests: Iterable[Mapping[str, Any]],
    *,
    labels: Sequence[str | None] | None = None,
    sources: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize every manifest, ordered by round, plus consecutive-round comparisons."""
    manifests = list(manifests)
    labels = list(labels) if labels is not None else [None] * len(manifests)
    if len(labels) != len(manifests):
        raise EvaluationError("labels must match manifests one-to-one")
    rounds = sorted(
        (summarize_round(m, label=lbl) for m, lbl in zip(manifests, labels)), key=lambda r: r["round"]
    )
    seen: set[int] = set()
    for entry in rounds:
        if entry["round"] in seen:
            raise EvaluationError(f"round {entry['round']} supplied more than once")
        seen.add(entry["round"])
    return {
        "feed_version": FEED_VERSION,
        "generated_at": utc_timestamp(),
        "metric": "held-out perplexity on each client's own never-trained-on files (lower is better)",
        "sources": list(sources or []),
        "rounds": rounds,
        "comparisons": [compare_rounds(a, b) for a, b in zip(rounds, rounds[1:])],
    }
