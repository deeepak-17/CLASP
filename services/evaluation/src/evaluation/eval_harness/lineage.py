"""Adapter lineage — which adapter was built from which, across rounds.

Reads edge round manifests and emits a small directed graph the dashboard's
Lineage page draws:

* ``client`` nodes — one per client adapter per round, annotated with what
  it was trained on (the bare base, or the frozen ``base + alpha*cluster`` the
  D3 order prescribes, naming the exact registry version);
* ``cluster`` nodes — the cluster adapter each round composed with: the
  edge-local aggregate when the manifest records no registry provenance, or
  the registry version (name, version, aggregation, sha256) when it does;
* ``composite`` nodes — the per-client three-layer model the round evaluated.

Edges are ``aggregated_into`` (client -> cluster), ``trained_on`` (cluster ->
the next round's client) and ``composed_into`` (client / cluster ->
composite). Every node and edge comes from a manifest field; nothing is
inferred from naming conventions except the client-id -> node-id mapping.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from evaluation.utils.errors import EvaluationError
from evaluation.utils.timing import utc_timestamp

#: Version of the ``lineage.json`` shape (``dashboard/src/lib/lineage.ts`` mirrors it).
FEED_VERSION = "1.0.0"


def _client_node(round_no: int, client_id: str) -> str:
    return f"r{round_no}:{client_id}"


def _cluster_node(round_no: int, cluster: str, provenance: Mapping[str, Any] | None) -> str:
    if provenance:
        return f"registry:{provenance['adapter']}@v{provenance['version']}"
    return f"r{round_no}:cluster-{cluster}"


def _short_client(client_id: str) -> str:
    return client_id.split("/")[-1].removeprefix("client-")


class _Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, str]] = []

    def node(self, node_id: str, **attrs: Any) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].update({k: v for k, v in attrs.items() if v is not None})
        else:
            self.nodes[node_id] = {"id": node_id, **attrs}

    def edge(self, source: str, target: str, kind: str) -> None:
        entry = {"source": source, "target": target, "kind": kind}
        if entry not in self.edges:
            self.edges.append(entry)


def build_lineage(manifests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Lineage graph over one or more edge round manifests."""
    graph = _Graph()
    rounds: list[int] = []
    for manifest in sorted(manifests, key=lambda m: int(m["round"])):
        try:
            round_no = int(manifest["round"])
            clusters = manifest["clusters"]
            clients = manifest["clients"]
        except KeyError as exc:
            raise EvaluationError(f"edge round manifest is missing {exc}") from exc
        rounds.append(round_no)

        cluster_ids: dict[str, str] = {}
        for cluster, block in sorted(clusters.items()):
            provenance = block.get("provenance")
            node_id = _cluster_node(round_no, cluster, provenance)
            cluster_ids[cluster] = node_id
            if provenance:
                graph.node(
                    node_id,
                    kind="cluster",
                    label=f"{provenance['adapter']} v{provenance['version']}",
                    cluster=cluster,
                    round=int(provenance.get("round", round_no)),
                    aggregation=provenance.get("aggregation"),
                    sha256=provenance.get("sha256_verified"),
                    source="registry",
                )
                for member in provenance.get("source_clients") or []:
                    source_round = int(provenance.get("round", round_no))
                    member_id = f"{cluster}/{member}"
                    graph.node(
                        _client_node(source_round, member_id),
                        kind="client",
                        label=f"{_short_client(member)} r{source_round}",
                        cluster=cluster,
                        round=source_round,
                    )
                    graph.edge(_client_node(source_round, member_id), node_id, "aggregated_into")
            else:
                graph.node(
                    node_id,
                    kind="cluster",
                    label=f"cluster-{cluster} r{round_no}",
                    cluster=cluster,
                    round=round_no,
                    aggregation="svd (edge-local)",
                    rel_err_vs_exact=(block.get("vs_exact_average") or {}).get("mean_rel_err"),
                    source="edge",
                )
                for member in block.get("members") or []:
                    graph.node(
                        _client_node(round_no, member),
                        kind="client",
                        label=f"{_short_client(member)} r{round_no}",
                        cluster=cluster,
                        round=round_no,
                    )
                    graph.edge(_client_node(round_no, member), node_id, "aggregated_into")

        for client_id, block in sorted(clients.items()):
            cluster = block["cluster"]
            client_node = _client_node(round_no, client_id)
            d3 = (block.get("training") or {}).get("d3")
            trained_on = "base"
            if d3:
                trained_on = f"base + {d3['alpha']}·{graph.nodes[cluster_ids[cluster]]['label']}"
                graph.edge(cluster_ids[cluster], client_node, "trained_on")
            graph.node(
                client_node,
                kind="client",
                label=f"{_short_client(client_id)} r{round_no}",
                cluster=cluster,
                round=round_no,
                trained_on=trained_on,
            )
            split = block.get("full_split") or {}
            composite = f"r{round_no}:composite:{client_id}"
            graph.node(
                composite,
                kind="composite",
                label=f"{_short_client(client_id)} composite r{round_no}",
                cluster=cluster,
                round=round_no,
                alpha=block.get("best_alpha"),
                beta=block.get("beta"),
                composite_ppl=split.get("composite_ppl"),
                base_ppl=split.get("base_ppl"),
            )
            graph.edge(client_node, composite, "composed_into")
            graph.edge(cluster_ids[cluster], composite, "composed_into")

    return {
        "feed_version": FEED_VERSION,
        "generated_at": utc_timestamp(),
        "rounds": rounds,
        "nodes": sorted(graph.nodes.values(), key=lambda n: (n["round"], n["kind"], n["id"])),
        "edges": graph.edges,
    }
