// Shape of results/lineage.json — written by src/evaluation/eval_harness/lineage.py
// (scripts/export_lineage.py). Keep in sync with FEED_VERSION there.

export type NodeKind = "client" | "cluster" | "composite";

export interface LineageNode {
  id: string;
  kind: NodeKind;
  label: string;
  cluster: string;
  round: number;
  trained_on?: string;
  aggregation?: string;
  sha256?: string | null;
  source?: "edge" | "registry";
  rel_err_vs_exact?: number | null;
  alpha?: number | null;
  composite_ppl?: number | null;
  base_ppl?: number | null;
}

export interface LineageEdge {
  source: string;
  target: string;
  kind: "aggregated_into" | "trained_on" | "composed_into";
}

export interface LineageFeed {
  feed_version: string;
  generated_at: string;
  rounds: number[];
  nodes: LineageNode[];
  edges: LineageEdge[];
  sources?: string[];
}

export function isLineageFeed(x: unknown): x is LineageFeed {
  if (typeof x !== "object" || x === null) return false;
  const r = x as Record<string, unknown>;
  if (typeof r.feed_version !== "string" || !r.feed_version.startsWith("1.")) return false;
  if (!Array.isArray(r.nodes) || !Array.isArray(r.edges) || !Array.isArray(r.rounds)) return false;
  const ids = new Set<string>();
  for (const n of r.nodes as Record<string, unknown>[]) {
    if (typeof n?.id !== "string" || typeof n.kind !== "string" || typeof n.round !== "number") return false;
    ids.add(n.id);
  }
  // Every edge must reference a node that exists, or the drawing would dangle.
  return (r.edges as Record<string, unknown>[]).every(
    (e) => typeof e?.source === "string" && ids.has(e.source) && typeof e.target === "string" && ids.has(e.target),
  );
}

/** Columns left to right: each round's clients, then the clusters built from them, then composites. */
export function lineageColumns(feed: LineageFeed): { title: string; nodes: LineageNode[] }[] {
  const columns: { title: string; nodes: LineageNode[] }[] = [];
  for (const round of feed.rounds) {
    const pick = (kind: NodeKind) =>
      feed.nodes
        .filter((n) => n.kind === kind && n.round === round)
        .sort((a, b) => a.cluster.localeCompare(b.cluster) || a.id.localeCompare(b.id));
    const clients = pick("client");
    const clusters = pick("cluster");
    const composites = pick("composite");
    if (clients.length) columns.push({ title: `Round ${round} client adapters`, nodes: clients });
    if (clusters.length) columns.push({ title: `Round ${round} cluster adapters`, nodes: clusters });
    if (composites.length) columns.push({ title: `Round ${round} composites`, nodes: composites });
  }
  return columns;
}

export function parentsOf(feed: LineageFeed, id: string): { node: LineageNode; kind: LineageEdge["kind"] }[] {
  const byId = new Map(feed.nodes.map((n) => [n.id, n]));
  return feed.edges.filter((e) => e.target === id).map((e) => ({ node: byId.get(e.source)!, kind: e.kind }));
}
