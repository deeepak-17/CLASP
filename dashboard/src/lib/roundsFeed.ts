// Shape of eval_harness/results/rounds.json — written by eval_harness/round_feed.py
// (scripts/export_round_feed.py) from scripts/demo_round.py's round manifests.
// Keep in sync with FEED_VERSION there. Dependency-free so it can be unit-run
// directly under Node (dashboard/tests/roundsFeed.test.ts).

export interface InProjectBlock {
  edit_similarity: number;
  exact_match: number;
  perplexity: number;
  n_examples: number;
  examples_sha256: string | null;
}

export interface VersionBlock {
  version: number;
  aggregation: string;
  sha256: string | null;
  /** null = the round did not measure it (never a fabricated zero). */
  in_project: InProjectBlock | null;
}

export interface ClusterRound {
  cluster_id: string;
  registry_name: string;
  representative_client: string | null;
  baseline: VersionBlock;
  candidate: VersionBlock;
  edit_similarity_delta: number | null;
  decision: { action: string; active_version_after: number; reason: string };
  authoritative: boolean;
}

export interface RoundEntry {
  round: number;
  finished_utc: string | null;
  wall_minutes: number | null;
  nfr_round_minutes: number | null;
  nfr_met: boolean | null;
  noise_band: number;
  counters: Record<string, number>;
  guard: {
    available: boolean;
    reason: string;
    candidate_pass_at_1: number | null;
    baseline_pass_at_1: number | null;
  };
  clusters: ClusterRound[];
}

export interface RoundAlert {
  round: number;
  cluster_id: string;
  kind: string;
  message: string;
}

export interface RoundsFeed {
  feed_version: string;
  generated_at: string;
  guard_pass_at_1_tolerance: number;
  rounds: RoundEntry[];
  alerts: RoundAlert[];
}

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null;
const isNumOrNull = (x: unknown) => x === null || typeof x === "number";

function isVersionBlock(x: unknown): x is VersionBlock {
  if (!isObj(x)) return false;
  const ip = x.in_project;
  return (
    typeof x.version === "number" &&
    typeof x.aggregation === "string" &&
    (ip === null || (isObj(ip) && typeof ip.edit_similarity === "number" && typeof ip.exact_match === "number"))
  );
}

function isClusterRound(x: unknown): x is ClusterRound {
  if (!isObj(x) || !isObj(x.decision)) return false;
  return (
    typeof x.cluster_id === "string" &&
    typeof x.registry_name === "string" &&
    isVersionBlock(x.baseline) &&
    isVersionBlock(x.candidate) &&
    isNumOrNull(x.edit_similarity_delta) &&
    typeof x.decision.action === "string" &&
    typeof x.decision.active_version_after === "number" &&
    typeof x.authoritative === "boolean"
  );
}

function isRoundEntry(x: unknown): x is RoundEntry {
  if (!isObj(x) || !isObj(x.guard)) return false;
  return (
    typeof x.round === "number" &&
    typeof x.noise_band === "number" &&
    typeof x.guard.available === "boolean" &&
    isNumOrNull(x.guard.candidate_pass_at_1) &&
    isNumOrNull(x.guard.baseline_pass_at_1) &&
    Array.isArray(x.clusters) &&
    x.clusters.every(isClusterRound)
  );
}

export function isRoundsFeed(x: unknown): x is RoundsFeed {
  if (!isObj(x)) return false;
  return (
    typeof x.feed_version === "string" &&
    x.feed_version.startsWith("1.") &&
    Array.isArray(x.rounds) &&
    x.rounds.every(isRoundEntry) &&
    Array.isArray(x.alerts) &&
    x.alerts.every((a) => isObj(a) && typeof a.kind === "string" && typeof a.message === "string")
  );
}

/** One row per round: round + candidate edit_similarity per cluster (null when unmeasured). */
export function candidateTrend(feed: RoundsFeed): Array<Record<string, number | null>> {
  return feed.rounds.map((r) => {
    const row: Record<string, number | null> = { round: r.round };
    for (const c of r.clusters) row[c.cluster_id] = c.candidate.in_project?.edit_similarity ?? null;
    return row;
  });
}

export function clusterIds(feed: RoundsFeed): string[] {
  return Array.from(new Set(feed.rounds.flatMap((r) => r.clusters.map((c) => c.cluster_id)))).sort();
}
