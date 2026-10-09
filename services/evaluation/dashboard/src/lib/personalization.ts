// Shape of results/personalization.json — written by
// src/evaluation/eval_harness/personalization.py (scripts/build_personalization_report.py)
// from the edge round manifests. Keep in sync with FEED_VERSION there.
// Dependency-free so it can be unit-run under Node (tests/personalization.test.ts).

export interface ClientRow {
  client_id: string;
  cluster: string;
  n_tokens: number;
  base_ppl: number;
  client_only_ppl: number;
  composite_ppl: number;
  best_alpha: number;
  alpha_ref: number;
  personalization_delta_ppl: number;
  cluster_contribution_at_alpha_ref: number;
  improved: boolean;
  decision: string | null;
  decision_is_authoritative: boolean;
}

export interface RoundSummary {
  n_clients: number;
  n_improved: number;
  mean_delta_ppl: number;
  min_delta_ppl: number;
  max_delta_ppl: number;
  mean_cluster_contribution_ppl: number;
  n_cluster_helps: number;
  n_alpha_zero: number;
}

export interface PersonalizationRound {
  round: number;
  label: string;
  utc: string | null;
  model_id: string | null;
  seed: number | null;
  alpha_grid: number[];
  summary: RoundSummary;
  clients: ClientRow[];
}

export interface RoundComparison {
  from_round: number;
  to_round: number;
  n_clients: number;
  n_composite_better: number;
  clients: {
    client_id: string;
    composite_ppl_change: number;
    cluster_contribution_before: number;
    cluster_contribution_after: number;
    best_alpha_before: number;
    best_alpha_after: number;
  }[];
}

export interface PersonalizationFeed {
  feed_version: string;
  generated_at: string;
  metric: string;
  sources: string[];
  rounds: PersonalizationRound[];
  comparisons: RoundComparison[];
}

function isObject(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null;
}

export function isPersonalizationFeed(x: unknown): x is PersonalizationFeed {
  if (!isObject(x) || typeof x.feed_version !== "string" || !x.feed_version.startsWith("1.")) return false;
  if (!Array.isArray(x.rounds) || !Array.isArray(x.comparisons)) return false;
  return x.rounds.every(
    (r) =>
      isObject(r) &&
      typeof r.round === "number" &&
      isObject(r.summary) &&
      Array.isArray(r.clients) &&
      r.clients.every(
        (c) =>
          isObject(c) &&
          typeof c.client_id === "string" &&
          typeof c.base_ppl === "number" &&
          typeof c.composite_ppl === "number" &&
          typeof c.cluster_contribution_at_alpha_ref === "number",
      ),
  );
}

/** "web/client-flask" -> "flask". */
export function shortClient(clientId: string): string {
  return clientId.split("/").pop()!.replace(/^client-/, "");
}

/** One row per client with the cluster contribution in each round (for the before/after chart). */
export function contributionByClient(feed: PersonalizationFeed): Record<string, number | string>[] {
  const rows = new Map<string, Record<string, number | string>>();
  for (const round of feed.rounds) {
    for (const c of round.clients) {
      const row = rows.get(c.client_id) ?? { client: shortClient(c.client_id) };
      row[`r${round.round}`] = c.cluster_contribution_at_alpha_ref;
      rows.set(c.client_id, row);
    }
  }
  return [...rows.values()];
}
