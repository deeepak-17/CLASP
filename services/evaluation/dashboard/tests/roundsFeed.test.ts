// Contract check: a rounds.json written by src/evaluation/eval_harness/round_feed.py must pass
// the dashboard's runtime guard. Run: ROUNDS_JSON=<path> npm test
// (Node >= 23 runs this TypeScript directly.)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { activeLabel, candidateTrend, clusterIds, decisionWord, isRoundsFeed } from "../src/lib/roundsFeed.ts";

const path = process.env.ROUNDS_JSON;

test("exporter output satisfies the dashboard guard", { skip: !path && "set ROUNDS_JSON" }, () => {
  const feed: unknown = JSON.parse(readFileSync(path as string, "utf8"));
  assert.ok(isRoundsFeed(feed));
  if (!isRoundsFeed(feed)) return;
  assert.equal(candidateTrend(feed).length, feed.rounds.length);
  assert.ok(clusterIds(feed).length > 0);
});

const side = (es: number) => ({ edit_similarity: es, exact_match: 0.1, n_examples: 300 });
const guard = (extra: object) => ({
  available: true, reason: "", candidate_pass_at_1: null, baseline_pass_at_1: null, ...extra,
});
const feedWith = (g: object) => ({
  feed_version: "1.0.0", generated_at: "x", guard_pass_at_1_tolerance: 0.02,
  rounds: [{ round: 1, noise_band: 0, guard: g, clusters: [] }], alerts: [],
});

test("guard accepts line-completion and old HumanEval guards, rejects a malformed one", () => {
  assert.equal(isRoundsFeed(feedWith(guard({ kind: "line_completion", candidate: side(0.6), baseline: side(0.61) }))), true);
  assert.equal(isRoundsFeed(feedWith(guard({ kind: "humaneval", candidate_pass_at_1: 0.3, baseline_pass_at_1: 0.31 }))), true);
  assert.equal(isRoundsFeed(feedWith(guard({ kind: "line_completion", candidate: { exact_match: 0.1 } }))), false);
});

test("guard rejects wrong shapes", () => {
  assert.equal(isRoundsFeed({}), false);
  assert.equal(isRoundsFeed({ feed_version: "2.0.0", rounds: [], alerts: [] }), false);
  assert.equal(isRoundsFeed({ feed_version: "1.0.0", rounds: [{ round: 1 }], alerts: [] }), false);
  assert.equal(isRoundsFeed({ feed_version: "1.0.0", rounds: [], alerts: [] }), true);
});

const cluster = (activeAfter: number | null, baselineVersion: number | null) => ({
  cluster_id: "web", registry_name: "cluster-web", representative_client: null,
  baseline: { version: baselineVersion, aggregation: "base model", sha256: null, in_project: null },
  candidate: { version: 1, aggregation: "svd_exact", sha256: "a", in_project: null },
  edit_similarity_delta: null,
  decision: { action: "not_sent", active_version_after: activeAfter, reason: "" },
  authoritative: false,
});
const feedWithCluster = (c: object) => ({
  feed_version: "1.0.0", generated_at: "x", guard_pass_at_1_tolerance: 0.02,
  rounds: [{ round: 1, noise_band: 0, guard: guard({}), clusters: [c] }], alerts: [],
});

test("nothing active is a valid round: a null pointer and a bare-base baseline", () => {
  assert.equal(isRoundsFeed(feedWithCluster(cluster(null, null))), true);
  assert.equal(isRoundsFeed(feedWithCluster(cluster(2, 1))), true);
  assert.equal(isRoundsFeed(feedWithCluster(cluster("v2" as unknown as number, 1))), false);
});

test("the badge says nothing active instead of vnull, and REJECT for a refusal", () => {
  assert.equal(activeLabel(null), "nothing active");
  assert.equal(activeLabel(0), "nothing active");
  assert.equal(activeLabel(3), "active v3");
  assert.equal(decisionWord("reject"), "REJECT");
  assert.equal(decisionWord("rollback"), "ROLLBACK");
  assert.equal(decisionWord("not_sent"), "NOT SENT");
});
