// Contract check: a rounds.json written by src/evaluation/eval_harness/round_feed.py must pass
// the dashboard's runtime guard. Run: ROUNDS_JSON=<path> npm test
// (Node >= 23 runs this TypeScript directly.)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { candidateTrend, clusterIds, isRoundsFeed } from "../src/lib/roundsFeed.ts";

const path = process.env.ROUNDS_JSON;

test("exporter output satisfies the dashboard guard", { skip: !path && "set ROUNDS_JSON" }, () => {
  const feed: unknown = JSON.parse(readFileSync(path as string, "utf8"));
  assert.ok(isRoundsFeed(feed));
  if (!isRoundsFeed(feed)) return;
  assert.equal(candidateTrend(feed).length, feed.rounds.length);
  assert.ok(clusterIds(feed).length > 0);
});

test("guard rejects wrong shapes", () => {
  assert.equal(isRoundsFeed({}), false);
  assert.equal(isRoundsFeed({ feed_version: "2.0.0", rounds: [], alerts: [] }), false);
  assert.equal(isRoundsFeed({ feed_version: "1.0.0", rounds: [{ round: 1 }], alerts: [] }), false);
  assert.equal(isRoundsFeed({ feed_version: "1.0.0", rounds: [], alerts: [] }), true);
});
