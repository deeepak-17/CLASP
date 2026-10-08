// Contract check: the committed personalization / lineage / noise feeds written by the
// Python exporters must pass the dashboard's runtime guards. Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { contributionByClient, isPersonalizationFeed, shortClient } from "../src/lib/personalization.ts";
import { isLineageFeed, lineageColumns, parentsOf } from "../src/lib/lineage.ts";
import { isNoiseReport } from "../src/lib/noise.ts";

const load = (name: string): unknown =>
  JSON.parse(readFileSync(new URL(`../../results/${name}`, import.meta.url), "utf8"));

test("personalization.json satisfies its guard", () => {
  const feed = load("personalization.json");
  assert.ok(isPersonalizationFeed(feed));
  if (!isPersonalizationFeed(feed)) return;
  const rows = contributionByClient(feed);
  assert.equal(rows.length, 6);
  assert.ok(rows.every((r) => "r1" in r && "r2" in r));
  assert.equal(shortClient("web/client-flask"), "flask");
});

test("lineage.json satisfies its guard and every edge resolves", () => {
  const feed = load("lineage.json");
  assert.ok(isLineageFeed(feed));
  if (!isLineageFeed(feed)) return;
  assert.ok(lineageColumns(feed).length >= 4);
  const parents = parentsOf(feed, "r2:web/client-flask");
  assert.deepEqual(parents.map((p) => p.kind), ["trained_on"]);
});

test("noise_report.json satisfies its guard", () => {
  assert.ok(isNoiseReport(load("noise_report.json")));
});

test("guards reject wrong shapes", () => {
  assert.equal(isPersonalizationFeed({ feed_version: "2.0.0", rounds: [], comparisons: [] }), false);
  assert.equal(
    isLineageFeed({ feed_version: "1.0.0", rounds: [1], nodes: [], edges: [{ source: "a", target: "b" }] }),
    false,
  );
  assert.equal(isNoiseReport({}), false);
});
