// Record identity and labels for results.json records. Dependency-free so it can
// be unit-run directly under Node (tests/feeds.test.ts).
import type { ResultRecord } from "./types";

/** Stable identity of one record: a run can score several benchmarks, and a
 * benchmark can appear in several runs (e.g. a DEMO_TEST mock and a REAL anchor). */
export function recordKey(r: ResultRecord): string {
  return `${r.eval_result.run_id ?? "run"}-${r.eval_result.benchmark}`;
}

/** Human label that never lets a REAL and a DEMO_TEST number share a name. */
export function recordLabel(r: ResultRecord): string {
  return r.run_metadata.provenance === "REAL"
    ? `${r.eval_result.benchmark} · REAL (${r.eval_result.num_tasks} tasks)`
    : `${r.eval_result.benchmark} · DEMO/TEST mock`;
}
