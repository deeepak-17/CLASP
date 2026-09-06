// Data loader — Week 5 Wednesday: "Wire skeleton to read static results.json".
//
//   evaluation/results/results.json  (written by evaluation/results_store.py)
//         |  scripts/sync-results.mjs (plain file copy, run before dev/build)
//         v
//   dashboard/public/data/results.json
//         |  fetch() at runtime, this module
//         v
//   ResultsIndex (typed, validated)
//         |
//         v
//   React state (useResults hook)  ->  pages/components  ->  Recharts
//
// No backend, no API route: this is a static file fetched by the browser.
// Malformed or missing data produces a typed error state, not a crash — see
// isResultsIndex below and its call site in useResults.
import { useEffect, useState } from "react";
import type { AdapterRef, EvalResult, ResultRecord, ResultsIndex, RunMetadata } from "./types";

const DATA_URL = `${import.meta.env.BASE_URL}data/results.json`;

/** Narrow, structural runtime check — enough to catch "wrong shape" without
 * re-implementing full JSON Schema validation client-side (that already
 * happens on the Python side: evaluation/results_store.py::append_results
 * validates every record against interfaces/schemas/eval_result.schema.json
 * before it is ever written). This is a last line of defence against a
 * hand-edited or truncated file reaching the browser. */
function isAdapterRef(x: unknown): x is AdapterRef {
  if (typeof x !== "object" || x === null) return false;
  const r = x as Record<string, unknown>;
  return typeof r.name === "string" && typeof r.version === "number" && typeof r.kind === "string";
}

function isEvalResult(x: unknown): x is EvalResult {
  if (typeof x !== "object" || x === null) return false;
  const r = x as Record<string, unknown>;
  return (
    isAdapterRef(r.adapter) &&
    typeof r.benchmark === "string" &&
    typeof r.pass_at_k === "object" &&
    r.pass_at_k !== null &&
    typeof r.num_tasks === "number" &&
    typeof r.num_samples_per_task === "number"
  );
}

function isRunMetadata(x: unknown): x is RunMetadata {
  if (typeof x !== "object" || x === null) return false;
  const r = x as Record<string, unknown>;
  return (
    typeof r.model_checkpoint === "string" &&
    typeof r.dataset_split === "string" &&
    typeof r.num_problems === "number" &&
    typeof r.generation === "object" &&
    r.generation !== null &&
    (r.provenance === "REAL" || r.provenance === "DEMO_TEST") &&
    typeof r.provenance_note === "string"
  );
}

function isResultRecord(x: unknown): x is ResultRecord {
  if (typeof x !== "object" || x === null) return false;
  const r = x as Record<string, unknown>;
  return isEvalResult(r.eval_result) && isRunMetadata(r.run_metadata);
}

export function isResultsIndex(x: unknown): x is ResultsIndex {
  if (typeof x !== "object" || x === null) return false;
  const r = x as Record<string, unknown>;
  return (
    typeof r.index_version === "string" &&
    typeof r.updated_at === "string" &&
    Array.isArray(r.results) &&
    r.results.every(isResultRecord)
  );
}

export type ResultsState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: ResultsIndex };

/** Fetches and validates results.json once on mount. Every failure mode —
 * network error, HTTP error, invalid JSON, JSON that doesn't match the
 * expected shape — resolves to a typed "error" state with a specific
 * message, never a thrown exception that would blank the page. */
export function useResults(): ResultsState {
  const [state, setState] = useState<ResultsState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const response = await fetch(DATA_URL, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status} fetching ${DATA_URL}`);
        }
        const parsed: unknown = await response.json();
        if (!isResultsIndex(parsed)) {
          throw new Error(
            "results.json does not match the expected ResultsIndex shape " +
              "(see dashboard/src/lib/types.ts). The file may be malformed or stale.",
          );
        }
        if (!cancelled) setState({ status: "ready", data: parsed });
      } catch (err) {
        if (!cancelled) {
          const message =
            err instanceof SyntaxError
              ? `results.json is not valid JSON (${err.message})`
              : err instanceof Error
                ? err.message
                : String(err);
          setState({ status: "error", message });
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}

/** Sorted, de-duplicated list of every k value present across all results
 * (e.g. [1, 10]) — used to build the Pass@k chart's x-axis without assuming
 * a fixed set of k's. */
export function allKValues(results: ResultRecord[]): number[] {
  const ks = new Set<number>();
  for (const record of results) {
    for (const k of Object.keys(record.eval_result.pass_at_k)) {
      ks.add(Number(k));
    }
  }
  return Array.from(ks).sort((a, b) => a - b);
}

/** True only when every record in the index is provenance "REAL". A single
 * DEMO_TEST record is enough to mark the whole view as placeholder data —
 * per the Week-5 instruction, placeholder values must never be presented as
 * if they were real. */
export function isAllReal(results: ResultRecord[]): boolean {
  return results.length > 0 && results.every((r) => r.run_metadata.provenance === "REAL");
}

export function formatPercent(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatOrNA(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") return "N/A";
  return String(value);
}
