// Loader for the in-project completion metric artifact — the D5 primary
// signal (edit similarity / exact match on held-out client code), plus the
// aggregation ablation. Same pattern as loadResults.ts: fetch a static file
// copied from evaluation/results/in_project_metric.json by sync-results.mjs,
// runtime-validate its shape, expose a typed state to the page.
import { useEffect, useState } from "react";

export interface AblationRow {
  cluster: string;
  naive_avg: number;
  svd: number;
  ratio: string;
}

export interface CompletionRow {
  client: string;
  cluster: string;
  version: string;
  edit_similarity: number;
  exact_match: number;
}

export interface InProjectData {
  environment: string;
  round_shape: string;
  held_out_examples: number;
  decoding: string;
  note: string;
  aggregation_ablation: { caption: string; rows: AblationRow[] };
  in_project_completion: { caption: string; rows: CompletionRow[] };
  takeaway: string;
}

const DATA_URL = `${import.meta.env.BASE_URL}data/in_project_metric.json`;

function isInProjectData(x: unknown): x is InProjectData {
  if (typeof x !== "object" || x === null) return false;
  const r = x as Record<string, unknown>;
  return (
    typeof r.environment === "string" &&
    typeof r.note === "string" &&
    typeof r.takeaway === "string" &&
    typeof r.aggregation_ablation === "object" &&
    r.aggregation_ablation !== null &&
    Array.isArray((r.aggregation_ablation as Record<string, unknown>).rows) &&
    typeof r.in_project_completion === "object" &&
    r.in_project_completion !== null &&
    Array.isArray((r.in_project_completion as Record<string, unknown>).rows)
  );
}

export type InProjectState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: InProjectData };

export function useInProject(): InProjectState {
  const [state, setState] = useState<InProjectState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const response = await fetch(DATA_URL, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${DATA_URL}`);
        const parsed: unknown = await response.json();
        if (!isInProjectData(parsed)) {
          throw new Error("in_project_metric.json does not match the expected shape (see dashboard/src/lib/inProject.ts).");
        }
        if (!cancelled) setState({ status: "ready", data: parsed });
      } catch (err) {
        if (!cancelled) {
          setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
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

export function pct(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export function fixed4(value: number): string {
  return value.toFixed(4);
}
