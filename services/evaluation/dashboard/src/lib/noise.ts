// Shape of results/noise_report.json — written by scripts/build_noise_report.py
// (src/evaluation/eval_harness/noise.py).

export interface PairedRow {
  discordance: number;
  min_detectable_drop_at_n: number;
  min_detectable_drop_full_benchmark: number;
  items_needed_for_tolerance: number;
}

export interface NoiseReport {
  generated_at: string;
  sources: Record<string, string>;
  humaneval_guard: {
    n_tasks: number;
    pass_at_1: number;
    standard_error: number;
    bootstrap_ci_95: [number, number];
    tolerance: number;
    full_benchmark_size: number;
    unpaired_min_detectable_drop_at_n: number;
    paired: PairedRow[];
  };
  locked_thresholds: {
    tolerance_pass_at_1: number;
    assumed_discordance: number;
    noise_floor_at_scored_n: number;
    noise_floor_full_benchmark: number;
    statements: string[];
  };
  in_project: {
    n_examples: number;
    /** D5's band: spread of 3 repeated baseline evaluations. */
    d5_noise_band: { definition: string; value: number; degenerate: boolean };
    exact_match_paired_noise_floor: number;
    edit_similarity_band: number | null;
    edit_similarity_band_status: string;
    rows: { client: string; version: string; edit_similarity: number; exact_match: number; exact_match_standard_error: number }[];
  };
  candidate?: {
    n_shared_tasks: number;
    measured_discordance: number;
    pass_at_1_drop: number;
    noise_floor: number;
    verdict: "pass" | "within_noise" | "regression";
  };
}

export function isNoiseReport(x: unknown): x is NoiseReport {
  if (typeof x !== "object" || x === null) return false;
  const r = x as Record<string, unknown>;
  const g = r.humaneval_guard as Record<string, unknown> | undefined;
  const t = r.locked_thresholds as Record<string, unknown> | undefined;
  const ip = r.in_project as Record<string, unknown> | undefined;
  return (
    !!g &&
    typeof g.pass_at_1 === "number" &&
    Array.isArray(g.bootstrap_ci_95) &&
    Array.isArray(g.paired) &&
    !!t &&
    Array.isArray(t.statements) &&
    !!ip &&
    Array.isArray(ip.rows) &&
    typeof (ip.d5_noise_band as Record<string, unknown> | undefined)?.value === "number"
  );
}
