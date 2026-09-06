// Mirrors the Python data model exactly. Keep this file in sync with:
//   interfaces/contracts.py        (AdapterRef, EvalResult)
//   evaluation/results_store.py    (GenerationSnapshot, ResultRecord, ResultsIndex)
//
// No field here is invented — every property matches a field the Python
// side actually serialises to evaluation/results/results.json.

export interface AdapterRef {
  name: string;
  version: number;
  kind: "client" | "cluster";
  cluster_id: string | null;
}

/** benchmark -> pass@k rate, e.g. {"1": 0.0, "10": 0.0}. Keys are strings
 * because JSON object keys are strings (interfaces/contracts.py::EvalResult.to_dict). */
export type PassAtKMap = Record<string, number>;

export interface EvalResult {
  adapter: AdapterRef;
  benchmark: "HumanEval" | "MBPP" | string;
  pass_at_k: PassAtKMap;
  num_tasks: number;
  num_samples_per_task: number;
  created_at: string | null;
  run_id: string | null;
  contract_version: string;
}

export interface GenerationSnapshot {
  max_new_tokens: number;
  temperature: number;
  stop_sequences: string[];
}

/** "REAL" | "DEMO_TEST" — see evaluation/results_store.py::Provenance. */
export type Provenance = "REAL" | "DEMO_TEST";

export interface RunMetadata {
  model_checkpoint: string;
  dataset_split: string;
  num_problems: number;
  seed: number | null;
  generation: GenerationSnapshot;
  provenance: Provenance;
  provenance_note: string;
  raw_artifact_path: string | null;
  benchmark_data_source: string | null;
}

export interface ResultRecord {
  eval_result: EvalResult;
  run_metadata: RunMetadata;
}

export interface ResultsIndex {
  index_version: string;
  updated_at: string;
  contract_version: string;
  results: ResultRecord[];
}
