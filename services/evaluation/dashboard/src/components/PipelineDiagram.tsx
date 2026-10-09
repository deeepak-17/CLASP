interface Step {
  title: string;
  desc: string;
  ref: string;
}

// Terminology and module references match the repository exactly — nothing
// here is invented. Cross-check: src/evaluation/eval_harness/harness.py, src/evaluation/eval_harness/base.py,
// src/evaluation/interfaces/edge_client.py, src/evaluation/interfaces/edge_transformers_adapter.py,
// src/evaluation/eval_harness/execution.py, src/evaluation/eval_harness/scoring.py, src/evaluation/eval_harness/results_store.py.
const STEPS: Step[] = [
  {
    title: "Dataset",
    desc: "Real HumanEval (164) / MBPP (392) task sets, fetched from their published sources",
    ref: "scripts/fetch_benchmark_data.py",
  },
  {
    title: "Prompt",
    desc: "Each task's function signature/docstring (HumanEval) or description (MBPP) becomes the model prompt",
    ref: "src/evaluation/eval_harness/{humaneval,mbpp}/adapter.py",
  },
  {
    title: "Merged Model",
    desc: "P1's Edge Layer generates a completion for each prompt (mock generator in this repository; real adapter implemented, unused pending weights)",
    ref: "src/evaluation/interfaces/edge_client.py, edge_transformers_adapter.py",
  },
  {
    title: "Generated Candidates",
    desc: "num_samples_per_task completions per task, truncated at stop sequences",
    ref: "src/evaluation/eval_harness/harness.py",
  },
  {
    title: "Code Execution",
    desc: "Each candidate is assembled with the task's tests and run in a subprocess sandbox",
    ref: "src/evaluation/eval_harness/execution.py",
  },
  {
    title: "Pass / Fail",
    desc: "Exit code 0 = pass; recorded per completion, tri-state (not-scored vs failed are distinct)",
    ref: "src/evaluation/eval_harness/models.py (TaskOutcome)",
  },
  {
    title: "Pass@k",
    desc: "Unbiased estimator over (n, c, k) per task, aggregated per benchmark",
    ref: "src/evaluation/eval_harness/scoring.py",
  },
  {
    title: "results.json",
    desc: "Schema-valid EvalResult + run metadata (model, dataset, seed, REAL/DEMO provenance)",
    ref: "src/evaluation/eval_harness/results_store.py",
  },
  {
    title: "Dashboard",
    desc: "This application — reads results.json as a static file, renders it",
    ref: "dashboard/ (this app)",
  },
];

/** Static, left-to-right pipeline explainer for the Week-5 panel review.
 * Deliberately not interactive/animated — a reader should be able to follow
 * it at a glance on a projector. */
export function PipelineDiagram() {
  return (
    <div className="pipeline">
      {STEPS.map((step, i) => (
        <div key={step.title} style={{ display: "contents" }}>
          <div className="pipeline-step">
            <span className="step-index">{String(i + 1).padStart(2, "0")}</span>
            <span className="step-title">{step.title}</span>
            <span className="step-desc">{step.desc}</span>
            <span className="step-ref mono">{step.ref}</span>
          </div>
          {i < STEPS.length - 1 && (
            <div className="pipeline-arrow" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path
                  d="M2 8h11m0 0-4-4m4 4-4 4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
