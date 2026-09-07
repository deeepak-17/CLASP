import { EmptyResultsState, ErrorState, LoadingState } from "../components/DataState";
import { ProvenanceBadge } from "../components/ProvenanceBadge";
import { formatOrNA, formatPercent, useResults } from "../lib/loadResults";

export function Benchmarks() {
  const state = useResults();

  if (state.status === "loading") return <LoadingState />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  const { results } = state.data;

  return (
    <div className="page">
      <div className="page-header">
        <span className="eyebrow">Benchmarks</span>
        <h1>HumanEval &amp; MBPP</h1>
        <p className="subtitle">
          Two published code-generation benchmarks. Both are loaded from their canonical published sources
          (see <code>scripts/fetch_benchmark_data.py</code>) — not hand-written stand-ins — and scored by
          actually executing every generated candidate against the task's own tests.
        </p>
        <div className="note">
          <strong>Pass@k here is the D5 regression guard, not the primary signal.</strong> It is currently
          run against the offline mock generator (so every rate is 0.0 by construction) — full guard
          scoring needs a Linux container for <code>evalplus</code>. The metric the promotion rule actually
          decides on is on the <strong>In-Project Metric</strong> page.
        </div>
      </div>

      {results.length === 0 ? (
        <EmptyResultsState />
      ) : (
        <div className="section">
          {results.map((record) => {
            const ks = Object.keys(record.eval_result.pass_at_k)
              .map(Number)
              .sort((a, b) => a - b);
            return (
              <div key={record.eval_result.benchmark} className="panel">
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    marginBottom: 12,
                  }}
                >
                  <div>
                    <h3>{record.eval_result.benchmark}</h3>
                    <p style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 4 }}>
                      {record.run_metadata.dataset_split}
                    </p>
                  </div>
                  <ProvenanceBadge provenance={record.run_metadata.provenance} />
                </div>

                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Tasks evaluated</th>
                      <th>Samples / task</th>
                      {ks.map((k) => (
                        <th key={k} className="num">
                          Pass@{k}
                        </th>
                      ))}
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="num">{formatOrNA(record.eval_result.num_tasks)}</td>
                      <td className="num">{formatOrNA(record.eval_result.num_samples_per_task)}</td>
                      {ks.map((k) => (
                        <td key={k} className="num">
                          {formatPercent(record.eval_result.pass_at_k[String(k)])}
                        </td>
                      ))}
                      <td>Scored (executed, not simulated)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            );
          })}

          <div className="note">
            <strong>Pass@k above reflects the model run recorded in results.json only.</strong> A separate
            repository check (<code>scripts/sanity_check_scoring.py</code>) verifies the scoring/execution
            pipeline itself by running each task's own canonical reference solution through the identical
            execute-and-score path. That is a pipeline self-test proving the scorer works — it is{" "}
            <em>not</em> a model-performance result, is not computed from any model's output, and is not
            shown as a Pass@k figure here.
          </div>
        </div>
      )}
    </div>
  );
}
