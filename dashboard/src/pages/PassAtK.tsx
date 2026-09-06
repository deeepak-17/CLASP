import { EmptyResultsState, ErrorState, LoadingState } from "../components/DataState";
import { PassAtKChart } from "../components/PassAtKChart";
import { ProvenanceBadge } from "../components/ProvenanceBadge";
import { useResults } from "../lib/loadResults";

export function PassAtK() {
  const state = useResults();

  if (state.status === "loading") return <LoadingState />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  const { results } = state.data;

  return (
    <div className="page">
      <div className="page-header">
        <span className="eyebrow">Pass@k</span>
        <h1>Pass@k by benchmark</h1>
        <p className="subtitle">
          Pass@k = 1 − C(n−c, k) / C(n, k) — the unbiased estimator (Chen et al., 2021) of the probability
          that at least one of k sampled completions passes, given n generated samples of which c passed.
          See <code>evaluation/scoring.py</code>.
        </p>
      </div>

      {results.length === 0 ? (
        <EmptyResultsState />
      ) : (
        <>
          <div className="section">
            <span className="section-title">Pass@k comparison</span>
            <div className="panel">
              <PassAtKChart records={results} />
              <div style={{ display: "flex", gap: 16, marginTop: 4, flexWrap: "wrap" }}>
                {results.map((r) => (
                  <div
                    key={r.eval_result.benchmark}
                    style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}
                  >
                    <ProvenanceBadge provenance={r.run_metadata.provenance} />
                    <span style={{ color: "var(--text-muted)" }}>{r.eval_result.benchmark}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="note">
            No trend is implied between k values — each bar is an independent estimate computed directly
            from the samples recorded for that run (<code>num_samples_per_task</code> ={" "}
            {results[0]?.eval_result.num_samples_per_task ?? "N/A"}). A k with no samples for a given
            benchmark is simply absent from that group, not shown as zero.
          </div>
        </>
      )}
    </div>
  );
}
