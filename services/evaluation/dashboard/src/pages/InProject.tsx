import { ErrorState, LoadingState } from "../components/DataState";
import { fixed4, pct, useInProject } from "../lib/inProject";

export function InProject() {
  const state = useInProject();

  if (state.status === "loading") return <LoadingState />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  const d = state.data;
  const completion = d.in_project_completion.rows;
  const best = completion.reduce((a, b) => (b.edit_similarity > a.edit_similarity ? b : a));
  const bold = (isBest: boolean, text: string) => (
    <span style={{ fontWeight: isBest ? 700 : 400 }}>{text}</span>
  );

  return (
    <div className="page">
      <div className="page-header">
        <span className="eyebrow">In-Project Metric</span>
        <h1>Completion quality on held-out client code</h1>
        <p className="subtitle">
          The <strong>primary</strong> signal the D5 promotion rule decides on. For each client a deterministic
          slice of its files is held out and never trained on; next-line completion is run over{" "}
          {d.held_out_examples} held-out examples ({d.decoding} decoding) and scored as{" "}
          <strong>edit similarity</strong> (character-level 1 &minus; lev / max&nbsp;len, the CodeXGLUE
          convention) and <strong>exact match</strong>. HumanEval/MBPP Pass@k is the separate regression
          guard.
        </p>
      </div>

      <div className="section">
        <span className="section-title">Measurement</span>
        <div className="panel">
          <table className="data-table">
            <tbody>
              <tr>
                <td>Environment</td>
                <td>{d.environment}</td>
              </tr>
              <tr>
                <td>Round</td>
                <td>{d.round_shape}</td>
              </tr>
              <tr>
                <td>Held-out examples</td>
                <td>
                  {d.held_out_examples} per client · {d.decoding} decoding
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="section">
        <span className="section-title">In-project completion — aggregation ablation</span>
        <div className="panel">
          <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 12 }}>
            {d.in_project_completion.caption}
          </p>
          <table className="data-table">
            <thead>
              <tr>
                <th>Client</th>
                <th>Cluster</th>
                <th>Aggregate</th>
                <th className="num">Edit similarity</th>
                <th className="num">Exact match</th>
              </tr>
            </thead>
            <tbody>
              {completion.map((row) => (
                <tr key={`${row.client}-${row.version}`}>
                  <td>{row.client}</td>
                  <td>{row.cluster}</td>
                  <td>{row.version}</td>
                  <td className="num">{bold(row === best, fixed4(row.edit_similarity))}</td>
                  <td className="num">{bold(row === best, fixed4(row.exact_match))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="section">
        <span className="section-title">Why the SVD aggregate wins — norm error</span>
        <div className="panel">
          <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 12 }}>
            {d.aggregation_ablation.caption}
          </p>
          <table className="data-table">
            <thead>
              <tr>
                <th>Cluster</th>
                <th className="num">Naive average</th>
                <th className="num">Exact SVD</th>
                <th className="num">Improvement</th>
              </tr>
            </thead>
            <tbody>
              {d.aggregation_ablation.rows.map((row) => (
                <tr key={row.cluster}>
                  <td>{row.cluster}</td>
                  <td className="num">{fixed4(row.naive_avg)}</td>
                  <td className="num">
                    <strong>{fixed4(row.svd)}</strong>
                  </td>
                  <td className="num">{row.ratio}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="note">
        <strong>Takeaway.</strong> {d.takeaway} Best edit similarity here: {pct(best.edit_similarity)} (
        {best.client}, {best.version}).
      </div>

      <div className="note">{d.note}</div>
    </div>
  );
}
