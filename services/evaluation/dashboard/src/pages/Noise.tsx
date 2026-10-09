import { AbsentFeedState, ErrorState, LoadingState } from "../components/DataState";
import { StatTile } from "../components/StatTile";
import { isNoiseReport } from "../lib/noise";
import { useFeed } from "../lib/useFeed";

const pts = (v: number) => `${(v * 100).toFixed(1)} pts`;

const VERDICT = {
  pass: "Pass — drop within tolerance",
  within_noise: "Within noise — drop exceeds tolerance but not the noise floor",
  regression: "Regression — drop beyond tolerance and noise floor",
};

export function Noise() {
  const state = useFeed("noise_report.json", isNoiseReport, "dashboard/src/lib/noise.ts");
  if (state.status === "loading") return <LoadingState />;
  if (state.status === "error") return <ErrorState message={state.message} />;
  if (state.status === "absent")
    return <AbsentFeedState file="results/noise_report.json" command="python scripts/build_noise_report.py" />;

  const r = state.data;
  const g = r.humaneval_guard;
  const t = r.locked_thresholds;

  return (
    <div className="page">
      <div className="page-header">
        <span className="eyebrow">Noise &amp; Guard</span>
        <h1>How big a change has to be before it counts</h1>
        <p className="subtitle">
          D5 promotes when the in-project gain clears a noise band and HumanEval pass@1 does not drop by more than{" "}
          {pts(g.tolerance)}. Both thresholds only mean something against the measurement noise of the metric they
          gate; this page measures it from the scored base-model anchor.
        </p>
      </div>

      <div className="tile-row">
        <StatTile label="Base pass@1" value={`${(g.pass_at_1 * 100).toFixed(0)}%`} hint={`${g.n_tasks} HumanEval tasks`} />
        <StatTile
          label="95% interval"
          value={`${(g.bootstrap_ci_95[0] * 100).toFixed(0)}–${(g.bootstrap_ci_95[1] * 100).toFixed(0)}%`}
          hint="bootstrap over tasks"
        />
        <StatTile label="D5 tolerance" value={pts(g.tolerance)} hint="max allowed pass@1 drop" />
        <StatTile
          label="Noise floor now"
          value={pts(t.noise_floor_at_scored_n)}
          hint={`paired, ${g.n_tasks} tasks, discordance ${t.assumed_discordance}`}
        />
        <StatTile
          label="Noise floor, full HumanEval"
          value={pts(t.noise_floor_full_benchmark)}
          hint={`${g.full_benchmark_size} tasks`}
        />
      </div>

      {r.candidate ? (
        <div className="note">
          <strong>Candidate:</strong> {VERDICT[r.candidate.verdict]} (drop {pts(r.candidate.pass_at_1_drop)}, floor{" "}
          {pts(r.candidate.noise_floor)}, measured discordance {r.candidate.measured_discordance}).
        </div>
      ) : (
        <div className="note">
          No candidate anchor scored yet — the guard verdict appears here once{" "}
          <code>build_noise_report.py --candidate-anchor</code> is run on the composite model's samples.
        </div>
      )}

      <div className="section">
        <span className="section-title">Smallest pass@1 drop distinguishable from noise</span>
        <div className="panel">
          <table className="data-table">
            <thead>
              <tr>
                <th className="num">Discordance</th>
                <th className="num">At {g.n_tasks} tasks</th>
                <th className="num">At {g.full_benchmark_size} tasks</th>
                <th className="num">Tasks needed for {pts(g.tolerance)}</th>
              </tr>
            </thead>
            <tbody>
              {g.paired.map((row) => (
                <tr key={row.discordance}>
                  <td className="num">{row.discordance}</td>
                  <td className="num">{pts(row.min_detectable_drop_at_n)}</td>
                  <td className="num">{pts(row.min_detectable_drop_full_benchmark)}</td>
                  <td className="num">{row.items_needed_for_tolerance.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 10 }}>
            Discordance is the share of tasks whose pass/fail flips between baseline and candidate; only those tasks
            carry noise in a paired comparison. It is measured once a candidate is scored.
          </p>
        </div>
      </div>

      <div className="section">
        <span className="section-title">Locked thresholds</span>
        <div className="panel">
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: 1.6 }}>
            {t.statements.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="section">
        <span className="section-title">In-project completion noise ({r.in_project.n_examples} examples)</span>
        <div className="panel">
          <table className="data-table">
            <thead>
              <tr>
                <th>Client</th>
                <th>Version</th>
                <th className="num">Edit similarity</th>
                <th className="num">Exact match</th>
                <th className="num">Exact-match SE</th>
              </tr>
            </thead>
            <tbody>
              {r.in_project.rows.map((row) => (
                <tr key={`${row.client}-${row.version}`}>
                  <td>{row.client}</td>
                  <td>{row.version}</td>
                  <td className="num">{row.edit_similarity.toFixed(4)}</td>
                  <td className="num">{row.exact_match.toFixed(4)}</td>
                  <td className="num">{row.exact_match_standard_error.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ fontSize: 13, marginTop: 10 }}>
            <strong>D5 noise band</strong> ({r.in_project.d5_noise_band.definition}):{" "}
            <strong>{r.in_project.d5_noise_band.value.toFixed(4)}</strong>.{" "}
            {r.in_project.d5_noise_band.degenerate
              ? "Greedy decoding is deterministic, so the three repeats are identical and the band is zero — any positive gain counts as an improvement. It reflects decode noise only, not variation between independently trained adapters."
              : null}
          </p>
          <p style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 10 }}>
            An exact-match change below {r.in_project.exact_match_paired_noise_floor.toFixed(3)} is inside noise.
            Supplementary edit-similarity band: {r.in_project.edit_similarity_band_status}.
          </p>
        </div>
      </div>
    </div>
  );
}
