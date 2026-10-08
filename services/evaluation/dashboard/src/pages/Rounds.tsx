import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ErrorState, LoadingState } from "../components/DataState";
import { StatTile } from "../components/StatTile";
import { useRounds } from "../lib/rounds";
import { candidateTrend, clusterIds, type ClusterRound, type VersionBlock } from "../lib/roundsFeed";

// Fixed categorical order (same validated slots the Pass@k chart uses). The
// system has two clusters (web, scientific); colour follows the cluster id.
const SERIES = ["var(--series-humaneval)", "var(--series-mbpp)"];

const fx = (v: number | null | undefined, digits = 4) => (v === null || v === undefined ? "N/A" : v.toFixed(digits));
const signed = (v: number | null) => (v === null ? "N/A" : `${v >= 0 ? "+" : ""}${v.toFixed(4)}`);

function versionCell(v: VersionBlock) {
  return (
    <>
      v{v.version} <span style={{ color: "var(--text-muted)" }}>({v.aggregation})</span>
    </>
  );
}

function DecisionBadge({ c }: { c: ClusterRound }) {
  const promote = c.decision.action === "promote";
  return (
    <span className={"badge " + (promote ? "badge-real" : "badge-demo")}>
      <span className="badge-dot" />
      {promote ? "▲ PROMOTE" : "▼ ROLLBACK"} → v{c.decision.active_version_after}
      {c.authoritative ? "" : " · provisional"}
    </span>
  );
}

export function Rounds() {
  const state = useRounds();
  if (state.status === "loading") return <LoadingState />;
  if (state.status === "error") return <ErrorState message={state.message} />;
  if (state.status === "missing") {
    return (
      <div className="page">
        <div className="page-header">
          <span className="eyebrow">Federated Rounds</span>
          <h1>No round has been exported yet</h1>
        </div>
        <div className="empty-state">
          Run a round (<code>python scripts/demo_round.py --round 1</code>), then from the repository root{" "}
          <code>python scripts/export_round_feed.py experiments/w12-integration/results/round1_manifest.json</code>{" "}
          and <code>npm run sync-data</code>.
        </div>
      </div>
    );
  }

  const feed = state.data;
  const latest = feed.rounds[feed.rounds.length - 1];
  const ids = clusterIds(feed);
  const trend = candidateTrend(feed);

  return (
    <div className="page">
      <div className="page-header">
        <span className="eyebrow">Federated Rounds</span>
        <h1>Registry versions, D5 decisions and regression alerts</h1>
        <p className="subtitle">
          Per round and cluster: the baseline and candidate registry versions, their in-project edit similarity
          (the D5 primary metric), the HumanEval guard, and the decision the State Registry actually took. Read
          from <code>round&#123;N&#125;_manifest.json</code> via <code>scripts/export_round_feed.py</code>; nothing is
          recomputed, and an unmeasured value shows as N/A.
        </p>
      </div>

      <div className="tile-row">
        <StatTile label="Rounds" value={String(feed.rounds.length)} />
        <StatTile
          label={`Round ${latest?.round ?? "-"} wall clock`}
          value={latest?.wall_minutes == null ? "N/A" : `${latest.wall_minutes.toFixed(1)} min`}
          hint={latest?.nfr_round_minutes == null ? undefined : `NFR ${latest.nfr_round_minutes} min${latest.nfr_met ? " · met" : " · missed"}`}
        />
        <StatTile
          label="HumanEval guard (latest)"
          value={latest?.guard.available ? `${fx(latest.guard.candidate_pass_at_1)} vs ${fx(latest.guard.baseline_pass_at_1)}` : "N/A"}
          hint={latest?.guard.available ? "candidate vs baseline pass@1" : "unavailable — decisions provisional"}
        />
        <StatTile label="Alerts" value={String(feed.alerts.length)} />
      </div>

      {feed.alerts.length > 0 && (
        <div className="section">
          <span className="section-title">Regression alerts</span>
          <div className="panel">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Round</th>
                  <th>Cluster</th>
                  <th>Alert</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {feed.alerts.map((a, i) => (
                  <tr key={i}>
                    <td>{a.round}</td>
                    <td>{a.cluster_id}</td>
                    <td>⚠ {a.kind.replace(/_/g, " ")}</td>
                    <td>{a.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {trend.length > 1 && (
        <div className="section">
          <span className="section-title">Candidate edit similarity by round</span>
          <div className="panel" style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trend} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid stroke="var(--gridline)" vertical={false} />
                <XAxis dataKey="round" tick={{ fill: "var(--text-muted)", fontSize: 12 }} axisLine={{ stroke: "var(--border-strong)" }} />
                <YAxis domain={[0, 1]} tick={{ fill: "var(--text-muted)", fontSize: 12 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: "var(--surface-2)", border: "1px solid var(--border)", fontSize: 12 }}
                  formatter={(v: number) => v.toFixed(4)}
                  labelFormatter={(r) => `Round ${r}`}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />
                {ids.map((id, i) => (
                  <Line
                    key={id}
                    type="monotone"
                    dataKey={id}
                    name={`cluster-${id}`}
                    stroke={SERIES[i % SERIES.length]}
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {feed.rounds
        .slice()
        .reverse()
        .map((r) => (
          <div className="section" key={r.round}>
            <span className="section-title">
              Round {r.round}
              {r.finished_utc ? ` · ${r.finished_utc}` : ""} · noise band {fx(r.noise_band)}
            </span>
            <div className="panel">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Cluster</th>
                    <th>Baseline</th>
                    <th>Candidate</th>
                    <th className="num">Edit sim (base)</th>
                    <th className="num">Edit sim (cand)</th>
                    <th className="num">Δ</th>
                    <th>Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {r.clusters.map((c) => (
                    <tr key={c.cluster_id} title={c.decision.reason}>
                      <td>{c.registry_name}</td>
                      <td>{versionCell(c.baseline)}</td>
                      <td>{versionCell(c.candidate)}</td>
                      <td className="num">{fx(c.baseline.in_project?.edit_similarity)}</td>
                      <td className="num">{fx(c.candidate.in_project?.edit_similarity)}</td>
                      <td className="num">{signed(c.edit_similarity_delta)}</td>
                      <td>
                        <DecisionBadge c={c} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!r.guard.available && (
                <div className="note" style={{ marginTop: 12 }}>
                  <strong>Guard unavailable.</strong> {r.guard.reason}
                </div>
              )}
            </div>
          </div>
        ))}
    </div>
  );
}
