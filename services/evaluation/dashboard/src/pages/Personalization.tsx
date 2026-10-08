import { AbsentFeedState, ErrorState, LoadingState } from "../components/DataState";
import { RoundBars } from "../components/RoundBars";
import { StatTile } from "../components/StatTile";
import { contributionByClient, isPersonalizationFeed, shortClient } from "../lib/personalization";
import type { PersonalizationFeed } from "../lib/personalization";
import { useFeed } from "../lib/useFeed";

const signed = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(3)}`;

function reductionByClient(feed: PersonalizationFeed): Record<string, number | string>[] {
  const rows = new Map<string, Record<string, number | string>>();
  for (const round of feed.rounds) {
    for (const c of round.clients) {
      const row = rows.get(c.client_id) ?? { client: shortClient(c.client_id) };
      // Positive = the composite lowers perplexity vs the frozen base.
      row[`r${round.round}`] = Number((c.base_ppl - c.composite_ppl).toFixed(4));
      rows.set(c.client_id, row);
    }
  }
  return [...rows.values()];
}

export function Personalization() {
  const state = useFeed("personalization.json", isPersonalizationFeed, "dashboard/src/lib/personalization.ts");
  if (state.status === "loading") return <LoadingState />;
  if (state.status === "error") return <ErrorState message={state.message} />;
  if (state.status === "absent")
    return <AbsentFeedState file="results/personalization.json" command="python scripts/build_personalization_report.py" />;

  const feed = state.data;
  const rounds = feed.rounds.map((r) => r.round);
  const last = feed.rounds[feed.rounds.length - 1];
  const first = feed.rounds[0];
  const comparison = feed.comparisons[feed.comparisons.length - 1];

  return (
    <div className="page">
      <div className="page-header">
        <span className="eyebrow">Personalization</span>
        <h1>Does each client's model get better on its own code?</h1>
        <p className="subtitle">
          Held-out perplexity on each client's never-trained-on files, for the frozen base, the client-only adapter
          and the three-layer composite base + α·cluster + β·client. Lower is better. Every number comes from the
          edge lane's round manifests.
        </p>
      </div>

      <div className="tile-row">
        <StatTile
          label={`Clients improved, round ${last.round}`}
          value={`${last.summary.n_improved} / ${last.summary.n_clients}`}
          hint="composite below base perplexity"
        />
        <StatTile
          label={`Mean Δ perplexity, round ${last.round}`}
          value={signed(last.summary.mean_delta_ppl)}
          hint={`round ${first.round}: ${signed(first.summary.mean_delta_ppl)}`}
        />
        <StatTile
          label="Cluster layer helps"
          value={`${last.summary.n_cluster_helps} / ${last.summary.n_clients}`}
          hint={`round ${first.round}: ${first.summary.n_cluster_helps} / ${first.summary.n_clients}`}
        />
        {comparison ? (
          <StatTile
            label={`Composite better r${comparison.from_round}→r${comparison.to_round}`}
            value={`${comparison.n_composite_better} / ${comparison.n_clients}`}
          />
        ) : null}
      </div>

      <div className="section">
        <span className="section-title">Perplexity reduction vs the frozen base (higher is better)</span>
        <div className="panel">
          <RoundBars data={reductionByClient(feed)} rounds={rounds} format={(v) => v.toFixed(3)} />
        </div>
      </div>

      <div className="section">
        <span className="section-title">Cluster layer's effect at α = 0.5 (below zero = it helps)</span>
        <div className="panel">
          <RoundBars data={contributionByClient(feed)} rounds={rounds} format={signed} zeroLine />
          <p style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 10 }}>
            Composite minus client-only perplexity. In round 1 clients were trained on the bare base and the cluster
            layer made every client slightly worse; in round 2 they were trained on the frozen base + 0.5·cluster (the
            D3 order) and it helps every client.
          </p>
        </div>
      </div>

      {feed.rounds.map((round) => (
        <div className="section" key={round.round}>
          <span className="section-title">{round.label}</span>
          <div className="panel">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Client</th>
                  <th className="num">Held-out tokens</th>
                  <th className="num">Base</th>
                  <th className="num">Client only</th>
                  <th className="num">Composite</th>
                  <th className="num">Δ vs base</th>
                  <th className="num">Cluster @ α_ref</th>
                  <th className="num">Best α</th>
                </tr>
              </thead>
              <tbody>
                {round.clients.map((c) => (
                  <tr key={c.client_id}>
                    <td>{c.client_id}</td>
                    <td className="num">{c.n_tokens.toLocaleString()}</td>
                    <td className="num">{c.base_ppl.toFixed(3)}</td>
                    <td className="num">{c.client_only_ppl.toFixed(3)}</td>
                    <td className="num">{c.composite_ppl.toFixed(3)}</td>
                    <td className="num">{signed(c.personalization_delta_ppl)}</td>
                    <td className="num">{signed(c.cluster_contribution_at_alpha_ref)}</td>
                    <td className="num">{c.best_alpha}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      <div className="note">
        1.3B dev profile on a 4 GB GPU, seed 0. client-requests' held-out split is a single 513-token block, so its
        row is the noisiest. Decisions are provisional until the HumanEval guard is scored on a candidate — see the
        Noise &amp; Guard page. Sources: {feed.sources.join(", ")}.
      </div>
    </div>
  );
}
