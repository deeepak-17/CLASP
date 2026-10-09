import { NavLink } from "react-router-dom";
import { EmptyResultsState, ErrorState, LoadingState } from "../components/DataState";
import { MetadataTable } from "../components/MetadataTable";
import { ProvenanceBadge } from "../components/ProvenanceBadge";
import { StatTile } from "../components/StatTile";
import { formatOrNA, formatPercent, useResults } from "../lib/loadResults";
import { recordKey, recordLabel } from "../lib/records";
import { isNoiseReport } from "../lib/noise";
import { isPersonalizationFeed } from "../lib/personalization";
import type { ResultRecord } from "../lib/types";
import { useFeed } from "../lib/useFeed";

function latestPassAtK(record: ResultRecord, k: number): number | undefined {
  return record.eval_result.pass_at_k[String(k)];
}

const signed = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(3)}`;

/** D5 headline: the personalization delta (primary signal) shown next to the
 * HumanEval guard — never Pass@k on its own. Each half renders N/A, not a
 * made-up number, when its feed has not been produced. */
function Headline() {
  const pers = useFeed("personalization.json", isPersonalizationFeed, "dashboard/src/lib/personalization.ts");
  const noise = useFeed("noise_report.json", isNoiseReport, "dashboard/src/lib/noise.ts");
  const round = pers.status === "ready" ? pers.data.rounds[pers.data.rounds.length - 1] : null;
  const guard = noise.status === "ready" ? noise.data : null;
  return (
    <div className="section">
      <span className="section-title">Headline — personalization (D5 primary) with the HumanEval guard</span>
      <div className="tile-row">
        <StatTile
          label="Clients improved on their own code"
          value={round ? `${round.summary.n_improved} / ${round.summary.n_clients}` : "N/A"}
          hint={round ? `round ${round.round}, held-out files never trained on` : "personalization.json not produced"}
        />
        <StatTile
          label="Mean personalization Δ perplexity"
          value={round ? signed(round.summary.mean_delta_ppl) : "N/A"}
          hint="composite − base, lower is better"
        />
        <StatTile
          label="Guard: base HumanEval pass@1"
          value={guard ? `${(guard.humaneval_guard.pass_at_1 * 100).toFixed(0)}%` : "N/A"}
          hint={guard ? `${guard.humaneval_guard.n_tasks} tasks · candidate not yet scored` : "noise_report.json not produced"}
        />
        <StatTile
          label="Guard tolerance vs noise floor"
          value={
            guard
              ? `${(guard.humaneval_guard.tolerance * 100).toFixed(0)} vs ${(guard.locked_thresholds.noise_floor_at_scored_n * 100).toFixed(1)} pts`
              : "N/A"
          }
          hint="a drop inside the floor is reported as within noise"
        />
      </div>
      <p style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 8 }}>
        Details: <NavLink to="/personalization">Personalization</NavLink> ·{" "}
        <NavLink to="/noise">Noise &amp; Guard</NavLink> · <NavLink to="/in-project">In-Project Metric</NavLink>
      </p>
    </div>
  );
}

export function Overview() {
  const state = useResults();

  if (state.status === "loading") return <LoadingState />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  const { results } = state.data;
  const nDemo = results.filter((r) => r.run_metadata.provenance !== "REAL").length;

  return (
    <div className="page">
      <div className="page-header">
        <span className="eyebrow">CLASP · P5 — Eval &amp; Data</span>
        <h1>Evaluation Overview</h1>
        <p className="subtitle">
          CLASP's primary measurement is personalization: does each client's composite model (base + α·cluster +
          β·client) complete and predict its <em>own</em> held-out code better than the base? HumanEval and MBPP
          Pass@k are the regression guard — they check that general coding ability did not drop. All numbers are
          read from files the Python pipeline writes; the dashboard computes nothing.
        </p>
      </div>

      <Headline />

      {nDemo > 0 && (
        <div className="banner banner-demo">
          <ProvenanceBadge provenance="DEMO_TEST" />
          <span>
            {nDemo === results.length
              ? "Every benchmark record below comes from a deterministic mock generator, not a trained model."
              : `${nDemo} of ${results.length} benchmark records below come from a deterministic mock generator; the REAL record is the base model's HumanEval subset.`}{" "}
            See the note on each record.
          </span>
        </div>
      )}

      {results.length === 0 ? (
        <EmptyResultsState />
      ) : (
        <>
          <div className="section">
            <span className="section-title">Where P5 fits in CLASP</span>
            <div className="panel">
              <ClaspTree />
            </div>
          </div>

          <div className="section">
            <span className="section-title">Regression guard — benchmark records</span>
            {results.map((record) => (
              <div key={recordKey(record)} className="panel" style={{ marginBottom: 0 }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 12,
                  }}
                >
                  <h3>{recordLabel(record)}</h3>
                  <ProvenanceBadge provenance={record.run_metadata.provenance} />
                </div>
                <div className="tile-row">
                  <StatTile label="Pass@1" value={formatPercent(latestPassAtK(record, 1))} />
                  <StatTile
                    label="Pass@10"
                    value={formatPercent(latestPassAtK(record, 10))}
                    hint={
                      latestPassAtK(record, 10) === undefined
                        ? "not computed for this run"
                        : undefined
                    }
                  />
                  <StatTile label="Tasks" value={formatOrNA(record.eval_result.num_tasks)} />
                  <StatTile
                    label="Samples / task"
                    value={formatOrNA(record.eval_result.num_samples_per_task)}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="section">
            <span className="section-title">Run metadata</span>
            <div className="panel">
              <MetadataTable records={results} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/** Module ownership exactly as documented in README.md's Team table and the
 * CLASP Daily Targets swimlanes — no internal detail about P1-P4 beyond
 * their already-published module names is asserted here. */
function ClaspTree() {
  const rows: Array<{ label: string; sub?: string; active?: boolean }> = [
    { label: "P1 — Edge Layer", sub: "Integration & Inference" },
    { label: "P2 — Cluster Layer", sub: "Documentation & Paper" },
    { label: "P3 — Security", sub: "Security & QA" },
    { label: "P4 — State Registry", sub: "DevOps & Orchestration" },
    { label: "P5 — Eval & Data", sub: "Data & Demo (this dashboard)", active: true },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div className="mono" style={{ fontSize: 12, color: "var(--text-muted)" }}>
        CLASP
      </div>
      {rows.map((row) => (
        <div
          key={row.label}
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 10,
            paddingLeft: 18,
            borderLeft: row.active ? "2px solid var(--accent)" : "2px solid var(--gridline)",
          }}
        >
          <span style={{ fontSize: 13, fontWeight: row.active ? 600 : 500 }}>{row.label}</span>
          <span style={{ fontSize: 11.5, color: "var(--text-muted)" }}>{row.sub}</span>
        </div>
      ))}
    </div>
  );
}
