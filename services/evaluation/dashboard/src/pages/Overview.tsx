import { EmptyResultsState, ErrorState, LoadingState } from "../components/DataState";
import { MetadataTable } from "../components/MetadataTable";
import { ProvenanceBadge } from "../components/ProvenanceBadge";
import { StatTile } from "../components/StatTile";
import { formatOrNA, formatPercent, isAllReal, useResults } from "../lib/loadResults";
import type { ResultRecord } from "../lib/types";

function latestPassAtK(record: ResultRecord, k: number): number | undefined {
  return record.eval_result.pass_at_k[String(k)];
}

export function Overview() {
  const state = useResults();

  if (state.status === "loading") return <LoadingState />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  const { results } = state.data;
  const allReal = isAllReal(results);

  return (
    <div className="page">
      <div className="page-header">
        <span className="eyebrow">CLASP · P5 — Eval &amp; Data</span>
        <h1>Evaluation Overview</h1>
        <p className="subtitle">
          CLASP evaluates the federated-LoRA merged model produced by the Edge and Cluster layers on two
          published code-generation benchmarks, HumanEval and MBPP, scored with the unbiased Pass@k
          estimator. Results are written to <code>results.json</code> and read by this dashboard.
        </p>
      </div>

      {!allReal && (
        <div className="banner banner-demo">
          <ProvenanceBadge provenance="DEMO_TEST" />
          <span>
            Every result below comes from a deterministic mock generator, not a trained model — P1's real
            merged checkpoint is not available in this environment. See the note on each result for why.
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
            <span className="section-title">Results at a glance</span>
            {results.map((record) => (
              <div key={record.eval_result.benchmark} className="panel" style={{ marginBottom: 0 }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 12,
                  }}
                >
                  <h3>{record.eval_result.benchmark}</h3>
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
