import { PipelineDiagram } from "../components/PipelineDiagram";

export function Pipeline() {
  return (
    <div className="page">
      <div className="page-header">
        <span className="eyebrow">Evaluation Pipeline</span>
        <h1>How a Pass@k number is produced</h1>
        <p className="subtitle">
          Each stage below is a real module in this repository, not a conceptual simplification — hover
          terms map 1:1 onto <code>evaluation/</code> and <code>interfaces/</code>.
        </p>
      </div>

      <div className="section">
        <div className="panel">
          <PipelineDiagram />
        </div>
      </div>

      <div className="section">
        <span className="section-title">Notes for panel presentation</span>
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <p style={{ fontSize: 13 }}>
            <strong>Merged Model</strong> is P1's Edge Layer output — the composition of the base model
            with cluster and client LoRA adapters. This repository's harness calls it through one interface
            (<code>interfaces/edge_client.py::EdgeInferenceClient</code>) so the same evaluation code runs
            unchanged whether the backend is the deterministic mock used today or a real checkpoint.
          </p>
          <p style={{ fontSize: 13 }}>
            <strong>Code Execution</strong> is a real subprocess run of the assembled program (prompt +
            completion + the task's own tests), not a static check — see{" "}
            <code>evaluation/execution.py</code>.
          </p>
          <p style={{ fontSize: 13 }}>
            <strong>results.json</strong> keeps two files apart: one raw, per-run artefact holding every
            completion, and this dashboard-facing index holding scored summaries plus REAL/DEMO provenance
            — see <code>evaluation/results_store.py</code>.
          </p>
        </div>
      </div>
    </div>
  );
}
