import { useState } from "react";
import { AbsentFeedState, ErrorState, LoadingState } from "../components/DataState";
import { isLineageFeed, lineageColumns, parentsOf } from "../lib/lineage";
import type { LineageNode } from "../lib/lineage";
import { useFeed } from "../lib/useFeed";

const EDGE_LABEL = { aggregated_into: "aggregated from", trained_on: "trained on", composed_into: "composed from" };

function NodeCard({ node, selected, onSelect }: { node: LineageNode; selected: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className="panel"
      style={{
        display: "block",
        width: "100%",
        textAlign: "left",
        padding: "8px 10px",
        marginBottom: 6,
        cursor: "pointer",
        borderColor: selected ? "var(--accent)" : undefined,
        background: selected ? "var(--accent-soft)" : undefined,
        color: "var(--text-primary)",
        font: "inherit",
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 600 }}>{node.label}</div>
      <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>
        {node.cluster}
        {node.aggregation ? ` · ${node.aggregation}` : ""}
        {node.trained_on ? ` · on ${node.trained_on}` : ""}
        {node.composite_ppl != null ? ` · ppl ${node.composite_ppl.toFixed(3)}` : ""}
      </div>
    </button>
  );
}

export function Lineage() {
  const state = useFeed("lineage.json", isLineageFeed, "dashboard/src/lib/lineage.ts");
  const [selected, setSelected] = useState<string | null>(null);
  if (state.status === "loading") return <LoadingState />;
  if (state.status === "error") return <ErrorState message={state.message} />;
  if (state.status === "absent")
    return <AbsentFeedState file="results/lineage.json" command="python scripts/export_lineage.py" />;

  const feed = state.data;
  const columns = lineageColumns(feed);
  const current = feed.nodes.find((n) => n.id === selected) ?? null;
  const parents = current ? parentsOf(feed, current.id) : [];

  return (
    <div className="page">
      <div className="page-header">
        <span className="eyebrow">Adapter Lineage</span>
        <h1>Which adapter was built from which</h1>
        <p className="subtitle">
          Client adapters are aggregated into cluster adapters; in the D3 order the next round's clients train on the
          frozen cluster; each round evaluates per-client composites. Registry versions carry their sha256. Select an
          adapter to see what it came from.
        </p>
      </div>

      <div className="section">
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${columns.length}, minmax(150px, 1fr))`, gap: 12, overflowX: "auto" }}>
          {columns.map((col) => (
            <div key={col.title}>
              <span className="section-title">{col.title}</span>
              {col.nodes.map((n) => (
                <NodeCard key={n.id} node={n} selected={n.id === selected} onSelect={() => setSelected(n.id)} />
              ))}
            </div>
          ))}
        </div>
      </div>

      <div className="section">
        <span className="section-title">Provenance</span>
        <div className="panel">
          {current ? (
            <table className="data-table">
              <tbody>
                <tr>
                  <td>Adapter</td>
                  <td>
                    <code>{current.id}</code>
                  </td>
                </tr>
                {current.sha256 ? (
                  <tr>
                    <td>sha256</td>
                    <td>
                      <code>{current.sha256}</code>
                    </td>
                  </tr>
                ) : null}
                {current.rel_err_vs_exact != null ? (
                  <tr>
                    <td>Rel. error vs exact average</td>
                    <td>{current.rel_err_vs_exact.toFixed(4)}</td>
                  </tr>
                ) : null}
                {parents.length ? (
                  parents.map((p) => (
                    <tr key={p.node.id}>
                      <td>{EDGE_LABEL[p.kind]}</td>
                      <td>
                        <button type="button" className="link-button" onClick={() => setSelected(p.node.id)}>
                          {p.node.label}
                        </button>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td>Built from</td>
                    <td>the frozen base model only</td>
                  </tr>
                )}
              </tbody>
            </table>
          ) : (
            <div className="empty-state">Select an adapter above.</div>
          )}
        </div>
      </div>

      <div className="note">Sources: {(feed.sources ?? []).join(", ")}.</div>
    </div>
  );
}
