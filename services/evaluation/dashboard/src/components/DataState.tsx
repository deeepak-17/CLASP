/** Shared loading / error presentation for pages driven by useResults().
 * A malformed or missing results.json renders this, not a blank screen or
 * an uncaught exception — see src/lib/loadResults.ts. */
export function LoadingState() {
  return <div className="empty-state">Loading results.json…</div>;
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="panel">
      <div className="note" style={{ borderLeftColor: "var(--status-critical)" }}>
        <strong>Could not load results.json.</strong>
        <div style={{ marginTop: 6 }}>{message}</div>
        <div style={{ marginTop: 6 }}>
          Run <code>npm run sync-data</code> from <code>dashboard/</code> (copies the repository's{" "}
          <code>results/results.json</code> into <code>public/data/</code>), or produce it first
          with <code>python scripts/run_baseline_eval.py</code> from the repository root.
        </div>
      </div>
    </div>
  );
}

export function EmptyResultsState() {
  return (
    <div className="empty-state">
      results.json contains no scored runs yet. Run{" "}
      <code>python scripts/run_baseline_eval.py</code> from the repository root, then{" "}
      <code>npm run sync-data</code>.
    </div>
  );
}

/** An optional feed that has not been produced yet: say which script makes it. */
export function AbsentFeedState({ file, command }: { file: string; command: string }) {
  return (
    <div className="empty-state">
      <code>{file}</code> has not been produced yet. Run <code>{command}</code> from{" "}
      <code>services/evaluation</code>, then <code>npm run sync-data</code>.
    </div>
  );
}
