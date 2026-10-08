import type { ResultRecord } from "../lib/types";
import { formatOrNA } from "../lib/loadResults";

/** Compact experiment-metadata table — Step 10: model, checkpoint,
 * benchmark, dataset, samples/task, seed, generation config, timestamp.
 * Only fields present on the record are shown; anything absent renders
 * "N/A" rather than being invented or silently dropped. */
export function MetadataTable({ records }: { records: ResultRecord[] }) {
  if (records.length === 0) return null;

  return (
    <div style={{ overflowX: "auto" }}>
      <table className="data-table">
        <thead>
          <tr>
            <th>Benchmark</th>
            <th>Model / checkpoint</th>
            <th>Dataset split</th>
            <th className="num">Samples / task</th>
            <th className="num">Seed</th>
            <th>Generation config</th>
            <th>Run timestamp</th>
          </tr>
        </thead>
        <tbody>
          {records.map((r) => (
            <tr key={`${r.eval_result.run_id ?? "run"}-${r.eval_result.benchmark}`}>
              <td>{r.eval_result.benchmark}</td>
              <td className="mono">{formatOrNA(r.run_metadata.model_checkpoint)}</td>
              <td>{formatOrNA(r.run_metadata.dataset_split)}</td>
              <td className="num">{formatOrNA(r.eval_result.num_samples_per_task)}</td>
              <td className="num">{formatOrNA(r.run_metadata.seed)}</td>
              <td className="mono">
                max_new_tokens={r.run_metadata.generation.max_new_tokens}, temperature=
                {r.run_metadata.generation.temperature}
              </td>
              <td className="mono">{formatOrNA(r.eval_result.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
