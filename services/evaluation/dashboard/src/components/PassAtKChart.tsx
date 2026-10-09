import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ResultRecord } from "../lib/types";
import { allKValues } from "../lib/loadResults";
import { recordKey, recordLabel } from "../lib/records";

interface PassAtKChartProps {
  records: ResultRecord[];
}

/** Categorical slots in fixed order, one per record (validated in light and dark). */
const SLOT = ["var(--series-humaneval)", "var(--series-mbpp)", "var(--series-3)"];

/** Grouped bar chart, x=k, y=Pass@k rate, one bar-series per benchmark.
 *
 * Chosen over a line chart deliberately: with only 1-2 discrete k values
 * (typically k=1 and k=10), a line would draw a slope across the gap that
 * implies a trend the data does not contain — see the dataviz skill's
 * "compare magnitude" guidance. A bar chart makes no such claim.
 *
 * No fake data: rows come straight from results.json's pass_at_k maps; a k
 * a benchmark doesn't report is simply absent from that bar group (Recharts
 * renders nothing for a missing key, not a zero-value bar).
 */
export function PassAtKChart({ records }: PassAtKChartProps) {
  // One series per record (not per benchmark): two runs of the same benchmark —
  // e.g. a DEMO_TEST mock and a REAL anchor — must never overwrite each other.
  const series = records.map((r, i) => ({ key: recordKey(r), label: recordLabel(r), record: r, color: SLOT[i] ?? "var(--text-muted)" }));
  const ks = allKValues(records);

  if (ks.length === 0) {
    return <div className="empty-state">No pass@k values are present in results.json.</div>;
  }

  const data = ks.map((k) => {
    const row: Record<string, number | string> = { k: `k = ${k}` };
    for (const s of series) {
      const value = s.record.eval_result.pass_at_k[String(k)];
      if (value !== undefined) {
        row[s.key] = value;
      }
    }
    return row;
  });

  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 12, right: 12, left: 0, bottom: 0 }} barGap={4} barCategoryGap="28%">
          <CartesianGrid stroke="var(--gridline)" strokeDasharray="0" vertical={false} />
          <XAxis
            dataKey="k"
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
            axisLine={{ stroke: "var(--border-strong)" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            width={44}
          />
          <Tooltip
            formatter={(value: number, name: string) => [`${(value * 100).toFixed(1)}%`, name]}
            contentStyle={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              fontSize: 12,
            }}
            cursor={{ fill: "var(--accent-soft)" }}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }}
            formatter={(value: string) => <span style={{ color: "var(--text-secondary)" }}>{value}</span>}
          />
          {series.map((s) => (
            <Bar
              key={s.key}
              dataKey={s.key}
              name={s.label}
              fill={s.color}
              radius={[4, 4, 0, 0]}
              maxBarSize={24}
            >
              <LabelList
                dataKey={s.key}
                position="top"
                formatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                style={{ fill: "var(--text-secondary)", fontSize: 11 }}
              />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
