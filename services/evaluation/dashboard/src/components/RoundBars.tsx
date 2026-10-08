import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/** Categorical slots in fixed order: round 1, round 2 (validated pair, light + dark). */
const SLOT = ["var(--series-humaneval)", "var(--series-mbpp)"];

interface RoundBarsProps {
  /** One row per client: { client, r1, r2, ... }. A round with no value for a client is left out, not zeroed. */
  data: Record<string, number | string>[];
  rounds: number[];
  /** Tooltip value formatter. */
  format: (v: number) => string;
  /** Draw the zero line (for signed values). */
  zeroLine?: boolean;
  height?: number;
}

/** Grouped bars, x = client, one series per round. Bars start at zero, so
 * length reads as magnitude (signed values extend either side of the zero line). */
export function RoundBars({ data, rounds, format, zeroLine = false, height = 300 }: RoundBarsProps) {
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 12, right: 12, left: 0, bottom: 0 }} barGap={2} barCategoryGap="24%">
          <CartesianGrid stroke="var(--gridline)" vertical={false} />
          <XAxis
            dataKey="client"
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
            axisLine={{ stroke: "var(--border-strong)" }}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(v: number) => format(v)}
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            width={56}
          />
          {zeroLine ? <ReferenceLine y={0} stroke="var(--border-strong)" /> : null}
          <Tooltip
            formatter={(value: number, name: string) => [format(value), name]}
            cursor={{ fill: "var(--accent-soft)" }}
            contentStyle={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              fontSize: 12,
              color: "var(--text-primary)",
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />
          {rounds.map((round, i) => (
            <Bar
              key={round}
              dataKey={`r${round}`}
              name={`Round ${round}`}
              fill={SLOT[i % SLOT.length]}
              radius={[4, 4, 0, 0]}
              maxBarSize={28}
              stroke="var(--surface-1)"
              strokeWidth={1}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
