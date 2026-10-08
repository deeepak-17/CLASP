interface StatTileProps {
  label: string;
  value: string;
  hint?: string;
}

/** label + strong value + optional hint — never a "giant card". Renders
 * "N/A" (via the caller passing that string) in the muted style rather than
 * inventing a number. */
export function StatTile({ label, value, hint }: StatTileProps) {
  return (
    <div className="stat-tile">
      <span className="label">{label}</span>
      <span className={"value" + (value === "N/A" ? " na" : "")}>{value}</span>
      {hint ? <span className="hint">{hint}</span> : null}
    </div>
  );
}
