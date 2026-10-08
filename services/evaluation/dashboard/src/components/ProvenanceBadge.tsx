import type { Provenance } from "../lib/types";

/** REAL vs DEMO/PLACEHOLDER — mandatory, per the Week-5 brief. A record with
 * provenance "DEMO_TEST" is shown as "DEMO / PLACEHOLDER DATA", never
 * dressed up as a measured result. */
export function ProvenanceBadge({ provenance }: { provenance: Provenance }) {
  if (provenance === "REAL") {
    return (
      <span className="badge badge-real">
        <span className="badge-dot" />
        REAL EVALUATION
      </span>
    );
  }
  return (
    <span className="badge badge-demo">
      <span className="badge-dot" />
      DEMO / PLACEHOLDER DATA
    </span>
  );
}
