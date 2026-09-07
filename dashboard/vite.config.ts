import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// CLASP-P5 Evaluation Dashboard — Week 5 (React + Recharts skeleton).
// Static single-page app: no backend, no API. Data is read at runtime from
// /data/results.json (see scripts/sync-results.mjs and src/lib/loadResults.ts).
export default defineConfig({
  plugins: [react()],
  base: "./",
});
