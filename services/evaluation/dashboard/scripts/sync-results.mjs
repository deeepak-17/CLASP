// Week 5 · Wednesday support script — "Wire skeleton to read static results.json"
//
// Copies the repository's canonical results/results.json (written
// by src/evaluation/eval_harness/results_store.py — see docs/eval_harness.md) into
// dashboard/public/data/results.json, where the dashboard fetches it at
// runtime as a static file. This is a plain file copy, not a build/transform
// step and not an API: the dashboard never talks to a backend, and Python
// remains the single source of truth for the data's shape and contents.
//
// Run automatically before `npm run dev` / `npm run build` (see package.json
// "predev"/"prebuild"); safe to run standalone: `npm run sync-data`.
import { copyFileSync, existsSync, mkdirSync, rmSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
// services/evaluation — the P5 module root (dashboard/scripts -> dashboard -> module).
const MODULE_ROOT = resolve(__dirname, "..", "..");
const SOURCE = resolve(MODULE_ROOT, "results", "results.json");
const DEST_DIR = resolve(__dirname, "..", "public", "data");
const DEST = resolve(DEST_DIR, "results.json");

// In-project completion metric artifact (the D5 primary signal). Copied the
// same way as results.json; the In-Project page fetches it at runtime.
const IN_PROJECT_SOURCE = resolve(MODULE_ROOT, "results", "in_project_metric.json");
const IN_PROJECT_DEST = resolve(DEST_DIR, "in_project_metric.json");

if (!existsSync(SOURCE)) {
  console.error(
    `[sync-data] ${SOURCE} does not exist.\n` +
      "  This dashboard reads the real results/results.json produced by\n" +
      "  scripts/run_baseline_eval.py (see docs/eval_harness.md). Run that script from\n" +
      "  services/evaluation first, e.g.:\n\n" +
      "    python scripts/run_baseline_eval.py --limit 1000 --num-samples 10 --k 1 10\n",
  );
  process.exit(1);
}

mkdirSync(DEST_DIR, { recursive: true });
copyFileSync(SOURCE, DEST);
const { size } = statSync(DEST);
console.log(`[sync-data] copied ${SOURCE} -> ${DEST} (${size} bytes)`);

if (existsSync(IN_PROJECT_SOURCE)) {
  copyFileSync(IN_PROJECT_SOURCE, IN_PROJECT_DEST);
  console.log(`[sync-data] copied ${IN_PROJECT_SOURCE} -> ${IN_PROJECT_DEST} (${statSync(IN_PROJECT_DEST).size} bytes)`);
} else {
  console.warn(`[sync-data] ${IN_PROJECT_SOURCE} not found — In-Project page will show an error state.`);
}

// Federated-round feed (scripts/export_round_feed.py). Optional: the Rounds
// page shows a "no round exported yet" state when it is absent.
const ROUNDS_SOURCE = resolve(MODULE_ROOT, "results", "rounds.json");
const ROUNDS_DEST = resolve(DEST_DIR, "rounds.json");
if (existsSync(ROUNDS_SOURCE)) {
  copyFileSync(ROUNDS_SOURCE, ROUNDS_DEST);
  console.log(`[sync-data] copied ${ROUNDS_SOURCE} -> ${ROUNDS_DEST} (${statSync(ROUNDS_DEST).size} bytes)`);
} else {
  if (existsSync(ROUNDS_DEST)) rmSync(ROUNDS_DEST);
  console.log(`[sync-data] ${ROUNDS_SOURCE} not found — Rounds page will show its empty state.`);
}

// Personalization, lineage and noise feeds (scripts/build_personalization_report.py,
// scripts/export_lineage.py, scripts/build_noise_report.py). Optional: each page
// shows how to produce its feed when the file is absent.
for (const name of ["personalization.json", "lineage.json", "noise_report.json"]) {
  const source = resolve(MODULE_ROOT, "results", name);
  const dest = resolve(DEST_DIR, name);
  if (existsSync(source)) {
    copyFileSync(source, dest);
    console.log(`[sync-data] copied ${source} -> ${dest} (${statSync(dest).size} bytes)`);
  } else {
    if (existsSync(dest)) rmSync(dest);
    console.log(`[sync-data] ${source} not found — its page will show how to produce it.`);
  }
}
