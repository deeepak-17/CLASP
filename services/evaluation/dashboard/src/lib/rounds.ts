// Loader for rounds.json (federated-round feed). Same pattern as inProject.ts,
// except a missing file is an expected "no rounds exported yet" state, not an error.
import { useEffect, useState } from "react";
import { isRoundsFeed, type RoundsFeed } from "./roundsFeed";

const DATA_URL = `${import.meta.env.BASE_URL}data/rounds.json`;

export type RoundsState =
  | { status: "loading" }
  | { status: "missing" }
  | { status: "error"; message: string }
  | { status: "ready"; data: RoundsFeed };

export function useRounds(): RoundsState {
  const [state, setState] = useState<RoundsState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const response = await fetch(DATA_URL, { cache: "no-store" });
        if (response.status === 404) {
          if (!cancelled) setState({ status: "missing" });
          return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${DATA_URL}`);
        const text = await response.text();
        // Vite's dev server answers unknown paths with index.html, not a 404.
        if (text.trimStart().startsWith("<")) {
          if (!cancelled) setState({ status: "missing" });
          return;
        }
        const parsed: unknown = JSON.parse(text);
        if (!isRoundsFeed(parsed)) {
          throw new Error("rounds.json does not match the expected shape (see dashboard/src/lib/roundsFeed.ts).");
        }
        if (!cancelled) setState({ status: "ready", data: parsed });
      } catch (err) {
        if (!cancelled) setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
