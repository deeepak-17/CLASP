// Generic loader for an optional static feed under public/data/ (copied there
// by scripts/sync-results.mjs). A missing file is "absent" — the page shows how
// to produce it — rather than an error; a malformed one is an error naming the
// guard that rejected it.
import { useEffect, useState } from "react";

export type FeedState<T> =
  | { status: "loading" }
  | { status: "absent"; file: string }
  | { status: "error"; message: string }
  | { status: "ready"; data: T };

export function useFeed<T>(file: string, guard: (x: unknown) => x is T, guardName: string): FeedState<T> {
  const [state, setState] = useState<FeedState<T>>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    const url = `${import.meta.env.BASE_URL}data/${file}`;

    async function load() {
      try {
        const response = await fetch(url, { cache: "no-store" });
        if (response.status === 404) {
          if (!cancelled) setState({ status: "absent", file });
          return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${url}`);
        const parsed: unknown = await response.json();
        if (!guard(parsed)) throw new Error(`${file} does not match the expected shape (see ${guardName}).`);
        if (!cancelled) setState({ status: "ready", data: parsed });
      } catch (err) {
        if (!cancelled) setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [file, guard, guardName]);

  return state;
}
