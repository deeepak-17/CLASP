"""``python -m evaluation`` — the evaluation container's entrypoint.

The P5 harness (HumanEval/MBPP Pass@k, in-project eval, guard anchors, round
feed) runs as one-shot CLIs from the repository's ``scripts/``; this package
ships the metric every service imports (``evaluation.completion``). So the
container reports what it provides and, when ``CLASP_REGISTRY_URL`` is set,
whether the State Registry is reachable — then exits. Exit status is 1 only
when a registry URL was given and could not be reached.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from evaluation import __version__


def main() -> int:
    print(f"clasp-evaluation {__version__}: evaluation.completion (D5 in-project metric)")
    url = os.environ.get("CLASP_REGISTRY_URL")
    if not url:
        print("CLASP_REGISTRY_URL not set; skipping registry check")
        return 0
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=5) as resp:  # noqa: S310
            print(f"registry {url}: {json.loads(resp.read().decode('utf-8'))}")
        return 0
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(f"registry {url} unreachable: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
