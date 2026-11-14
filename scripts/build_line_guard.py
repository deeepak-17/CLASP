"""Build the frozen line-completion guard set and write its manifest and NOTICE.

    python scripts/build_line_guard.py \
        --materialized datasets/materialized/web/_extra \
        --out services/evaluation/datasets/guard

Reads the three source repositories under ``--materialized``, writes
``line_guard_v1.jsonl``, ``line_guard_v1.manifest.json`` and ``NOTICE.md``
(all generated here). Source URL, pinned ref and licence come from the committed
partition files (``--partitions``); every row of a file must agree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "evaluation" / "src"))
sys.path.insert(0, str(REPO_ROOT / "services"))

from evaluation.completion import examples_fingerprint  # noqa: E402
from evaluation.eval_harness import line_guard  # noqa: E402

PALLETS_COPYRIGHT = "Copyright 2014 Pallets"
STRIDE = 7  # collect_examples' default; recorded in the manifest

BSD3 = """Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def read_provenance(partitions: Path, source: str) -> dict:
    seen = set()
    with (partitions / f"{source}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                seen.add((r["license"], r["source_url"], r["source_ref"]))
    if len(seen) != 1:
        raise SystemExit(f"{source}: partition rows disagree on provenance: {sorted(seen)}")
    lic, url, ref = next(iter(seen))
    return {"license": lic, "source_url": url, "source_ref": ref}


def copyright_line(source: str, source_dir: Path) -> str:
    """Pallets projects share one line; colorama's is quoted from its file header."""
    if source != "client-colorama":
        return PALLETS_COPYRIGHT
    for path in sorted((source_dir / "repo").rglob("*.py")):
        first = path.read_text(encoding="utf-8").splitlines()[0]
        if "Copyright" in first:
            return first.split(". BSD")[0].lstrip("# ").rstrip(".")
    raise SystemExit("no copyright header found in colorama sources")


def notice(entries: dict) -> str:
    out = ["# Third-party notices", "",
           "`line_guard_v1.jsonl` contains short excerpts of source code from the "
           "following BSD-3-Clause projects. Their copyright notices are kept here.", ""]
    for name, e in entries.items():
        out += [f"## {name[len('client-'):]}", "",
                f"- Source: {e['source_url']}",
                f"- Pinned ref: {e['source_ref']}",
                f"- Licence: {e['license']}",
                f"- {e['copyright']}", ""]
    out += ["## BSD 3-Clause License", "",
            "The text below applies to each project above, with that project's "
            "copyright holder as listed.", "", BSD3]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--materialized", type=Path,
                    default=REPO_ROOT / "datasets/materialized/web/_extra")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "services/evaluation/datasets/guard")
    ap.add_argument("--partitions", type=Path,
                    default=REPO_ROOT / "services/evaluation/datasets/partitions")
    ap.add_argument("--size", type=int, default=line_guard.GUARD_SIZE)
    ap.add_argument("--seed", type=int, default=line_guard.GUARD_SEED)
    args = ap.parse_args()

    dirs = [args.materialized / s for s in line_guard.GUARD_SOURCES]
    for d in dirs:
        if not (d / "repo").is_dir():
            raise SystemExit(f"missing materialized source: {d / 'repo'}")
    examples = line_guard.build_guard_set(dirs, size=args.size, seed=args.seed)

    sources = {}
    for name, d in zip(line_guard.GUARD_SOURCES, dirs):
        sources[name] = {**read_provenance(args.partitions, name),
                         "copyright": copyright_line(name, d),
                         "n_examples": sum(e.file.startswith(name + "/") for e in examples)}
    manifest = {"benchmark": line_guard.GUARD_BENCHMARK, "version": 1,
                "size": len(examples), "seed": args.seed, "stride": STRIDE,
                "fingerprint": examples_fingerprint(examples), "sources": sources}

    args.out.mkdir(parents=True, exist_ok=True)
    line_guard.save_guard_set(examples, args.out / "line_guard_v1.jsonl")
    (args.out / "line_guard_v1.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    (args.out / "NOTICE.md").write_text(notice(sources), encoding="utf-8", newline="\n")
    print(f"wrote {len(examples)} examples; fingerprint {manifest['fingerprint']}")
    for name, e in sources.items():
        print(f"  {name}: {e['n_examples']}  ({e['copyright']})")


if __name__ == "__main__":
    main()
