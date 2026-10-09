"""Reproducibility pack (D9): everything needed to re-run and audit, in one file.

    python -m registry.repro pack   [--repo .] [--store DIR] [--registry-url URL]
                                    [--with-outputs] [--out clasp-repro.tar.gz]
    python -m registry.repro verify clasp-repro.tar.gz

The pack holds:

    MANIFEST.json            git commit (+ dirty flag), created_at, contracts
                             version, and a sha256 for every other file
    REPRODUCE.md             the commands to rebuild the environment and re-run
    environment/packages.json  installed distributions + versions; python, platform
    experiments/<name>/      each experiment's config.yaml and results write-ups
    runs/<exp>/<run_id>/     run.json (manifest v2) + config.resolved.json; outputs
                             only with --with-outputs (otherwise listed by sha256)
    registry/<adapter>/      versions, lineage and audit trail from a live
                             registry (metadata only — tensors are re-derivable
                             from configs + seeds and are verified by sha256)

``verify`` re-hashes every file against MANIFEST.json, re-validates every
config (seed, GPU-hours estimate, D8 caps) and checks that every run is
completed with a full manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

import yaml
from contracts import CONTRACTS_VERSION, utcnow_iso

from .runs import ResultStore, validate_config

_WRITEUPS = ("config.yaml", "RESULTS.md", "SUMMARY.md", "PRESENTATION.md")
_REQUIRED_MANIFEST = ("run_id", "seed", "config_hash", "gpu_hours_estimate", "gpu_hours_actual")

REPRODUCE = """# Reproducing this pack

1. Check out the recorded commit: `git checkout {commit}`
2. Rebuild the environment: `pip install -e contracts -e services/registry`
   (exact versions in `environment/packages.json`), or `docker compose build`.
3. Re-run an experiment exactly as archived — its sweep resumes, so only
   missing points run: `python -m registry.runs run experiments/<name>/config.yaml -- <command>`
   (each `runs/<exp>/<run_id>/run.json` records the command, seed and config hash).
4. Check this pack is intact: `python -m registry.repro verify <this file>`.
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                             text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _environment() -> dict:
    packages = sorted({(d.metadata["Name"], d.version) for d in metadata.distributions()
                       if d.metadata["Name"]})
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": [{"name": n, "version": v} for n, v in packages]}


def _get_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def _registry_files(base_url: str) -> dict[str, bytes]:
    base = base_url.rstrip("/")
    files: dict[str, bytes] = {}
    for name in _get_json(f"{base}/adapters")["adapters"]:
        for leaf, path in (("versions", "versions"), ("lineage", "lineage"),
                           ("promotions", "promotions")):
            try:
                body = _get_json(f"{base}/adapters/{name}/{path}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue  # e.g. an adapter dir with no versions yet
                raise  # a 500 or an auth failure must not yield a silently partial pack
            files[f"registry/{name}/{leaf}.json"] = json.dumps(body, indent=2).encode()
    return files


def _collect(repo_root: Path, store: ResultStore, with_outputs: bool,
             registry_url: str | None) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for exp in sorted((repo_root / "experiments").glob("*/")):
        if exp.name.startswith("_"):
            continue
        for leaf in _WRITEUPS:
            if (exp / leaf).is_file():
                files[f"experiments/{exp.name}/{leaf}"] = (exp / leaf).read_bytes()
    runs_root = store.root / "runs"
    for path in sorted(runs_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(runs_root)
        if "outputs" in rel.parts and not with_outputs:
            continue
        files[f"runs/{rel.as_posix()}"] = path.read_bytes()
    files["environment/packages.json"] = json.dumps(_environment(), indent=2).encode()
    if registry_url:
        files.update(_registry_files(registry_url))
    return files


def build_pack(*, repo_root: str | Path, store: ResultStore, out: str | Path,
               registry_url: str | None = None, with_outputs: bool = False) -> Path:
    repo_root, out = Path(repo_root), Path(out)
    files = _collect(repo_root, store, with_outputs, registry_url)
    commit = _git(repo_root, "rev-parse", "HEAD")
    dirty = bool(_git(repo_root, "status", "--porcelain")) if commit else None
    files["REPRODUCE.md"] = REPRODUCE.format(commit=commit or "<not a git checkout>").encode()
    manifest = {
        "pack_version": 1, "created_at": utcnow_iso(), "git_commit": commit,
        "git_dirty": dirty, "contracts_version": CONTRACTS_VERSION,
        "files": {name: _sha256(data) for name, data in sorted(files.items())},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for name, data in [("MANIFEST.json", json.dumps(manifest, indent=2).encode()),
                           *sorted(files.items())]:
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
    return out


@dataclass
class VerifyReport:
    problems: list[str] = field(default_factory=list)
    configs: int = 0
    runs: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems


def _check_run(name: str, record: dict, report: VerifyReport) -> None:
    report.runs += 1
    if record.get("status") != "completed":
        report.problems.append(f"{name}: run not completed (status={record.get('status')})")
    manifest = record.get("manifest") or {}
    for key in _REQUIRED_MANIFEST:
        if manifest.get(key) is None:
            report.problems.append(f"{name}: manifest missing {key}")


def verify_pack(path: str | Path) -> VerifyReport:
    report = VerifyReport()
    with tarfile.open(path) as tar:
        contents = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    try:
        manifest = json.loads(contents.pop("MANIFEST.json", b"{}") or b"{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return VerifyReport(problems=["MANIFEST.json is not valid JSON"])
    expected = manifest.get("files", {}) if isinstance(manifest, dict) else {}
    if not isinstance(expected, dict) or not expected:
        return VerifyReport(problems=["MANIFEST.json missing or empty"])
    for name in sorted(set(expected) | set(contents)):
        if name not in contents:
            report.problems.append(f"{name}: listed in MANIFEST.json but missing")
        elif name not in expected:
            report.problems.append(f"{name}: present but not in MANIFEST.json")
        elif _sha256(contents[name]) != expected[name]:
            report.problems.append(f"{name}: sha256 does not match MANIFEST.json")
    for name, data in sorted(contents.items()):
        if name.startswith("experiments/") and name.endswith("/config.yaml"):
            report.configs += 1
            config = yaml.safe_load(data)
            report.problems += [f"{name}: {p}" for p in
                                validate_config(config if isinstance(config, dict) else {})]
        elif name.startswith("runs/") and name.endswith("/run.json"):
            try:
                record = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError):
                report.problems.append(f"{name}: not valid JSON")
                continue
            _check_run(name, record if isinstance(record, dict) else {}, report)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m registry.repro")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--repo", default=".")
    p.add_argument("--store", default=None)
    p.add_argument("--registry-url", default=None)
    p.add_argument("--with-outputs", action="store_true")
    p.add_argument("--out", default="clasp-repro.tar.gz")
    v = sub.add_parser("verify")
    v.add_argument("pack")
    args = ap.parse_args(argv)

    if args.cmd == "pack":
        out = build_pack(repo_root=args.repo, store=ResultStore(args.store), out=args.out,
                         registry_url=args.registry_url, with_outputs=args.with_outputs)
        print(f"wrote {out}")
        return 0
    try:
        report = verify_pack(args.pack)
    except (OSError, tarfile.TarError) as e:
        print(f"error: cannot read {args.pack}: {e}", file=sys.stderr)
        return 2
    for problem in report.problems:
        print(f"FAIL {problem}")
    print(f"{'OK' if report.ok else 'FAILED'}: {report.configs} config(s), {report.runs} run(s)")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
