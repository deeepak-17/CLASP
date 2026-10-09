#!/usr/bin/env bash
# Fresh-machine bring-up: prove the committed repo (nothing untracked, no local
# venv, no warm caches) installs, tests, packs and runs from scratch.
#
#   scripts/fresh_machine_check.sh [git-ref]      # default: HEAD
#
# 1. git clone the ref into an empty temp dir   — only committed files exist
# 2. in a bare python:3.11-slim container: install contracts + registry, run the
#    registry suite, validate every experiment config, build + verify a repro pack
# 3. from the same clone: docker compose builds the registry image and the
#    demo-seed job drives the whole lifecycle against it (fails on any wrong answer)
# Everything it creates (clone, compose project, volumes) is removed on exit.
set -euo pipefail

REF="${1:-HEAD}"
REPO="$(git rev-parse --show-toplevel)"
COMMIT="$(git -C "$REPO" rev-parse "$REF")"
WORK="$(mktemp -d)"
PROJECT="clasp-fresh-$$"
cleanup() {
  (cd "$WORK/CLASP" 2>/dev/null && docker compose -p "$PROJECT" --profile demo down -v >/dev/null 2>&1) || true
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "== 1. clean clone of $COMMIT"
git clone --quiet --no-hardlinks "$REPO" "$WORK/CLASP"
git -C "$WORK/CLASP" checkout --quiet "$COMMIT"

echo "== 2. install + test in a bare python:3.11-slim"
docker run --rm -v "$WORK/CLASP:/src:ro" python:3.11-slim bash -euo pipefail -c '
  apt-get update -qq >/dev/null && apt-get install -y -qq git >/dev/null
  cp -r /src /work && cd /work
  pip install --quiet --no-cache-dir -e contracts -e "services/registry[test]"
  python -m pytest contracts/tests -q -p no:cacheprovider | tail -1
  python -m pytest services/registry/tests -q -p no:cacheprovider \
    --cov=registry --cov-fail-under=90 | tail -1
  python -m registry.runs check experiments
  python -m registry.repro pack --out /tmp/pack.tar.gz --store /tmp/store
  python -m registry.repro verify /tmp/pack.tar.gz
'

echo "== 3. docker compose bring-up from the clone"
cd "$WORK/CLASP"
docker compose -p "$PROJECT" --profile demo up --build --exit-code-from demo-seed \
  registry demo-seed 2>&1 | grep -E "demo-seed-1 +\| (ok|FAIL|registry demo)" || true
docker compose -p "$PROJECT" --profile demo ps -a --format '{{.Service}} exit={{.ExitCode}}' \
  | grep -q "demo-seed exit=0" || { echo "FAIL: demo-seed did not pass"; exit 1; }

echo "== fresh-machine bring-up of $COMMIT: PASS"
