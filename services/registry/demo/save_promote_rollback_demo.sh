#!/usr/bin/env bash
# CLASP State Registry (P4) — save -> promote -> rollback demo.
#
# Week 5 Mon target: "Demo script: save->promote->rollback cycle."
# Walks the full lifecycle against a *running* registry instance using only
# curl + python3 (stdlib), so it doubles as the Week 5 Thu "dry-run demo"
# and as rubric evidence for criterion 5 (docs/rubric.md).
#
# Usage:
#   docker compose up -d --build registry     # or: uvicorn registry.app:app --port 8004
#   services/registry/demo/save_promote_rollback_demo.sh [base_url] [adapter_name]
#
# Exits non-zero on the first unexpected response so a broken demo fails
# loudly in CI/dry-run instead of silently printing wrong output.
set -euo pipefail

BASE_URL="${1:-http://localhost:8004}"
NAME="${2:-demo-django}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }
expect_status() {
  # expect_status <got> <want> <label>
  if [ "$1" != "$2" ]; then
    echo "FAIL: $3 — expected HTTP $2, got $1" >&2
    exit 1
  fi
  echo "ok ($3 -> HTTP $2)"
}

# A minimal-but-structurally-valid safetensors blob (8-byte LE header length +
# JSON header + payload), built with stdlib only — mirrors
# tests/conftest.py::make_safetensors so the demo exercises the same
# validation path (`is_safetensors`) as the test suite.
make_safetensors() {
  python3 - "$1" <<'PY'
import json, struct, sys
values = [0.1, 0.2, 0.3, 0.4]
data = struct.pack(f"<{len(values)}f", *values)
header = {"lora_A": {"dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(data)]}}
header_bytes = json.dumps(header).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(header_bytes)))
    f.write(header_bytes)
    f.write(data)
PY
}

step "0. healthz"
curl -sf "$BASE_URL/healthz"; echo

step "1. save v1 (auto-activates per D9)"
make_safetensors "$WORKDIR/v1.safetensors"
v1=$(curl -s -o "$WORKDIR/v1.json" -w '%{http_code}' \
  -F "file=@$WORKDIR/v1.safetensors" \
  -F 'meta={"kind":"client"}' \
  "$BASE_URL/adapters/$NAME/versions")
expect_status "$v1" 201 "POST versions (v1)"
cat "$WORKDIR/v1.json"; echo

step "2. save v2 (auto-activates; v1 stays on disk, D9 immutability)"
make_safetensors "$WORKDIR/v2.safetensors"
v2=$(curl -s -o "$WORKDIR/v2.json" -w '%{http_code}' \
  -F "file=@$WORKDIR/v2.safetensors" \
  -F 'meta={"kind":"client"}' \
  "$BASE_URL/adapters/$NAME/versions")
expect_status "$v2" 201 "POST versions (v2)"
cat "$WORKDIR/v2.json"; echo

step "3. confirm v2 is active"
active_before=$(curl -sf "$BASE_URL/adapters/$NAME/active" | python3 -c 'import json,sys;print(json.load(sys.stdin)["ref"]["version"])')
echo "active version: $active_before"
[ "$active_before" = "2" ] || { echo "FAIL: expected v2 active, got v$active_before" >&2; exit 1; }

step "4. promote v2 — good eval (clears noise band, guard within tolerance) -> PROMOTE"
promote_good=$(curl -s -o "$WORKDIR/promote_good.json" -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -d "{
    \"eval\": {
      \"adapter\": {\"name\": \"$NAME\", \"version\": 2, \"kind\": \"client\"},
      \"in_project\": {\"edit_similarity\": 0.80, \"exact_match\": 0.5, \"perplexity\": 3.0, \"n_examples\": 20},
      \"guard\": [{\"benchmark\": \"HumanEval\", \"pass_at_k\": {\"1\": 0.30}}],
      \"baseline_in_project\": {\"edit_similarity\": 0.70, \"exact_match\": 0.5, \"perplexity\": 3.2, \"n_examples\": 20},
      \"baseline_noise_band\": 0.02
    },
    \"baseline_guard\": [{\"benchmark\": \"HumanEval\", \"pass_at_k\": {\"1\": 0.30}}]
  }" \
  "$BASE_URL/adapters/$NAME/promote")
expect_status "$promote_good" 200 "POST promote (good eval)"
cat "$WORKDIR/promote_good.json"; echo
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["action"]=="promote", d; assert d["active_version_after"]==2, d; print("assert ok: action=promote, active stays v2")' "$WORKDIR/promote_good.json"

step "5. save v3 (auto-activates)"
make_safetensors "$WORKDIR/v3.safetensors"
v3=$(curl -s -o "$WORKDIR/v3.json" -w '%{http_code}' \
  -F "file=@$WORKDIR/v3.safetensors" \
  -F 'meta={"kind":"client"}' \
  "$BASE_URL/adapters/$NAME/versions")
expect_status "$v3" 201 "POST versions (v3)"

step "6. promote v3 — bad eval (misses noise band AND guard tolerance) -> ROLLBACK to v2"
promote_bad=$(curl -s -o "$WORKDIR/promote_bad.json" -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -d "{
    \"eval\": {
      \"adapter\": {\"name\": \"$NAME\", \"version\": 3, \"kind\": \"client\"},
      \"in_project\": {\"edit_similarity\": 0.705, \"exact_match\": 0.4, \"perplexity\": 3.4, \"n_examples\": 20},
      \"guard\": [{\"benchmark\": \"HumanEval\", \"pass_at_k\": {\"1\": 0.10}}],
      \"baseline_in_project\": {\"edit_similarity\": 0.70, \"exact_match\": 0.5, \"perplexity\": 3.2, \"n_examples\": 20},
      \"baseline_noise_band\": 0.02
    },
    \"baseline_guard\": [{\"benchmark\": \"HumanEval\", \"pass_at_k\": {\"1\": 0.30}}]
  }" \
  "$BASE_URL/adapters/$NAME/promote")
expect_status "$promote_bad" 200 "POST promote (bad eval)"
cat "$WORKDIR/promote_bad.json"; echo
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["action"]=="rollback", d; assert d["active_version_after"]==2, d; print("assert ok: action=rollback, active reverted to v2")' "$WORKDIR/promote_bad.json"

step "7. confirm the pointer actually moved (not just the response body)"
active_after=$(curl -sf "$BASE_URL/adapters/$NAME/active" | python3 -c 'import json,sys;print(json.load(sys.stdin)["ref"]["version"])')
echo "active version after rollback: $active_after"
[ "$active_after" = "2" ] || { echo "FAIL: expected active v2 after rollback, got v$active_after" >&2; exit 1; }

step "8. lineage — full version list + active pointer"
curl -sf "$BASE_URL/adapters/$NAME/versions" | python3 -m json.tool

step "9. audit trail — promotion/rollback decisions"
curl -sf "$BASE_URL/adapters/$NAME/promotions" | python3 -m json.tool

echo
echo "=== DEMO OK: save -> save -> promote -> save -> rollback, lineage + audit trail verified ==="
