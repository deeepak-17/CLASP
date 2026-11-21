# Panel Completion, Plan B (pieces 7–10: privacy budget, security wiring, re-cluster button, remote access)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce ε ≤ 8 over a two-round run, put the security library to work (roles from certificates, HMAC, JWT logins, middleware, audit), turn mTLS on in compose and the demo scripts, and show the re-clustering dry run on the demo panel.

**Architecture:** The privacy budget is a per-client ledger kept on the edge, charged at upload, checked again by the cluster and recorded by the registry. Services identify each other by certificate name mapped to a role; people log in with JWT on the browser pages; HMAC protects evaluation results and upload payloads against tampering. Every check falls back to "allowed, with a warning" when its secret or certificate is not configured (U6, C11).

**Tech Stack:** Python 3.10, Opacus 1.6 accountants, FastAPI/Starlette middleware, PyJWT (already a dependency of `security`), `hashlib.scrypt`, uvicorn TLS, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-14-panel-completion-design.md`. **Depends on Plan A being fully merged** (contracts 1.2.0, candidate flow, SQLite repository).

## Global Constraints

- All Global Constraints of Plan A apply unchanged (branching, user-only commits with no AI attribution, never push, test and lint commands, CPU-only CI, **[USER]** steps).
- Numbers fixed by the spec: run budget `ε = 8.0`, `δ = 1e-5`, run length `2` rounds (U3); JWT lifetime `8 hours`; rate limit default `600` requests per minute, burst `60`.
- Certificate names are the ones `scripts/make_dev_certs.sh` issues today: `registry`, `cluster`, `demo-ui`, `clasp-dev-client`, and one per edge equal to its client id (`client-flask`, …). This plan adds `clasp-admin` (C10).
- Fall-back rule (U6, C11), applied identically everywhere: **not configured → allowed, one warning logged at startup and an audit entry per request; configured but missing or wrong → refused.**
- Secrets never appear in logs, audit entries, test output or commit history. Generated key files live under `.runtime/` (gitignored).

## Review Focus

1. **A third DP upload in the same run:** refused on the edge before any network call, with a message naming the run id, the ε already spent and how to start a new run. Test in Task 7.2.
2. **An edge that has not pulled this change uploads a DP adapter without `epsilon_cumulative`:** the cluster refuses with 422 `privacy_budget_unreported` and a hint to update, and the upload is not buffered. Test in Task 7.3.
3. **A certificate issued for an old hotspot address:** the edge's upload fails with a hint that names the address it tried, the addresses in the server certificate, and the command that regenerates certificates. Test in Task 8.7.
4. **An expired or tampered JWT on the demo panel during a round:** every `/api/live/*` call returns 401 and the page shows the login form without losing the round's state on the server. Test in Task 8.5.
5. **A real-size upload (192 tensors) through the new middleware:** accepted; the request validator's body limit and the rate limiter must not reject a normal round of six clients. Test in Task 8.6.

---

# Piece 7: ε ≤ 8 across two rounds (branch `plan/07-privacy-budget`)

### Task 7.1: The budget ledger in the security library

**Files:**
- Create: `security/src/security/dp/ledger.py`
- Modify: `security/src/security/dp/__init__.py`, `security/src/security/__init__.py` (exports)
- Test: `security/tests/test_ledger.py`

**Interfaces (produced):**

```python
RUN_EPSILON_BUDGET = 8.0
RUN_DELTA = 1e-5
RUN_ROUNDS = 2

@dataclass(frozen=True)
class RoundRecord:
    round: int                # 1-based position in the run
    noise_multiplier: float
    sample_rate: float
    steps: int
    epsilon_round: float      # this round alone, as the training manifest reported it
    adapter_sha256: str       # what was uploaded
    uploaded_at: str

class BudgetLedger:
    def __init__(self, path: Path, *, budget: float = RUN_EPSILON_BUDGET, delta: float = RUN_DELTA,
                 rounds_total: int = RUN_ROUNDS, accountant: str = DEFAULT_ACCOUNTANT) -> None: ...
    def records(self) -> list[RoundRecord]: ...
    def cumulative_epsilon(self, extra: RoundRecord | None = None) -> float: ...
    def check(self, extra: RoundRecord) -> None: ...     # raises BudgetExhaustedError
    def charge(self, record: RoundRecord) -> float: ...  # check, append, fsync; returns the new cumulative ε
    def has(self, adapter_sha256: str) -> bool: ...

def ledger_path(root: Path, client_id: str, run_id: str) -> Path   # <root>/<client_id>/<run_id>.json

def calibrate_run_sigma(*, steps_per_round: int, sample_rate: float, rounds_total: int = RUN_ROUNDS,
                        budget: float = RUN_EPSILON_BUDGET, delta: float = RUN_DELTA,
                        accountant: str = DEFAULT_ACCOUNTANT) -> float
```

- `cumulative_epsilon` replays every record's `(noise_multiplier, sample_rate, steps)` into one Opacus accountant created by the existing `create_accountant` helper in `security/dp/accountant.py`, then reads ε at `delta`. It never sums per-round ε values.
- `check` refuses when `len(records) >= rounds_total` ("run already has N of N rounds") or when `cumulative_epsilon(extra) > budget + 1e-6`.
- `calibrate_run_sigma` returns the smallest σ (binary search, tolerance 0.005) for which `rounds_total` identical rounds compose to at most `budget` (C9).
- The ledger file is JSON `{"client_id", "run_id", "budget", "delta", "rounds_total", "records": [...]}`, written with tmp + fsync + replace.

- [ ] **Step 1: Failing tests:**

```python
def test_two_calibrated_rounds_fit_and_one_alone_is_below_budget(tmp_path):
    s = calibrate_run_sigma(steps_per_round=160, sample_rate=1 / 160)
    led = BudgetLedger(tmp_path / "l.json")
    e1 = led.charge(_rec(1, s, 1 / 160, 160)); e2 = led.charge(_rec(2, s, 1 / 160, 160))
    assert e1 < e2 <= 8.0 + 1e-6 and e2 > 7.5          # tight, not wasteful

def test_composition_is_not_a_sum(tmp_path):            # cumulative < epsilon_round_1 + epsilon_round_2
def test_third_round_is_refused_and_nothing_is_written(tmp_path)
def test_a_round_trained_for_epsilon_8_alone_makes_round_two_impossible(tmp_path)   # today's behaviour, now refused
def test_ledger_survives_reopen_and_has_finds_the_sha(tmp_path)
def test_corrupt_ledger_file_raises_and_is_not_overwritten(tmp_path)
```

- [ ] **Step 2:** Run `pytest security/tests/test_ledger.py -q` → FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5: Commit** `security: a per-client privacy ledger that composes rounds with the accountant`.

### Task 7.2: The edge trains with the run's noise level and charges the ledger at upload

**Files:**
- Modify: `services/edge/src/edge/train_client.py` (`train_dp`, `main`: flags `--run-id` default `"run-1"`, `--rounds-total` default `2`; manifest), `services/edge/src/edge/wire.py` (`privacy_block`, `privacy_from_training_manifest`), `services/edge/src/edge/upload.py` (`upload_adapter`, CLI), `services/edge/src/edge/webui.py` (`train_command`, status JSON), `services/edge/src/edge/webui_page.html`
- Test: `services/edge/tests/test_d3_dp.py`, `test_upload.py`, `test_wire.py`, `test_webui.py`

**Interfaces:**
- Consumes: Task 7.1.
- `train_dp(..., rounds_total: int = RUN_ROUNDS)`: σ comes from `calibrate_run_sigma(steps_per_round=<logical steps of this run>, sample_rate=<loader's>, rounds_total=rounds_total)` instead of the per-round `get_noise_multiplier(target_epsilon=8)` call. The training manifest's `privacy` gains `sample_rate`, `steps`, `run_id`, `rounds_total`, and `within_d7_budget` becomes "this round's ε is at most the run budget divided as calibrated".
- `upload_adapter(..., ledger_root: Path = <artifacts>/privacy_ledger)`: for a DP adapter, builds the `RoundRecord` from the training manifest, calls `ledger.check` **before any network call**, sends `privacy.epsilon_cumulative` and `privacy.rounds_accounted`, and calls `ledger.charge` only after the cluster answers 201. A retry of an adapter whose sha is already in the ledger re-sends the recorded values and does not charge twice (C8).
- `wire.privacy_block` emits `epsilon_cumulative`, `rounds_accounted`, `sample_rate`, `steps` when present.
- Edge page: status JSON gains `privacy: {"run_id", "epsilon_spent", "budget": 8.0, "rounds_used", "rounds_total": 2}`; the page shows it as "Privacy budget: ε 3.9 of 8 · round 1 of 2" and disables Upload when the ledger would refuse.

- [ ] **Step 1: Failing tests:**

```python
def test_train_dp_asks_for_the_two_round_sigma(monkeypatch)      # CPU, tiny model: calibrate_run_sigma called with rounds_total=2
def test_manifest_records_sample_rate_steps_and_run_id(tmp_path)
def test_upload_sends_cumulative_epsilon_and_charges_after_201(tmp_path, fake_cluster)
def test_upload_refused_by_the_cluster_does_not_charge(tmp_path, fake_cluster)       # 409 → ledger unchanged
def test_retrying_the_same_adapter_does_not_charge_twice(tmp_path, fake_cluster)
def test_third_upload_in_a_run_fails_before_any_request(tmp_path, fake_cluster):      # Review Focus 1
    # two charged rounds, then a third adapter: UploadError; fake_cluster.requests == [];
    # str(error) contains "run-1", "of 8", and "--run-id"
def test_non_dp_adapter_is_untouched_by_the_ledger(tmp_path, fake_cluster)
def test_page_status_reports_the_budget_and_disables_upload_when_exhausted(tmp_path)
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** `pytest services/edge/tests -q` → PASS.
- [ ] **Step 5: Commit** `edge: train for a two-round budget and charge the ledger when an adapter is uploaded`.

### Task 7.3: The cluster refuses over-budget uploads and reports honest cluster ε

**Files:**
- Modify: `services/cluster/src/cluster/schemas/messages.py` (`PrivacyBlock`: `epsilon_cumulative`, `rounds_accounted`, `sample_rate`, `steps`; `ClusterAdapterBroadcast`: `client_privacy`), `services/cluster/src/cluster/server.py` (`_StoredUpload`, `_upload_locked`, `cluster_epsilon` use in `_aggregate_locked` lines ~895–943, `publish_to_registry` meta)
- Test: `services/cluster/tests/test_privacy_budget.py` (new), `tests/integration/test_epsilon_seam.py`

**Interfaces (produced):**
- Upload rules for a DP upload (`epsilon` is not null): missing `privacy.epsilon_cumulative` → **422** `{"error": "privacy_budget_unreported", "hint": "update the edge: uploads must report the run's cumulative epsilon"}`; `epsilon_cumulative > 8.0 + 1e-6` → **422** `privacy_budget_exceeded`; `rounds_accounted > 2` → **422** `privacy_budget_exceeded`; `epsilon_cumulative < epsilon` → 422. A refused upload is not buffered.
- Cluster ε for a round = the maximum over the round's clients of `epsilon_cumulative`. The lines that add the base adapter's ε to this round's ε are removed. Manifest: `"epsilon_composition": "per-client accountant composition over the run; cluster epsilon is the max over clients"`, plus `"client_privacy": [{"client_id", "epsilon_round", "epsilon_cumulative", "rounds_accounted"}]`.
- Publish meta sent to the registry gains `"privacy": {"epsilon": <cluster ε>, "epsilon_cumulative": <same>}` and `"client_privacy": [...]` (an API-boundary extension of the meta envelope, like `set_active`).

- [ ] **Step 1: Failing tests:**

```python
def test_dp_upload_without_cumulative_is_422_and_not_buffered()      # Review Focus 2
def test_upload_over_eight_is_422_and_not_buffered()
def test_upload_claiming_a_third_round_is_422()
def test_cluster_epsilon_is_max_of_client_cumulative_not_a_sum()     # two rounds at 3.9/7.8 → 7.8, never 11.7
def test_non_dp_upload_still_makes_cluster_epsilon_none()
def test_publish_meta_carries_client_privacy()
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** cluster tests and `tests/integration/test_epsilon_seam.py` → PASS.
- [ ] **Step 5: Commit** `cluster: refuse uploads past the run's privacy budget; cluster epsilon is the clients' maximum`.

### Task 7.4: The registry records the ledger; the panel shows it; documents

**Files:**
- Modify: `services/registry/src/registry/db.py` (table), `repository.py` (two methods), `app.py` (`_parse_save_meta`, `save_version`, new route), `demo_ui/server.py` (`live_state`), `demo_ui/static/live.html`, `docs/registry_api.md`, `docs/dp_config.md`, `docs/decisions/2026-11-14-amended-decisions.md` (D7 section), `docs/devops.md` (ε row of the NFR table)
- Test: `services/registry/tests/test_privacy_ledger.py`, `tests/integration/test_epsilon_seam.py`, `test_demo_ui_live.py`

**Interfaces (produced):**
- Table `privacy_ledger(id PK, client_id NOT NULL, cluster_id NOT NULL, adapter_version_id FK, round INT, epsilon_round REAL, epsilon_cumulative REAL NOT NULL, rounds_accounted INT, delta REAL, recorded_at, UNIQUE(client_id, adapter_version_id))`, index `(client_id, recorded_at)`.
- `repo.add_privacy_rows(name: str, version: int, rows: list[dict]) -> None` (called inside the same transaction as `add_version`), `repo.privacy_ledger(cluster_id: str | None = None) -> list[dict]`.
- `GET /privacy/ledger?cluster_id=` → `{"budget": 8.0, "delta": 1e-5, "rows": [...]}`.
- Panel: each cluster card lists its clients with "ε 7.8 of 8 · 2 of 2 rounds".

- [ ] **Step 1: Failing tests:** an upload whose meta has `client_privacy` creates one row per client; a meta without it creates none and still succeeds; the route filters by cluster; integration: edge upload → aggregate → publish → `GET /privacy/ledger` shows the edge's cumulative ε unchanged end to end.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Full suite and ruff → PASS.
- [ ] **Step 5:** D7 section of the amended-decisions file: "ε ≤ 8 at δ = 1e-5 over the whole run of 2 rounds, per client, example-level; composed with the Opacus accountant; charged when an adapter is uploaded; enforced on the edge and at the cluster, recorded in the registry." State one consequence plainly: an uploaded round whose merge is later **rejected** has still spent its share of the budget, because the update left the laptop.
- [ ] **Step 6: Commit** `registry, demo: record and show each client's privacy spend`.
- [ ] **Step 7 [USER, GPU]:** Train `web/client-flask` with `--dp --run-id check-1` for round 1, upload, aggregate, promote, pull, train round 2, upload. Record in `experiments/d3-round2-dp/RESULTS.md`: σ used, ε after round 1, ε after round 2 (must be ≤ 8), and held-out loss before and after each round. **If held-out loss does not improve over the base model, report that to the user; do not change the budget, the round count or the batch size on your own.**

---

# Piece 8: security wiring (branch `plan/08-security-wiring`)

### Task 8.1: One peer-identity helper in the security library

**Files:**
- Create: `security/src/security/mtls/peer.py`
- Modify: `services/cluster/src/cluster/serve.py`, `services/cluster/src/cluster/tls.py` (import and re-export from the library), `services/registry/src/registry/serve.py`, `security/src/security/mtls/__init__.py`, `.github/workflows/ci.yml` (install `security` for the registry job as well as the cluster job), `services/registry/pyproject.toml` (dependency `clasp-security`), `services/registry/Dockerfile`
- Test: `security/tests/test_peer.py`, `services/registry/tests/test_serve.py`

**Interfaces (produced):** in `security.mtls.peer`: `SCOPE_KEY = "clasp.peer_common_name"`, `common_name_from_peercert(peercert: dict | None) -> str | None`, `class PeerCertH11Protocol(H11Protocol)`, `peer_identity(request) -> str | None`. These are moved from `cluster/serve.py` and `cluster/tls.py` without behaviour change; the cluster keeps the old import paths as re-exports. The registry's uvicorn config uses `http=PeerCertH11Protocol` when TLS is on.

- [ ] **Step 1:** Move the cluster's existing tests for these helpers to `security/tests/test_peer.py`; add a registry test that `build_config` selects the protocol class when the three `CLASP_TLS_*` variables are set and not otherwise. **Step 2:** Run → FAIL. **Step 3:** Move the code. **Step 4:** security, cluster and registry suites → PASS.
- [ ] **Step 5: Commit** `security: the verified peer certificate name is read in one place`.

### Task 8.2: Roles from certificate names, on the registry and the cluster

**Files:**
- Create: `security/src/security/auth/roles.py`
- Modify: `services/registry/src/registry/app.py` (dependencies on routes), `services/cluster/src/cluster/server.py` (dependencies on routes), `scripts/make_dev_certs.sh` (issue `admin` with CN `clasp-admin`)
- Test: `security/tests/test_roles.py`, `services/registry/tests/test_authz.py`, `services/cluster/tests/test_authz.py`

**Interfaces (produced):**

```python
ROLE_CLUSTER, ROLE_EVALUATION, ROLE_ADMIN, ROLE_EDGE, ROLE_OPEN = "svc:cluster", "svc:evaluation", "admin", "edge", "open"
DEFAULT_ROLES = {"cluster": ROLE_CLUSTER, "demo-ui": ROLE_EVALUATION, "clasp-admin": ROLE_ADMIN}
def role_of(common_name: str | None, *, table: Mapping[str, str] = DEFAULT_ROLES) -> str
    # None → ROLE_OPEN (no TLS: the fall-back); a name starting "client-" → ROLE_EDGE; unknown name → "unknown"
def require_roles(*allowed: str) -> Callable   # FastAPI dependency: ROLE_OPEN passes; a role outside `allowed` → 403
```

The table can be replaced by a JSON file named in `CLASP_ROLE_TABLE`.

Registry policy:

| Routes | Allowed roles |
|---|---|
| `GET /healthz` | everyone |
| all other `GET` | `svc:cluster`, `svc:evaluation`, `admin` |
| `POST /adapters/{name}/versions` | `svc:cluster`, `admin` |
| `POST .../promote`, `POST .../compose`, `POST /sync/{id}/done` | `svc:evaluation`, `admin` |
| `POST .../restore`, `POST .../gc`, `POST /gc` | `admin`, and `svc:evaluation` for `restore` only |

Cluster policy: uploads keep the existing rule (certificate name must equal `client_id`); `GET .../adapters/active` keeps the membership rule; `aggregate`, `publish`, `candidate/promote`, `candidate/reject`, `active/restore`, `recluster`, `PUT .../members`, `GET .../adapters/candidate` → `svc:evaluation`, `admin`.

- [ ] **Step 1: Failing tests,** table-driven: for each (route, role) pair in the two policies assert 2xx/4xx-not-403 for allowed, 403 for refused; one test that with no identity provider installed every route behaves as before (fall-back); one test that an `edge` role gets 403 from every registry route except `/healthz`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement; in tests, install an identity provider that reads a test header to simulate a certificate name. **Step 4:** PASS, registry coverage ≥ 90 %.
- [ ] **Step 5: Commit** `registry, cluster: each route names the service roles allowed to call it`.

### Task 8.3: HMAC on evaluation results

**Files:**
- Modify: `scripts/evaluate_candidate.py` (`evidence_document`), `demo_ui/evidence.py`, `demo_ui/server.py` (`live_promote`), `services/registry/src/registry/app.py` (`_parse_promote_body`, `_promote`), `scripts/make_dev_certs.sh` (write `.runtime/certs/secrets/eval_hmac.key`, 32 random bytes hex)
- Test: `services/registry/tests/test_promote_signature.py`, `tests/integration/test_live_evaluation.py`

**Interfaces (produced):**
- Secret: env `CLASP_EVAL_HMAC_SECRET`, or the file named by `CLASP_EVAL_HMAC_SECRET_FILE`.
- Signed message: `canonical = json.dumps({"adapter": name, "version": v, "candidate_sha256": sha, "eval": <EvalResult json>, "baseline_guard": [...]}, sort_keys=True, separators=(",", ":")).encode()`; `signature = security.auth.hmac_signer.sign_payload(canonical, secret)`.
- Promote body gains `candidate_sha256` and `eval_signature`.
- Registry: `candidate_sha256`, when present, must equal the stored version's sha256, else **409** `"evaluation was run on different bytes"`. With a secret configured: missing or wrong `eval_signature`, or missing `candidate_sha256` → **401**, nothing recorded. Without a secret: accepted, warning (fall-back rule).

- [ ] **Step 1: Failing tests:** valid signature → decision recorded; one changed metric → 401 and no decision; missing signature with secret set → 401; sha of another version → 409; no secret configured → accepted. **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5: Commit** `registry: evaluation results are signed and tied to the candidate's bytes`.

### Task 8.4: HMAC on edge uploads

**Files:**
- Modify: `services/cluster/src/cluster/schemas/messages.py` (`AdapterUpload.signature: str | None`), `services/cluster/src/cluster/server.py` (`_upload_locked`), `services/edge/src/edge/wire.py` (`upload_payload(..., hmac_secret: str | None = None)`), `services/edge/src/edge/upload.py` (`--hmac-key`, default `<mtls-dir>/hmac.key` when it exists), `scripts/make_dev_certs.sh` (per edge: `edges/<client_id>/hmac.key`; for the cluster: `cluster/upload_hmac.json` mapping client id → key), `scripts/make_edge_bundle.py`
- Test: `services/cluster/tests/test_upload_signature.py`, `services/edge/tests/test_wire.py`, `tests/integration/test_edge_bundle.py`

**Interfaces (produced):**
- Edge: `signature = security.auth.hmac_signer.sign_adapter_upload(tensors, metadata, secret)` where `tensors` maps each tensor name to its raw little-endian bytes (the bytes that are base64-encoded on the wire) and `metadata` is the upload body without `tensors`, `signature` and `timestamp`.
- Cluster: secrets from the JSON file named in `CLASP_UPLOAD_HMAC_SECRETS`. A client with a configured secret must send a signature that `verify_adapter_upload` accepts, else **401** `upload_signature_invalid` and the upload is not buffered. A client with no configured secret is accepted with a warning.

- [ ] **Step 1: Failing tests:** signed upload accepted; one flipped tensor byte → 401; changed `num_examples` → 401; unsigned upload from a client that has a secret → 401; client without a secret → accepted; the bundle for `client-flask` contains its `hmac.key` and no other edge's. **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5: Commit** `edge, cluster: uploads are signed by the edge and verified before they are buffered`.

### Task 8.5: JWT logins on the demo panel and the edge page

**Files:**
- Create: `security/src/security/auth/users.py`, `scripts/make_users.py`
- Modify: `demo_ui/server.py`, `demo_ui/static/live.html`, `demo_ui/static/common.js`, `services/edge/src/edge/webui.py` (`make_handler`, `main`: `--users`), `services/edge/src/edge/webui_page.html`, `demo_ui/requirements.txt`, `scripts/make_edge_bundle.py`
- Test: `security/tests/test_users.py`, `tests/integration/test_demo_ui_live.py`, `services/edge/tests/test_webui.py`

**Interfaces (produced):**

```python
class UserStore:                         # JSON file: {"users": {name: {"salt", "hash", "role", "subject"}}}
    def __init__(self, path: Path) -> None: ...
    def add(self, username: str, password: str, role: str, subject: str | None = None) -> None: ...   # scrypt n=2**14, r=8, p=1
    def verify(self, username: str, password: str) -> dict | None: ...                                # {"role", "subject"} or None
def issue_token(store: UserStore, username: str, password: str, config: JWTConfig) -> str | None      # via jwt_handler.create_token
def bearer_claims(authorization_header: str | None, config: JWTConfig) -> dict                        # raises jwt_handler.TokenError
```

- Roles: `operator` (demo panel), `developer` with `subject = <client id>` (edge page).
- Secret: `CLASP_JWT_SECRET` or `CLASP_JWT_SECRET_FILE`; lifetime 8 hours. Users file: `CLASP_USERS_FILE` for the panel, `--users` for the edge page.
- Both pages: `POST /api/login` `{username, password}` → `{"token", "role", "expires_at"}` or 401. Every other `/api/*` route except the health route requires `Authorization: Bearer <token>`; the panel requires role `operator`; the edge page requires `developer` whose `subject` equals the page's own client id. Wrong or expired token → 401; right token, wrong role or subject → 403.
- Pages keep the token in `sessionStorage`, send it on every call, and show the login form on any 401.
- No users file configured → the page runs open with a startup warning (fall-back rule).
- `scripts/make_users.py --out .runtime/users/panel.json --add operator:operator` (prompts for the password; never takes it as an argument) and the same for each edge.

- [ ] **Step 1: Failing tests:** `UserStore` round trip and wrong-password `None`; hash file contains no plaintext; login success and failure on both pages; a call without a token → 401; a `developer` token for `client-flask` on `client-numpy`'s page → 403; an expired token (lifetime patched to −1 s) on `/api/live/state` → 401 **and** a following login plus `GET /api/live/state` returns the same round state as before (Review Focus 4); no users file → routes open.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5: Commit** `demo, edge: people log in to the pages; tokens carry a role`.

### Task 8.6: Middleware and the audit log

**Files:**
- Create: `services/registry/src/registry/audit.py`
- Modify: `services/registry/src/registry/app.py`, `db.py`, `repository.py`, `services/cluster/src/cluster/server.py`, `demo_ui/server.py`, `docs/registry_api.md`
- Test: `services/registry/tests/test_middleware.py`, `test_access_log.py`, `services/cluster/tests/test_middleware.py`, `tests/integration/test_four_seams.py`

**Interfaces (produced):**
- Registry, cluster and demo panel each add, in this order: `SecurityHeadersMiddleware` (library defaults), `RequestValidationMiddleware(max_body_bytes=...)`, `RateLimitMiddleware(RateLimiter(RateLimitConfig(...)))`.
  - Body limit: registry = its existing upload cap (`_max_upload()`, 1 GiB) plus 1 MiB; cluster = 512 MiB; panel = 1 MiB.
  - Rate limit: `requests_per_minute = int(env CLASP_RATE_LIMIT_PER_MIN, default 600)`, `burst_size = 60`; the limiter's key is the certificate name when there is one, else the remote address; `/healthz` is exempt.
- One `security.audit.AuditLogger` per service, file from `CLASP_AUDIT_LOG` (default `<data dir>/audit.jsonl`). Logged: every refused authorisation (`log_auth_failure`), every accepted mutating call (`log_adapter_event`: upload, publish, promote, reject, restore, gc, recluster), every budget refusal and every recorded privacy row (`log_privacy_event`).
- Registry table `access_log(id PK, at, identity, role, method, path, adapter, version INT NULL, status INT)`, index `(identity, at)`; written for every non-health request by one middleware in `registry/audit.py`; `repo.add_access(row)`, `repo.access_log(limit: int = 200, identity: str | None = None)`; `GET /access-log` (role `admin`).

- [ ] **Step 1: Failing tests:** each service's responses carry the security headers; a body one byte over the limit → 413; request 61 in a burst from one identity → 429 while a different identity still gets 200; `/healthz` never 429; **a six-client round with 192-tensor uploads through the real apps passes with no 413 or 429** (Review Focus 5, extend `test_four_seams.py`); a promote writes one audit line and one `access_log` row with the caller's role; `GET /access-log` as `svc:cluster` → 403; no audit line contains a secret, token or signature value.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. If the library's middleware cannot key the limiter by identity as written, extend the library (with tests in `security/tests`) rather than wrapping it in the service. **Step 4:** Full suite and ruff → PASS.
- [ ] **Step 5: Commit** `registry, cluster, demo: security headers, body limits, rate limits and an audit trail`.

### Task 8.7: mTLS on in compose and the demo scripts; HTTPS for the pages

**Files:**
- Modify: `docker-compose.yml` (merge the TLS settings and certificate mounts from `docker-compose.mtls.yml`; add a one-shot `certs` service that runs `scripts/make_dev_certs.sh` when `.runtime/certs/ca.pem` is missing; secrets and users mounts), `scripts/run_services.py` (`--mtls-dir` defaults to `.runtime/certs`, generating if missing; new `--insecure`), `scripts/make_dev_certs.sh` (issue `pages` server certificate with the LAN addresses), `scripts/make_edge_bundle.py` (bundle always carries `edges/<client_id>/`), `services/edge/src/edge/webui.py` and `demo_ui` launcher (serve HTTPS when `CLASP_PAGE_TLS_CERT` and `CLASP_PAGE_TLS_KEY` are set), `services/evaluation/dashboard/Dockerfile` (nginx TLS when the certificate is mounted), `services/edge/src/edge/upload.py` (TLS error hint), `scripts/fresh_machine_check.sh`, `docs/mtls_setup.md`, `docs/devops.md`, `docs/multi-laptop-demo.md`, `docs/architecture.md` §4, `README.md`
- Delete: `docker-compose.mtls.yml`
- Create: `docker-compose.insecure.yml` (overlay that clears the TLS variables, for debugging only)
- Test: `tests/integration/test_run_services.py` (new), `services/edge/tests/test_upload.py`

**Interfaces (produced):**
- `run_services.py` with no flags starts registry and cluster with mTLS from `.runtime/certs`; `--insecure` starts plain HTTP and prints the existing warning. Starting a service by hand without the `CLASP_TLS_*` variables still runs plain HTTP with the warning (U6); nothing in `registry/serve.py` or `cluster/serve.py` changes.
- TLS failure hint on the edge: when the upload fails certificate verification because the server address is not in the certificate, `UploadError.hint` reads: `"<address> is not in the cluster's certificate (it lists: <SANs>). On laptop A run: CLASP_LAN_IPS=<address> scripts/make_dev_certs.sh, restart the services, and copy the new edge bundle."`

- [ ] **Step 1: Failing tests:** `run_services._env` yields the three TLS variables by default and none with `--insecure`; the certificate directory is generated when missing (patch the script call); the hostname-mismatch error produces the hint with the tried address and the certificate's addresses (Review Focus 3, build a throwaway CA with `security.mtls` in the test). **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Full suite and ruff → PASS.
- [ ] **Step 5:** Rewrite `docs/mtls_setup.md` and the security section of `docs/architecture.md`: which channels use mTLS, which pages use HTTPS + login, the role tables from Task 8.2, where HMAC applies, the fall-back rule, and "API-key authentication exists in the security library and is not used, because services are identified by certificate" (U4). Add to the amended-decisions file: D7 transport as built.
- [ ] **Step 6: Commit** `devops: mTLS and HTTPS are what the compose stack and the demo scripts start`.
- [ ] **Step 7 [USER, Docker Desktop running]:** `docker compose --profile demo up --build`; `curl --cacert .runtime/certs/ca.pem --cert .runtime/certs/client.pem --key .runtime/certs/client.key https://localhost:8004/healthz` → 200; the same without `--cert` → handshake refused; the panel at `https://localhost:8010` asks for a login.
- [ ] **Step 8 [USER, three laptops]:** Regenerate certificates with the hotspot address, rebuild and copy the edge bundles, and run one full round over mTLS with logins. Record the outcome in the "Live evidence" section of `docs/superpowers/plans/2026-10-14-federated-loop-audit-fedprox.md`.

---

# Piece 9: re-cluster button (branch `plan/09-recluster-button`)

### Task 9.1: Dry-run re-clustering on the demo panel

**Files:**
- Modify: `demo_ui/server.py` (new route, `live_state`), `demo_ui/static/live.html`, `demo_ui/static/common.js`, `demo_ui/static/live.css`, `docs/decisions/2026-11-14-amended-decisions.md` (D4 section), `docs/architecture.md`, `docs/multi-laptop-demo.md`
- Test: `tests/integration/test_demo_ui_live.py`

**Interfaces (produced):**
- `POST /api/live/recluster` → calls the cluster's `POST /recluster` with `{"apply": false, "include_similarity": true, "seed": 0}` and returns its response unchanged plus `"applied": false`. The panel never sends `apply: true`.
- `live_state` gains `"recluster_ready": bool`: true when every cluster with members has aggregated round 2 (its `round_id >= 2`).
- Page: a "Re-cluster (dry run)" button, disabled until `recluster_ready`; the result shows current and proposed membership side by side, clients that would move highlighted, the similarity matrix as a small table, and the cluster's own statement when it falls back to the static grouping. Caption: "Computed from round 2's updates. Nothing has been moved: applying a new grouping starts a new run with a new privacy budget."
- A cluster 409 (no aggregated round yet) is shown as its message, not as an error page.

- [ ] **Step 1: Failing tests:** the route forwards `apply: false` and `include_similarity: true` whatever the browser sends (post `{"apply": true}` and assert the fake cluster received `false`); `recluster_ready` is false after round 1 and true after round 2 for both clusters; a cluster 409 comes back as a 409 with the cluster's detail.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Full suite and ruff → PASS.
- [ ] **Step 5:** D4 section of the amended-decisions file (U7): "Rounds 1–2 use the project grouping. After the final round the cluster computes k-means (k = 2) on the cosine similarity of the clients' updates and reports the grouping it would choose. Applying it is the start of a new run." Note that the route requires the `operator` login (Task 8.5) and the cluster's `recluster` route requires `svc:evaluation` (Task 8.2).
- [ ] **Step 6: Commit** `demo: show the re-clustering the cluster would choose after round 2`.
- [ ] **Step 7 [USER, live round]:** After a real round 2, press the button and save the response as `experiments/clustering-ablation/results/live-dry-run-<date>.json`.

---

# Piece 10: remote access runbook (branch `plan/10-remote-access-runbook`)

### Task 10.1: WireGuard runbook

**Files:**
- Create: `docs/remote-access.md`
- Modify: `docs/multi-laptop-demo.md` (link; replace the "find the hotspot address" step with a pointer when the tunnel is used), `docs/architecture.md` (deployment section)

No code. The document covers, with exact commands for Windows laptops:
- Why a private tunnel and not a public address: the registry and cluster hold models derived from private code.
- Why plain WireGuard for this team (three laptops, fixed peers, no control server) and when Headscale becomes worth running (more than a handful of machines, or peers that change) (C12). Tailscale's hosted coordination is named and ruled out for the on-premise claim.
- Laptop A as the hub: key generation, one `[Peer]` block per edge laptop, a fixed tunnel address per laptop (`10.77.0.1` for A, `.2`, `.3` for B and C).
- Generating certificates for the tunnel address so they stop changing with the hotspot: `CLASP_LAN_IPS=10.77.0.1 scripts/make_dev_certs.sh`.
- A check list: `ping`, then the `curl` health call from Task 8.7 Step 7 over the tunnel address.
- What still must not be exposed: ports 8002, 8004, 8010 on any public interface.

- [ ] **Step 1:** Write the document.
- [ ] **Step 2 [USER, two laptops]:** Follow it once end to end and correct any command that did not work as written.
- [ ] **Step 3: Commit** `docs: reaching laptop A's services over a WireGuard tunnel`.

---

## After the last piece

- [ ] Run the complete suite and ruff one more time on `integration/demo` with all ten pieces merged, and record the counts.
- [ ] Re-read `README.md`, `docs/architecture.md` and `docs/decisions/2026-11-14-amended-decisions.md` against the six gaps and the registry review; every claim must name something that runs.
