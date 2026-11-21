# Panel Completion, Plan A (pieces 1–6: documents, durability, contracts, guard, candidate flow, database)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the registry, cluster, evaluation and demo panel behave the way the amended decisions say: a line-completion guard, publish-as-candidate with promote-to-live, and a SQLite metadata store with transactional promotion.

**Architecture:** Six pieces, each on its own branch off `integration/demo`, merged in order (U13). Pieces 4 and 5 change promotion behaviour first; piece 6 then builds the database once around the final behaviour. Weight files stay on disk; only metadata, pointers and decisions move to SQLite.

**Tech Stack:** Python 3.10 `.venv` (torch 2.14+cu126, peft 0.20, opacus 1.6), FastAPI, SQLAlchemy 2.x Core on SQLite, pytest, ruff 0.15.20.

**Spec:** `docs/superpowers/specs/2026-11-14-panel-completion-design.md` (decision ids U1–U13, C1–C13 are cited below). Plan B (`2026-11-14-panel-completion-b-privacy-security.md`) covers pieces 7–10 and depends on this plan.

## Global Constraints

- Repo `CLASP_Team_Bundle/CLASP`. Start each piece with `git switch integration/demo && git pull --ff-only && git switch -c plan/<NN>-<name>`. A piece starts only after the previous piece is merged.
- Commits are authored as the user with **no AI attribution or co-author trailer** (standing rule). Commit subjects follow the repo style: `area: lower-case summary`.
- **Never push, open a pull request or merge.** Stop at the end of each piece and tell the user the branch is ready.
- Test command per package: `.venv/Scripts/python.exe -m pytest <path> -q -p no:cacheprovider`. Lint: `.venv/Scripts/python.exe -m ruff check .` must be clean for the whole repo.
- CI is CPU-only: any test that needs the real model is marked `skipif(not torch.cuda.is_available())`.
- Edge code imports other packages; it never copies their logic. `contracts/` and `security/` are the only packages other modules may import across module boundaries.
- Every piece ends with the full run: contracts, security, edge, cluster, registry, evaluation, `tests/integration`, `tests/e2e`, and ruff. Baseline at `381d08d`: 22 / 143 / 178 (+1 skip) / 261 (+2 skip) / 209 / 510 (+1 skip) / 45 / 1.
- Steps marked **[USER]** need the user (GPU, Docker Desktop, team sign-off). Do not fake their output; stop and ask.
- Numbers fixed by the spec: guard tolerance `0.02`; guard set size `300`; guard seed `0`.

## Review Focus

1. **A registry volume created before the database exists** (version folders and `active` files, no `registry.db`): on first start every version, pointer and decision must appear through the API exactly as before. Test in Task 6.3.
2. **The demo panel stops between the registry's PROMOTE and the cluster's promote call:** after a restart the cluster must end up serving the promoted adapter without anyone re-evaluating. Test in Task 6.5.
3. **A laptop that has not pulled sends an old promote body** (HumanEval guard, no `LineCompletion` entry): the registry answers 422 naming the missing guard and the contracts version it expects, and records no decision. Test in Task 4.3.
4. **Aggregate is pressed while a candidate is still waiting for a decision:** 409 with a message saying which candidate is pending; uploads already buffered are kept. Test in Task 5.2.
5. **The very first candidate of a cluster is rejected:** nothing is active afterwards, edges still get "no adapter yet" from the cluster, and the next round trains from the base model again. Test in Tasks 5.1 and 5.2.

---

# Piece 1: documents match the build (branch `plan/01-docs-match-build`)

Only statements that are already true at `381d08d` change here. Wording that later pieces make true (guard, candidate flow, ε budget, mTLS default, re-cluster button) is edited in those pieces.

### Task 1.1: Decision record and architecture

**Files:**
- Create: `docs/decisions/2026-11-14-amended-decisions.md`
- Create: `docs/registry_api.md`
- Modify: `docs/architecture.md`, `README.md`, `docs/threat_model.md`, `docs/dp_config.md`

- [ ] **Step 1: Write `docs/decisions/2026-11-14-amended-decisions.md`.** One section per amended decision, each with *as written*, *as built*, *reason*:
  - **D1:** web cluster is {flask, requests, werkzeug}, not django (copy the reason text from `docs/dataset.md:61-66`).
  - **D2/D3:** since 2026-10-14 the cluster adds each round onto the last good merge (`C_{N+1} = C_N + (1/α)·Σ ŵ R`) and edges train with FedProx μ = 0.01.
  - **D6 (U1):** the composite is merged **on the edge** at pull time. The registry's `/compose` path and the `composite` block of `/promote` exist, are tested, and are not called by the live loop.
  - **D7 (U2):** DP is **example-level, applied on each client** (Opacus DP-SGD over 1024- or 512-token blocks, LoRA parameters only). State plainly that this is weaker than client-level DP because one developer contributes many blocks.
  - **Base model:** design names DeepSeek-Coder-6.7B; all Phase II runs use the 1.3B profile because the team's GPUs have 4 GB.
  - **Cluster transport:** the live cluster is a FastAPI service; Flower is used in `cluster/simulation.py` and the experiment code only.
  - **Evaluation:** a library plus scripts (`scripts/evaluate_candidate.py`), not a long-running service.
- [ ] **Step 2: Edit `docs/architecture.md`.** §1: D6 bullet and base-model bullet per Step 1. §2 table: cluster row "FastAPI service; FedProx; SVD aggregation; Flower in simulation only", evaluation row "library + scripts + static dashboard", contracts "1.1.0". §3 diagram: the return arrow goes **Cluster → Edge** ("merged cluster adapter; edge merges with its own client layer"), and Evaluation → Registry stays. §3a: replace the "Registry → Edge" seam row with "Cluster → Edge `GET /clusters/{id}/adapters/active`"; delete the sentence "nothing on this channel is differentially private or mutually authenticated". §4: remove "Auth: … currently open" (piece 8 rewrites it).
- [ ] **Step 3: Edit `README.md`** to match: opening D6/base-model bullets, "Flower + FedProx aggregation (FastAPI)" → "FedProx + SVD aggregation (FastAPI; Flower for simulation)", module table, tech stack line.
- [ ] **Step 4: Edit `docs/threat_model.md` and `docs/dp_config.md`:** every "client-level" becomes "example-level, applied per client", with the one-sentence consequence from Step 1.
- [ ] **Step 5: Write `docs/registry_api.md`** from the route docstrings in `services/registry/src/registry/app.py` (13 routes: method, path, body, status codes, which are mutating). `docs/architecture.md` §4 already links to it.
- [ ] **Step 6: Verify.** Run: `grep -rn -i "client-level" docs README.md` → only the amended-decisions file (in its "as written" lines). Run: `grep -rn "registry_api.md" docs` → the link resolves.
- [ ] **Step 7: Commit** `docs: record amended decisions and make the architecture match the build`.

### Task 1.2: Lint errors in `proof_of_work/`

**Files:** Modify `proof_of_work/common.py:30`, `proof_of_work/w1_scaffold_env.py:112`

- [ ] **Step 1:** Run `.venv/Scripts/python.exe -m ruff check .` → expect exactly 2 errors (F401 unused `typing.Optional`; E731 lambda assignment).
- [ ] **Step 2:** Remove the unused import; turn the assigned lambda into a `def` with the same name.
- [ ] **Step 3:** Run ruff → `All checks passed!`.
- [ ] **Step 4:** `proof_of_work/` is untracked. **Ask the user whether it should be committed**; if not, leave the fixes in the working tree and commit nothing for this task.

---

# Piece 2: ResultStore durability (branch `plan/02-resultstore-fsync`)

### Task 2.1: Shared durable-write helper, used by the result store

**Files:**
- Create: `services/registry/src/registry/durable.py`
- Modify: `services/registry/src/registry/storage.py` (imports the helper under its existing private name), `services/registry/src/registry/runs.py:168-171,302`, `services/registry/src/registry/manifest.py:44`
- Test: `services/registry/tests/test_durability.py` (extend)

**Interfaces:**
- Produces: `durable.fsync_path(path: Path) -> None` (the body of today's `storage._fsync`, moved unchanged, including the Windows directory skip); `durable.atomic_write_text(path: Path, text: str) -> None` (write `<path>.tmp` → `fsync_path(tmp)` → `os.replace` → `fsync_path(path.parent)`).
- `storage.py` keeps the name `_fsync` (`from .durable import fsync_path as _fsync`), because `test_durability.py` patches `storage._fsync`.

- [ ] **Step 1: Failing tests** appended to `test_durability.py`. Read the existing tests first and reuse their call-recording fixture.

```python
def test_atomic_write_text_syncs_file_then_replaces_then_syncs_dir(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(durable, "fsync_path", lambda p: calls.append(("fsync", p.name)))
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append(("replace", Path(b).name)), real_replace(a, b)))
    durable.atomic_write_text(tmp_path / "run.json", "{}")
    assert calls == [("fsync", "run.json.tmp"), ("replace", "run.json"), ("fsync", tmp_path.name)]

def test_run_record_config_and_manifest_all_go_through_atomic_write(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(durable, "atomic_write_text", lambda p, t: written.append(Path(p).name))
    # write one run record via ResultStore.write_record, one resolved config via the code path
    # at runs.py:302, and one manifest via manifest.write_manifest
    assert {"run.json", "config.resolved.json"} <= set(written)
    assert any(name.endswith(".json") and "manifest" in name for name in written)
```

- [ ] **Step 2:** Run `pytest services/registry/tests/test_durability.py -q` → the two new tests FAIL (`durable` does not exist).
- [ ] **Step 3: Implement.** Create `durable.py`; `runs._atomic_write` becomes a call to `durable.atomic_write_text`; `runs.py:302` and `manifest.write_manifest` call it instead of `write_text`. Call it as `durable.atomic_write_text(...)` (module attribute), so the monkeypatch in the test takes effect.
- [ ] **Step 4:** Run `pytest services/registry/tests -q` → 211 passed.
- [ ] **Step 5:** Add `_demo_data/` to `.gitignore` under the "Experiment outputs" block (it is the documented demo data path, `demo_ui/README.md:45`). Run `git status --short` → no `_demo_data` entries.
- [ ] **Step 6: Commit** `fix: fsync the experiment result store's records, configs and manifests` (the `.gitignore` line goes in a second commit, `chore: ignore _demo_data`).

---

# Piece 3: contracts 1.2.0 (branch `plan/03-contracts-1-2`)

### Task 3.1: Additive fields and the version bump

**Files:**
- Modify: `contracts/src/contracts/types.py` (`CONTRACTS_VERSION`, `PromotionAction`, `PrivacySpec`, `GuardMetrics`), `contracts/src/contracts/__init__.py:32`, `contracts/pyproject.toml:3`
- Test: `contracts/tests/test_v1_2.py` (new)

**Interfaces (produced; pieces 4, 5 and 7 rely on these exact names):**
- `CONTRACTS_VERSION = "1.2.0"`.
- `PromotionAction.REJECT = "reject"` (C6): the candidate was judged and not made live; the active pointer did not move.
- `GuardMetrics(benchmark: str, pass_at_k: dict[int, float] = {}, edit_similarity: float | None = None, exact_match: float | None = None, n_examples: int | None = None)`. `benchmark` gains the value `"LineCompletion"`. Validation: `edit_similarity` and `exact_match`, when present, are in `[0, 1]`; `n_examples`, when present, is `>= 1`. `pass_at_k` validation is unchanged.
- `PrivacySpec(..., epsilon_cumulative: float | None = None, rounds_accounted: int | None = None)`. Validation: when both `epsilon` and `epsilon_cumulative` are present, `epsilon_cumulative >= epsilon`; `rounds_accounted >= 1`.
- Every `from_json` accepts a 1.1.0 payload (the new keys absent); every `to_json` emits the new keys.

- [ ] **Step 1: Failing tests** in `contracts/tests/test_v1_2.py`:

```python
def test_version_is_1_2_0():
    assert contracts.CONTRACTS_VERSION == contracts.__version__ == "1.2.0"

def test_line_completion_guard_round_trips():
    g = GuardMetrics(benchmark="LineCompletion", edit_similarity=0.61, exact_match=0.22, n_examples=300)
    assert GuardMetrics.from_json(g.to_json()) == g and g.pass_at_k == {}

def test_guard_from_1_1_payload_still_parses():
    g = GuardMetrics.from_json({"benchmark": "HumanEval", "pass_at_k": {"1": 0.3}})
    assert g.pass_at_k == {1: 0.3} and g.edit_similarity is None

@pytest.mark.parametrize("field,value", [("edit_similarity", 1.2), ("exact_match", -0.1), ("n_examples", 0)])
def test_guard_rejects_out_of_range(field, value):
    with pytest.raises(ValueError):
        GuardMetrics(benchmark="LineCompletion", **{field: value})

def test_privacy_cumulative_round_trips_and_is_checked():
    p = PrivacySpec(epsilon=3.9, epsilon_cumulative=7.8, rounds_accounted=2)
    assert PrivacySpec.from_json(p.to_json()) == p
    assert PrivacySpec.from_json({"epsilon": 3.9, "delta": 1e-5}).epsilon_cumulative is None
    with pytest.raises(ValueError):
        PrivacySpec(epsilon=5.0, epsilon_cumulative=4.0)

def test_reject_action_round_trips():
    d = PromotionDecision(adapter=AdapterRef("cluster-web", 3, AdapterKind.CLUSTER),
                          action=PromotionAction.REJECT, active_version_after=2, reason="r")
    assert PromotionDecision.from_json(d.to_json()).action is PromotionAction.REJECT
```

- [ ] **Step 2:** Run `pytest contracts/tests -q` → new tests FAIL.
- [ ] **Step 3: Implement** the Interfaces block. `pass_at_k` uses `field(default_factory=dict)`; keep `benchmark` first so positional construction in existing code still works.
- [ ] **Step 4:** Run every package's tests. Expect only assertions on the literal `"1.1.0"` to fail; update those literals and nothing else.
- [ ] **Step 5: Commit** `contracts: 1.2.0 adds the line-completion guard, cumulative epsilon and a reject action`.
- [ ] **Step 6 [USER]:** Team rule: a contracts bump needs all-hands sign-off. Stop and ask the user to confirm sign-off before this branch is merged.

---

# Piece 4: line-completion guard (branch `plan/04-line-completion-guard`)

### Task 4.1: The frozen guard set

**Files:**
- Create: `services/evaluation/src/evaluation/eval_harness/line_guard.py`
- Create: `scripts/build_line_guard.py`
- Create: `services/evaluation/datasets/guard/line_guard_v1.jsonl`, `services/evaluation/datasets/guard/line_guard_v1.manifest.json`
- Test: `services/evaluation/tests/test_line_guard.py`

**Interfaces (produced):**
- `line_guard.GUARD_BENCHMARK = "LineCompletion"`, `GUARD_SIZE = 300`, `GUARD_SEED = 0`, `GUARD_SOURCES = ("client-click", "client-jinja", "client-colorama")`, `GUARD_FILE: Path` (the committed JSONL).
- `build_guard_set(source_dirs: Sequence[Path], *, size: int = GUARD_SIZE, seed: int = GUARD_SEED) -> list[CompletionExample]`: calls `evaluation.completion.collect_examples` on each source's `repo/` directory, prefixes each example's `file` with the source name, then takes a seeded sample of `size` spread evenly across the three sources (100 each; if a source has fewer, the remainder is split across the others).
- `save_guard_set(examples, path)`, `load_guard_set(path: Path = GUARD_FILE) -> list[CompletionExample]` (one JSON object per line: `file`, `line_no`, `prompt`, `target`).
- `guard_metrics(scored: dict, n_examples: int) -> dict` → `{"benchmark": "LineCompletion", "edit_similarity": ..., "exact_match": ..., "n_examples": ...}`, the `GuardMetrics.to_json` shape.
- `paired_noise(base_rows: Sequence[float], cand_rows: Sequence[float], *, n_boot: int = 1000, seed: int = 0) -> float`: half-width of the 95 % bootstrap interval of the mean per-example edit-similarity difference (resample example indices with replacement, same indices for both).

The source repositories are under `datasets/materialized/web/_extra/<client>/repo` (gitignored), which is why the built set is committed. All three are BSD-3-Clause; the manifest records source URL, ref and licence per source (read them from each `materialize_manifest.json`) plus `evaluation.completion.examples_fingerprint` of the set.

- [ ] **Step 1: Failing tests:**

```python
def test_build_is_deterministic_and_balanced(tmp_path):
    srcs = [_fake_repo(tmp_path, n, files=4, lines=200) for n in line_guard.GUARD_SOURCES]
    a = line_guard.build_guard_set(srcs, size=30, seed=0)
    assert a == line_guard.build_guard_set(srcs, size=30, seed=0)
    assert len(a) == 30
    assert Counter(e.file.split("/")[0] for e in a) == {s: 10 for s in line_guard.GUARD_SOURCES}

def test_short_source_is_topped_up_from_the_others(tmp_path): ...   # one source yields 3 examples → still 30 total

def test_committed_guard_set_matches_its_manifest():
    ex = line_guard.load_guard_set()
    man = json.loads(line_guard.GUARD_FILE.with_suffix(".manifest.json").read_text())
    assert len(ex) == 300 and examples_fingerprint(ex) == man["fingerprint"]

def test_paired_noise_is_zero_for_identical_rows_and_positive_otherwise():
    assert line_guard.paired_noise([0.5] * 50, [0.5] * 50) == 0.0
    assert line_guard.paired_noise([0.0, 1.0] * 25, [1.0, 0.0] * 25) > 0.1
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement** `line_guard.py` and `scripts/build_line_guard.py` (`--materialized datasets/materialized/web/_extra`, `--out services/evaluation/datasets/guard`).
- [ ] **Step 4:** Run `scripts/build_line_guard.py`; it writes the JSONL and the manifest. Confirm 300 lines. If `_extra` is missing on this machine, stop and ask the user where the materialized corpus is.
- [ ] **Step 5:** Run `pytest services/evaluation/tests/test_line_guard.py -q` → PASS.
- [ ] **Step 6: Commit** `evaluation: a frozen 300-example line-completion guard set from click, jinja and colorama`.

### Task 4.2: Candidate evaluation scores the guard set instead of HumanEval

**Files:**
- Modify: `scripts/evaluate_candidate.py` (`score_version`, `_guard`, `evidence_document`, CLI: drop `--max-new-tokens`), `demo_ui/evidence.py` (`GUARD_BENCHMARK`, `_guard`, `guard_anchors`, `evidence_for`, provenance text), `services/edge/src/edge/promote.py` (`_guard_metrics`, `resolve_guard`, `_guard_contract`), `scripts/demo_round.py`, `services/evaluation/configs/guard_thresholds.yaml`
- Test: `tests/integration/test_live_evaluation.py`, `tests/integration/test_demo_ui_live.py`, `services/edge/tests/test_promote.py`

**Interfaces:**
- Consumes: Task 4.1's `load_guard_set`, `guard_metrics`, `paired_noise`; `edge.completion_eval.evaluate_completion(model, tokenizer, examples, perplexity)` (already used for the in-project score).
- Produces, in the cached per-version score: `"guard": {"benchmark","edit_similarity","exact_match","n_examples"}` and `"guard_rows": [per-example edit similarity]`; the keys `pass_at_1`, `humaneval_solved`, `humaneval_n` are gone.
- Produces, in the evidence document for a candidate: `"guard": [<candidate guard>]`, `"baseline_guard": [<baseline guard>]`, `"guard_noise": paired_noise(baseline rows, candidate rows)`, `"guard_examples_sha256"`.
- `edge.promote.resolve_guard` returns the guard from a line-guard evidence file, or `None` with a reason string when there is none (same "never fabricate" rule it has today).

- [ ] **Step 1: Failing tests.** In `test_live_evaluation.py`: with the fake model fixture already used there, `score_version` returns a `guard` whose `benchmark == "LineCompletion"` and `n_examples == 300`, and no `pass_at_1` key; `evidence_document` carries `guard_noise` as a float. In `test_demo_ui_live.py`: the body the panel posts to `/promote` contains a `LineCompletion` guard for candidate and baseline. In `test_promote.py`: `resolve_guard` on a line-guard evidence file returns its guard; on a missing file returns `(None, reason)`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** In `score_version` the HumanEval block (generation, sandbox scoring, anchor) is replaced by one `evaluate_completion` call on the guard set. The cache key gains the guard fingerprint so old cached scores are not reused. `guard_thresholds.yaml`: replace the `humaneval_guard` block with `line_guard: {tolerance_edit_similarity: 0.02, size: 300, seed: 0, file: datasets/guard/line_guard_v1.jsonl}`.
- [ ] **Step 4:** Run `tests/integration`, `services/edge/tests`, `services/evaluation/tests` → PASS. The HumanEval harness under `evaluation/eval_harness/humaneval` and its tests are **not** removed: it stays as a reportable number.
- [ ] **Step 5: Commit** `evaluation: score candidates on the line-completion guard`.

### Task 4.3: The registry's D5 rule uses the new guard

**Files:**
- Modify: `services/registry/src/registry/promotion.py` (module docstring, constant, `_guard_holds`), `services/registry/src/registry/app.py` (`_parse_promote_body` error text)
- Test: `services/registry/tests/test_promotion.py`, `services/registry/tests/test_promote_api.py`, `tests/integration/test_four_seams.py`, `tests/e2e/test_full_loop.py`

**Interfaces (produced):**
- `promotion.GUARD_BENCHMARK = "LineCompletion"`, `promotion.GUARD_EDIT_SIMILARITY_TOLERANCE = 0.02`. `GUARD_PASS_AT_1_TOLERANCE` is deleted.
- `_guard_holds`: picks the `LineCompletion` entry on both sides; fails with `"LineCompletion guard metrics missing on candidate or baseline (contracts >= 1.2.0 required)"` if either is absent or has `edit_similarity is None`; otherwise `drop = baseline.edit_similarity - candidate.edit_similarity`, holds iff `drop <= 0.02`.

- [ ] **Step 1: Failing tests** (rewrite the HumanEval cases in `test_promotion.py` one-for-one):

```python
def test_guard_holds_at_exactly_two_points():      # baseline 0.60, candidate 0.58 → PROMOTE
def test_guard_fails_just_past_two_points():       # baseline 0.60, candidate 0.5799 → not promoted
def test_guard_improving_always_holds():           # candidate above baseline
def test_humaneval_only_body_is_refused_with_the_contract_hint():   # Review Focus 3
    # eval.guard = [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.3}}] → reason contains
    # "LineCompletion guard metrics missing" and "1.2.0"
```

In `test_promote_api.py`: the HumanEval-only body gets **422** with that text and `GET /promotions` shows no new decision.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** In `app._promote`, check the guard's presence right after `_parse_promote_body` and raise `HTTPException(422, ...)` before `decide` is called, so nothing is recorded.
- [ ] **Step 4:** Update the guard dictionaries in `test_four_seams.py` and `test_full_loop.py`. Run registry, integration and e2e tests → PASS.
- [ ] **Step 5: Commit** `registry: D5's guard is line-completion edit similarity, 2 points`.

### Task 4.4: Measure the guard's noise on the real model; PY150 as a reported number

**Files:**
- Create: `scripts/eval_py150.py`, `services/evaluation/results/line_guard/noise.json`
- Modify: `docs/decisions/2026-11-14-amended-decisions.md` (D5 section), `docs/architecture.md` §3 items 4–5, `services/evaluation/docs/` guard page if one exists

- [ ] **Step 1 [USER, GPU]:** With registry and cluster running and at least two published versions of `cluster-web`, run `python scripts/evaluate_candidate.py --cluster web`. Record wall time and the `guard_noise` value in `results/line_guard/noise.json` with the two version shas.
- [ ] **Step 2: Gate.** If `guard_noise > 0.02`, **stop and ask the user** (U12). Do not change the tolerance or the set size on your own.
- [ ] **Step 3: PY150.** Before writing code, confirm the licence and record format of CodeXGLUE's line-level Python completion set from its official repository. If the licence does not clearly allow this use, stop and ask the user. Otherwise write `scripts/eval_py150.py`: `--data <local path>` (downloaded once by the user, never fetched by the script), a seeded 300-example subset, the same `evaluate_completion` scoring, output to `services/evaluation/results/py150/<sha>.json`. It never feeds `/promote`.
- [ ] **Step 4:** Add the D5 section to the amended-decisions file: guard = line completion on a held-out general set, tolerance 0.02, measured noise from Step 1, HumanEval and PY150 reported only; and the reason (HumanEval tests docstring-to-function synthesis on 20 problems, where one problem is 5 points).
- [ ] **Step 5: Commit** `evaluation: guard noise measurement and a reported PY150 score`.

---

# Piece 5: candidate flow (branch `plan/05-candidate-flow`)

Behaviour after this piece (U10, U11, C6, C7): the cluster's aggregate produces a **candidate**; the cluster keeps serving the previously promoted adapter; publish stores the candidate in the registry **inactive**; `/promote` either makes that version active (PROMOTE) or records REJECT and leaves the pointer alone; the demo panel then tells the cluster to promote or drop its candidate.

### Task 5.1: Registry

**Files:**
- Modify: `services/registry/src/registry/app.py` (`_parse_save_meta` default, `_promote`, `promote` docstring), `services/registry/src/registry/promotion.py` (`decide`, docstring), `services/registry/src/registry/storage.py` (`save` default), `services/registry/src/registry/demo.py`
- Test: `services/registry/tests/test_promotion.py`, `test_promote_api.py`, `test_api.py`, `test_demo.py`, `test_restore.py`

**Interfaces (produced):**
- `RegistryStore.save(..., set_active: bool = False)` and the upload API's `meta.set_active` default `False`. An explicit `true` is still honoured (operator use).
- `decide(eval_result, *, baseline_guard, active_version_before: int | None) -> PromotionDecision`. The `previous_version` parameter is removed. PROMOTE → `active_version_after = eval_result.adapter.version`. Otherwise `action = PromotionAction.REJECT`, `active_version_after = active_version_before` if not `None`, else `0` (meaning "nothing active"), reason starts `"rejected: "`.
- `/promote` rules: the evaluated version must exist (404) and must be **newer than the active version** (409 `"v{n} is not a candidate: active is v{m}"`); a version that already has a recorded decision gets 409 `"v{n} was already decided"`. PROMOTE → `store.set_active(name, version)`, then record. REJECT → record only. The `composite` block still builds on PROMOTE.
- `/restore` and the cascade are unchanged (operator pointer moves).

- [ ] **Step 1: Failing tests:**

```python
def test_upload_does_not_activate(client):                    # POST versions → GET /active is 404
def test_promote_makes_the_candidate_active(client):          # v1 promoted and active; publish v2; /promote good eval → active == 2
def test_reject_leaves_the_pointer_and_records_reject(client):# bad eval → active stays 1; decisions[-1]["action"] == "reject"
def test_first_candidate_rejected_leaves_nothing_active(client):   # Review Focus 5 → GET /active 404, decision active_version_after == 0
def test_first_candidate_can_be_promoted_with_no_previous_version(client):
def test_promoting_the_active_or_an_older_version_is_409(client)
def test_deciding_a_version_twice_is_409(client)
def test_decide_reject_is_pure_and_never_raises_for_missing_previous()
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** Rewrite `registry.demo` (the compose `demo-seed` walkthrough) to the new sequence: upload → promote → upload → reject → restore drill.
- [ ] **Step 4:** Fix the existing tests that relied on auto-activation by adding an explicit `"set_active": true` to their upload meta **only where the test is not about promotion**; tests about promotion are rewritten to the new flow. Run `pytest services/registry/tests -q` → PASS with coverage ≥ 90 % (`--cov=registry --cov-fail-under=90`).
- [ ] **Step 5: Commit** `registry: uploads are candidates; only a promote makes a version active`.

### Task 5.2: Cluster

**Files:**
- Modify: `services/cluster/src/cluster/server.py` (`_ClusterState`, `_reset_clusters`, `_aggregate_locked`, `publish_to_registry`, `list_clusters`, `_check_base`; new routes), `services/cluster/src/cluster/schemas/messages.py` if a request model is added there
- Test: `services/cluster/tests/test_candidate_flow.py` (new), existing `test_fedprox_server.py`

**Interfaces (produced):**
- `_ClusterState.candidate: ClusterAdapterBroadcast | None`. `_aggregate_locked` stores its result in `candidate`; `active` is untouched. `last_round` and `last_manifest` are still set at aggregate time.
- Aggregate while `candidate is not None` → **409** `{"error": "candidate_pending", "sha256": <candidate sha>}`; buffered uploads are kept (Review Focus 4).
- `GET /clusters/{id}/adapters/candidate` → the candidate broadcast, 404 if none.
- `POST /clusters/{id}/adapters/candidate/promote` body `{"sha256": str}` → 409 if it is not the pending candidate's sha; otherwise `active = candidate`, `candidate = None`, returns `{"active_sha256", "round_id"}`.
- `POST /clusters/{id}/adapters/candidate/reject` body `{"sha256": str}` → same check; `candidate = None`; uploads buffered for the next round are kept; returns `{"active_sha256": <unchanged or null>}`.
- `publish_to_registry` publishes the **candidate** (404 if none) with `"set_active": False`; `PublishRequest.set_active` default becomes `False`.
- `GET /clusters` adds `candidate_sha256` per cluster next to `active_sha256`.
- `GET .../adapters/active`, `/download` and upload base checks (`_check_base`) keep using `active`, so during evaluation edges still see, and must train on, the previous adapter. The single-cluster legacy routes (`/aggregate`, `/adapters/cluster/active`) go through the same state and follow the same rules.
- A rejected round keeps its round number; the next aggregate uses the next number.

- [ ] **Step 1: Failing tests** in `test_candidate_flow.py` (use the upload helpers from `test_fedprox_server.py`):

```python
def test_aggregate_creates_a_candidate_and_active_is_unchanged()
def test_active_is_404_until_the_first_candidate_is_promoted()            # Review Focus 5
def test_promote_makes_the_candidate_active_and_clears_it()
def test_reject_drops_the_candidate_and_keeps_active()
def test_promote_or_reject_with_the_wrong_sha_is_409()
def test_second_aggregate_while_a_candidate_is_pending_is_409_and_keeps_uploads()   # Review Focus 4
def test_round_two_uploads_must_declare_the_promoted_adapter_not_the_candidate()
def test_after_a_first_round_reject_uploads_without_a_base_are_accepted_again()     # Review Focus 5
def test_publish_sends_the_candidate_with_set_active_false()              # capture the outbound meta
def test_clusters_listing_reports_candidate_sha256()
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Update existing cluster tests that read `/adapters/active` straight after `/aggregate` to promote the candidate first. Run `pytest services/cluster/tests -q` → PASS.
- [ ] **Step 5: Commit** `cluster: an aggregate is a candidate until it is promoted`.

### Task 5.3: Demo panel, evaluation script, edge page, round script

**Files:**
- Modify: `demo_ui/server.py` (`live_state`, `_evaluated`, `live_evaluate`, `live_promote`), `demo_ui/static/live.html`, `demo_ui/static/common.js`, `scripts/evaluate_candidate.py` (`pick_versions`, `score_version`), `scripts/demo_round.py`, `services/edge/src/edge/promote.py` (`_previous_version`, `predict_decision`, `promote_candidate`), `services/edge/src/edge/webui.py`, `services/edge/src/edge/webui_page.html`, `docs/multi-laptop-demo.md`, `demo_ui/README.md`
- Test: `tests/integration/test_demo_ui_live.py`, `test_live_evaluation.py`, `test_four_seams.py`, `tests/e2e/test_full_loop.py`, `services/edge/tests/test_promote.py`, `test_webui.py`

**Interfaces:**
- Consumes: Task 5.1's promote rules; Task 5.2's candidate routes and `candidate_sha256`.
- `pick_versions(listing, active_version: int | None) -> tuple[dict, dict | None]`: candidate = newest version above the active one; baseline = the active version's metadata, or `None` when nothing is active.
- `score_version(..., meta=None)` scores the **bare base model** (no adapter loaded) and caches it under the key `"base"` (C7).
- `live_promote`: posts to the registry; on `promote` calls the cluster's `candidate/promote` with the registry's sha256 for that version; on `reject` calls `candidate/reject`. The call to `/clusters/{id}/adapters/active/restore` on ROLLBACK is removed from this path (the restore route itself stays for operator use).
- Panel wording: the step is "Promoted" or "Rejected"; "Rollback" appears only on the operator restore control.
- Edge page tracker: step 4 reads "Awaiting evaluation" while the cluster reports a `candidate_sha256`, "Merged and promoted" when `active_sha256` changes, "Rejected, previous adapter kept" when the candidate disappears and `active_sha256` did not change. Auto-pull still triggers only on a changed `active_sha256`.

- [ ] **Step 1: Failing tests.** `test_demo_ui_live.py`: (a) promote path → the fake cluster receives `candidate/promote` with the registry's sha; (b) reject path → `candidate/reject`, and no call to `active/restore`; (c) `live_state` exposes `candidate_sha256`. `test_live_evaluation.py`: `pick_versions` with `active_version=None` returns `(candidate, None)`; with active 2 and versions 1–3 returns `(v3, v2)`. `test_webui.py`: the three tracker states above from fake `/clusters` responses. `test_four_seams.py` and `test_full_loop.py`: the round is upload → aggregate → publish (inactive) → evaluate → promote → cluster promote → pull.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run the full suite and ruff (Global Constraints) → PASS.
- [ ] **Step 5 [USER, GPU + two processes]:** One-laptop live check: `python scripts/run_services.py --ui`, train and upload one web client from the edge page, aggregate, publish, evaluate, promote; confirm the edge page pulls only after Promote. Repeat with a candidate that is rejected (operator can force it by evaluating a deliberately bad upload) and confirm the edge keeps its adapter.
- [ ] **Step 6:** Update `docs/architecture.md` §3 item 5 and the amended-decisions D5 section: "a published version is a candidate; promotion makes it live; a rejected candidate is never served". Update the runbook steps in `docs/multi-laptop-demo.md`.
- [ ] **Step 7: Commit** `demo, edge: follow the candidate through evaluation to promote or reject`.

---

# Piece 6: SQLite metadata store (branch `plan/06-metadata-db`)

Files on disk after this piece: `adapters/<name>/vN/adapter.safetensors` and `metadata.json` (kept as a self-describing sidecar, never read on the request path), plus `registry.db`. The `active` files and `promotions.jsonl` are read once by the migration and then no longer written.

### Task 6.1: Schema and engine

**Files:**
- Create: `services/registry/src/registry/db.py`
- Modify: `services/registry/pyproject.toml` (dependency `sqlalchemy>=2.0,<3`), `services/registry/Dockerfile` if it pins installs
- Test: `services/registry/tests/test_db.py`

**Interfaces (produced):**
- `db.DB_ENV = "CLASP_REGISTRY_DB"`; default URL `sqlite:///<CLASP_REGISTRY_DATA>/registry.db`.
- `db.make_engine(url: str) -> sqlalchemy.Engine`. For SQLite: on every connection `PRAGMA journal_mode=WAL`, `PRAGMA synchronous=FULL`, `PRAGMA foreign_keys=ON`.
- `db.metadata: sqlalchemy.MetaData` holding these tables (Plan B adds more to the same object), created by `db.create_all(engine)`:
  - `adapters(id PK, name UNIQUE NOT NULL, kind NOT NULL)`
  - `versions(id PK, adapter_id FK, version INT NOT NULL, sha256 NOT NULL, num_bytes INT, created_at, metadata_json TEXT NOT NULL, UNIQUE(adapter_id, version))`
  - `active_pointer(adapter_id PK FK, version_id FK NOT NULL, moved_at)`
  - `composite_parts(composite_version_id FK, part_version_id FK, role CHECK IN ('cluster','client'), weight REAL, PRIMARY KEY(composite_version_id, role))`, index on `part_version_id`
  - `promotion_decisions(id PK, adapter_id FK, version INT NOT NULL, action NOT NULL, active_version_after INT NOT NULL, reason, decided_at, decision_json TEXT NOT NULL)`, index on `(adapter_id, id)`
  - `cluster_sync(id PK, decision_id FK UNIQUE, cluster_id NOT NULL, sha256 NOT NULL, action CHECK IN ('promote','reject'), status CHECK IN ('pending','done') DEFAULT 'pending', attempts INT DEFAULT 0, last_error, updated_at)`

- [ ] **Step 1: Failing tests:** `create_all` on a temp file creates the six tables; a second `create_all` is a no-op; the three pragmas read back as `wal`, `2` (FULL), `1`; inserting a `versions` row with an unknown `adapter_id` raises `IntegrityError`; a duplicate `(adapter_id, version)` raises `IntegrityError`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `registry: SQLite schema for versions, pointers, decisions and cluster sync`.

### Task 6.2: Repository

**Files:**
- Create: `services/registry/src/registry/repository.py`
- Test: `services/registry/tests/test_repository.py`

**Interfaces (produced):**

```python
class MetadataRepository(Protocol):
    def transaction(self) -> ContextManager[None]: ...          # re-entrant; commit on exit, rollback on exception
    def list_adapters(self) -> list[str]: ...
    def list_versions(self, name: str) -> list[int]: ...
    def get_metadata(self, name: str, version: int) -> AdapterMetadata: ...   # raises storage.AdapterNotFound
    def add_version(self, meta: AdapterMetadata) -> None: ...   # raises storage.VersionExists, storage.KindMismatch
    def delete_version(self, name: str, version: int) -> None: ...
    def get_active(self, name: str) -> int | None: ...
    def set_active(self, name: str, version: int) -> None: ...
    def record_decision(self, name: str, decision: PromotionDecision) -> int: ...  # returns decision id
    def list_decisions(self, name: str) -> list[PromotionDecision]: ...
    def decided_versions(self, name: str) -> set[int]: ...
    def referenced_versions(self, name: str) -> set[int]: ...   # versions that are a part of some composite
    def composites_built_from(self, name: str, version: int) -> list[AdapterRef]: ...
    def add_cluster_sync(self, decision_id: int, cluster_id: str, sha256: str, action: str) -> int: ...
    def pending_cluster_sync(self) -> list[dict]: ...
    def mark_cluster_sync(self, sync_id: int, *, done: bool, error: str | None = None) -> None: ...

class SqlMetadataRepository:        # the only implementation; takes a sqlalchemy.Engine
    def __init__(self, engine: Engine) -> None: ...
```

`add_version` also writes `composite_parts` rows when `meta.composed_from` is set. Outside an explicit `transaction()` each call is its own transaction.

- [ ] **Step 1: Failing tests,** one per method, plus:

```python
def test_exception_inside_transaction_rolls_everything_back(repo):
    with pytest.raises(RuntimeError):
        with repo.transaction():
            repo.add_version(_meta("a", 1)); repo.set_active("a", 1)
            repo.record_decision("a", _decision("a", 1)); raise RuntimeError
    assert repo.list_versions("a") == [] and repo.get_active("a") is None and repo.list_decisions("a") == []

def test_referenced_versions_is_one_indexed_query(repo, sql_log): ...   # exactly one SELECT, touching composite_parts
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement with SQLAlchemy Core (`engine.begin()`, no ORM). **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `registry: metadata repository over SQLAlchemy`.

### Task 6.3: The store reads and writes metadata through the repository; old volumes migrate

**Files:**
- Create: `services/registry/src/registry/migrate.py`
- Modify: `services/registry/src/registry/storage.py` (`RegistryStore.__init__`, every metadata/pointer/log method), `services/registry/src/registry/app.py` (`get_store`), `services/registry/src/registry/retention.py`, `cascade.py`, `lineage.py`, `composite.py` (replace directory scans with repository calls), `services/registry/tests/conftest.py`
- Test: `services/registry/tests/test_migrate.py`, existing suite

**Interfaces:**
- `RegistryStore(root, repo: MetadataRepository | None = None)`; when `repo` is `None` it builds a `SqlMetadataRepository` from `db.make_engine(...)` and calls `db.create_all`. Public method names and exceptions of `RegistryStore` do not change; `get_store()` stays the FastAPI dependency (singleton).
- `save()` order: stage and fsync the files → publishing rename → fsync directories (all as today) → `repo.add_version(meta)` (+ `repo.set_active` when asked) in one transaction. If the repository call fails, the version directory is removed and the error re-raised.
- `migrate.import_volume(store_root: Path, repo: MetadataRepository) -> dict` → `{"adapters": n, "versions": n, "pointers": n, "decisions": n}`. Runs at `RegistryStore` construction when the database has no adapters: imports every `vN/metadata.json`, every `active` file and every `promotions.jsonl` line in one transaction; idempotent.

- [ ] **Step 1: Failing tests** in `test_migrate.py`:

```python
def test_volume_written_by_the_old_registry_reads_back_identically(tmp_path):   # Review Focus 1
    # build a volume with the pre-database code path captured as a fixture directory:
    # 2 adapters, 3 versions, one active file each, a 2-line promotions.jsonl, one composite
    # open RegistryStore on it (no registry.db present) and assert list_adapters, list_versions,
    # get_metadata, get_active, list_promotions and lineage equal the fixture's expected values

def test_import_is_idempotent_and_skipped_when_the_db_has_data(tmp_path)
def test_corrupt_metadata_json_fails_the_import_and_leaves_the_db_empty(tmp_path)
```

Commit the fixture volume under `services/registry/tests/fixtures/legacy_volume/` (generate it with the code at `integration/demo` **before** changing `storage.py`; tiny synthetic safetensors).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** `retention` uses `repo.referenced_versions`; `cascade` uses `repo.composites_built_from`.
- [ ] **Step 4:** Run the registry suite. Tests that read the `active` file or `promotions.jsonl` directly are changed to use the store's methods; no API-level test should need a change. Coverage ≥ 90 %.
- [ ] **Step 5: Commit** `registry: metadata lives in SQLite; existing volumes are imported on first start`.

### Task 6.4: Promotion in one transaction, and the orphan sweep

**Files:**
- Modify: `services/registry/src/registry/app.py` (`_promote`, `_restore`, `_gc_one`, `_gc_all`, startup), `services/registry/src/registry/composite.py` (`store_composite` split into write-files and register), `services/registry/src/registry/retention.py`
- Test: `services/registry/tests/test_promote_atomic.py` (new)

**Interfaces (produced):**
- `composite.write_composite_files(store, name, plan) -> AdapterMetadata` (files only, fsynced, no database row) and `composite.register_composite(repo, meta) -> None`.
- `_promote` order: validate → decide → write composite files if requested → **one** `with store.repo.transaction():` containing `register_composite` and `set_active` (PROMOTE only), `record_decision`, and `add_cluster_sync` with action `promote` or `reject` (both outcomes, when the adapter's metadata has a `cluster_id`: the cluster must either switch to its candidate or drop it) → return. `_restore` wraps its pointer move, decision and cascade moves in one transaction the same way.
- `retention.sweep_orphans(store) -> list[str]`: deletes version directories that have no `versions` row, and `.tmp-v*` / `.trash-*` directories older than 60 seconds; returns the removed paths. Called at application startup and at the start of every GC.

- [ ] **Step 1: Failing tests:**

```python
@pytest.mark.parametrize("fail_at", ["set_active", "record_decision", "add_cluster_sync"])
def test_failure_inside_the_transaction_leaves_no_trace(client, store, monkeypatch, fail_at):
    # monkeypatch the repository method to raise; POST /promote → 500
    # then: active unchanged, no new decision, no composite row, and after sweep_orphans no composite directory

def test_failure_after_files_before_commit_is_cleaned_by_the_sweep(store, monkeypatch)
def test_sweep_never_removes_a_registered_version_or_a_fresh_staging_dir(store)
def test_restore_moves_pointer_decision_and_cascade_together(client, store, monkeypatch)
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run registry suite → PASS, coverage ≥ 90 %.
- [ ] **Step 5: Commit** `registry: a promotion commits as one transaction; orphaned files are swept`.

### Task 6.5: Cluster sync reconciliation

**Files:**
- Modify: `services/registry/src/registry/app.py` (two routes), `demo_ui/server.py` (`live_promote`, new `_reconcile_cluster_sync`, call it from `live_state`), `docs/registry_api.md`
- Test: `services/registry/tests/test_cluster_sync.py`, `tests/integration/test_demo_ui_live.py`

**Interfaces (produced):**
- `GET /sync/pending` → `{"pending": [{"id", "adapter", "version", "cluster_id", "sha256", "action", "attempts"}]}`.
- `POST /sync/{id}/done` body `{"ok": bool, "error": str | None}` → marks done, or increments `attempts` and stores `last_error`.
- Demo panel: `live_promote` no longer calls the cluster directly; it calls `_reconcile_cluster_sync()`, which for each pending row calls the cluster's `candidate/promote` or `candidate/reject` and reports back. A cluster 409 saying the sha is not the pending candidate **and** `active_sha256` already equals the row's sha (promote) or no candidate is pending (reject) counts as done. `live_state` calls it too, so a restarted panel finishes the job.

- [ ] **Step 1: Failing tests.** Registry: a PROMOTE of a cluster adapter creates one pending row; `done` with `ok: true` empties the list; `ok: false` keeps it with `attempts == 1`. Integration (Review Focus 2): registry PROMOTE succeeds while the fake cluster is unreachable → the row stays pending; make the cluster reachable and call `GET /api/live/state` → the cluster received `candidate/promote` exactly once and the row is done; calling state again sends nothing.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Full suite and ruff → PASS.
- [ ] **Step 5:** Update `docs/architecture.md` §4 (storage layout, "metadata in SQLite, weights on disk", the transaction and the sweep) and add a short "Data model" section listing the six tables and their indices.
- [ ] **Step 6: Commit** `registry, demo: the cluster's promote or reject is retried until it lands`.
- [ ] **Step 7 [USER, Docker Desktop running]:** `docker compose --profile demo up --build`; confirm `registry.db` appears in the `registry-data` volume and the demo-seed walkthrough report shows upload → promote → upload → reject → restore.
