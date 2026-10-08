# Week 6 — Panel Review 1 fixes (P2 Cluster)

## Original panel feedback: **not available**

The plan (Daily Targets v2, Week 6 Thu) says "Log panel feedback -> debt list" and Fri
"Fix top P2 item". No panel feedback or debt list exists anywhere in the repository:

* `docs/rubric.md` is the unfilled template (every field is still a `<<...>>` placeholder),
* `docs/review/` does not exist,
* a search of `docs/`, `README.md`, `contracts/`, `security/`, `services/*` for
  `TODO | FIXME | panel review | tech debt | feedback` (in `.py`, `.md`, `.yml`, `.toml` files) found only that template,
* `services/cluster/WEEK5_STATUS.md` is a Week 5 status note, not panel feedback.

**No feedback was invented.** Instead this is a code-review pass over the P2 code, fixing what is
clearly wrong or misleading inside P2's scope, each with a test (`tests/test_week6_review.py`
and the files named below). If the real panel comments turn up, they should be triaged on
top of this list — nothing here claims to answer them.

## Findings and fixes

| # | Finding | Fix | Test |
|---|---|---|---|
| 1 | `DEFAULT_ALPHA = 32.0` but Edge pins `lora_alpha = 16` (`services/edge/src/edge/lora_init.py`; Edge's `attach_lora` default is also 16). An adapter built with Cluster defaults produced a PEFT config Edge's compatibility check would reject; the old test only *documented* the mismatch. | `DEFAULT_ALPHA = 16.0` (scaling alpha/rank = 1.0). The toy simulation's learning rate went 0.05 -> 0.1 because the effective update rate scales with (alpha/rank)^2: with the same lr, a mislabelled client was recovered in only 4 of 8 seeds, with lr 0.1 in 8 of 8 (measured). | `test_peft_interop.py::test_default_alpha_now_matches_edge_contract`; `test_to_peft_config_reports_a_non_default_alpha_honestly` keeps the old honesty property |
| 2 | Aggregation only checked `target_modules` / `num_layers` across clients. Adapters with different `alpha` or `rank` (different LoRA scaling) were silently averaged. | `aggregation.ensure_compatible`: rank, alpha, modules, layers must agree, in `aggregate_svd`, `aggregate_naive` and `exact_average_delta`. | `test_mixed_alpha_is_rejected`, `test_mixed_rank_is_rejected` |
| 3 | `aggregate_svd(rank=r')` with `r' != r` copied `alpha`, silently rescaling the effective update (alpha/r) by `r/r'`. | alpha scales with the output rank (no-op when `r' == r`). | `test_changing_the_output_rank_preserves_the_effective_update` |
| 4 | `truncated_svd_refactor` accepted rank 0, negative, or larger than `min(out, in)` and returned wrongly shaped factors. | `ValueError` with the allowed range; HTTP `/aggregate` maps it to 422 and keeps the buffer. | `test_truncated_svd_rank_bounds`, `test_http_impossible_rank_is_a_422_and_keeps_the_buffer` |
| 5 | `exact_average_delta` did not validate its adapters (non-finite values passed straight to the accumulator) or the `layer` index. | validates adapters and `0 <= layer < num_layers`. | `test_exact_average_delta_validates_inputs_and_layer` |
| 6 | The HTTP service accepted an upload whose structure differed from the round's other uploads and only failed later, at `/aggregate`, for everybody. | rejected at upload time (422), only the offender is turned away. | `test_http_upload_with_mismatching_alpha_is_rejected_at_intake` |
| 7 | `to_peft_config` hard-coded the 1.3B dev model name, and its docstring described the alpha mismatch as an open cross-team issue. | `base_model_name_or_path` is a parameter (default = Edge's dev profile model `deepseek-ai/deepseek-coder-1.3b-base`); docstring updated. | `test_base_model_name_is_configurable` |
| 8 | `demo_all_weeks.py` cited source line numbers that no longer matched (`client.py line 51-56`, ...), printed `alpha=32`, claimed "All services (P1/P2/P3/P4/P5) use these same constants", and claimed "Without FedProx: clients diverge ... bad aggregation" without showing it. | cites symbols instead of lines, alpha text corrected, claim about other services removed, FedProx claim replaced with a pointer to the measured evidence. | `test_demos_smoke.py::test_demo_all_weeks_runs_end_to_end` |
| 9 | No test ran either demo script. | smoke tests for `demo_all_weeks` and the new `demo_phase2`. | `tests/test_demos_smoke.py` |
| 10 | Junk next to the package: `pytest_out.txt` (0 bytes), `.coverage*`. | not deleted (no destructive cleanup); a package-local `.gitignore` keeps them out of commits. | — |

## Reported, not changed (other teams' code)

* `services/edge/src/edge/lora_init.py` line 1: `from peft.tuners.lora.corda import target_modules`.
  `target_modules` is never used from that import (the function parameter shadows it); it looks like an
  accidental editor auto-import of a PEFT-internal name and may fail on some PEFT versions
  (not verified here — PEFT is not installed in the verification environment). P1 should remove it.
