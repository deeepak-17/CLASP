# Panel Review Rubric — CLASP

> The `latex-doc-generator` reads this file first. If it is missing or has unfilled
> `<<...>>` placeholders, the generator will ask you for the criteria interactively
> before generating. Fill in your panel's actual rubric below so reviews are reproducible.

**Review:** <<e.g. 23CSE498 — Project Phase 2, Panel Review 1>>
**Date:** <<DD Month YYYY>>
**Total marks:** <<e.g. 5>>

## Criteria

| # | Criterion | Weight / Marks | What the panel looks for | Maps to (report §, deck frame) |
|---|-----------|----------------|--------------------------|--------------------------------|
| 1 | <<e.g. Well-defined Requirements & Feasibility>> | <<1>> | <<clear SRS, functional/non-functional, feasibility justified>> | Report §1; Deck "Requirements & Feasibility" |
| 2 | <<e.g. Architecture Validation>> | <<1>> | <<design backed by literature, paper-to-architecture mapping>> | Report §2.3, §5; Deck "Architecture Validation" |
| 3 | <<e.g. Comprehensive Architecture & System Flow>> | <<1>> | <<end-to-end design, module boundaries, data flow>> | Report §5; Deck "Comprehensive Architecture" |
| 4 | <<e.g. Process Audit (CI/CD, Dockerization)>> | <<1>> | <<reproducible build/deploy plan, testing gates>> | Report §6; Deck "Process Audit" |
| 5 | <<e.g. Works Completed & Plan>> | <<1>> | <<demonstrable progress, credible next-semester plan>> | Report §6, §7; Deck summary |

## Notes for the generator
- Tag each deck content frame with its criterion using the `\rubric{...}` macro.
- For every criterion, cite concrete repo evidence (file paths, test results, experiment
  outputs) rather than asserting completion.
- Flag any criterion with no supporting evidence in the repo as a **gap** in the run report,
  so the team knows what to close before the panel.
