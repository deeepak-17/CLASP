# Panel Review Rubric — CLASP

> The `latex-doc-generator` reads this file first. If it is missing or has unfilled
> `<<...>>` placeholders, the generator will ask you for the criteria interactively
> before generating. Fill in your panel's actual rubric below so reviews are reproducible.

**Review:** 23CSE498 — Project Phase 2, Panel Review 1
**Date:** 20–24 July 2026 (Phase II W6)
**Total marks:** 5

> Marks below mirror the standard Phase-2 panel sheet (1 mark per criterion).
> Confirm the official weighting with Dr. Swapna T R before the panel and update
> this row if the panel uses a different split.

## Criteria

| # | Criterion | Weight / Marks | What the panel looks for | Maps to (report §, deck frame) |
|---|-----------|----------------|--------------------------|--------------------------------|
| 1 | Well-defined Requirements & Feasibility | 1 | Clear SRS; functional + non-functional requirements (measurable NFRs, D11); single-GPU feasibility justified (D8) | Report §1; Deck "Requirements & Feasibility" |
| 2 | Architecture Validation | 1 | Design backed by literature; paper-to-architecture mapping; locked decisions D1–D12 traceable | Report §2.3, §5; Deck "Architecture Validation" |
| 3 | Comprehensive Architecture & System Flow | 1 | End-to-end design; module boundaries + ownership; the three contract seams; data flow of one federated round | Report §5; Deck "Comprehensive Architecture" (`docs/architecture.md`) |
| 4 | Process Audit (CI/CD, Dockerization) | 1 | Reproducible build/deploy; CI ruff+pytest 6-package matrix; monorepo + compose with stateful registry volume; testing gates | Report §6; Deck "Process Audit" (`.github/workflows/ci.yml`, `docker-compose.yml`) |
| 5 | Works Completed & Plan | 1 | Each module demoable in isolation; contracts v1.0 frozen; registry save/load/list live with tests; credible W7–W16 plan | Report §6, §7; Deck summary |

## Notes for the generator
- Tag each deck content frame with its criterion using the `\rubric{...}` macro.
- For every criterion, cite concrete repo evidence (file paths, test results, experiment
  outputs) rather than asserting completion.
- Flag any criterion with no supporting evidence in the repo as a **gap** in the run report,
  so the team knows what to close before the panel.
