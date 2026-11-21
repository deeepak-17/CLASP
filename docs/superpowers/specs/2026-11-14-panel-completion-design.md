# CLASP panel-completion design (2026-11-14)

Record of the decisions Adithyaa made on 2026-11-13/14. The two plans in
`docs/superpowers/plans/2026-11-14-panel-completion-*.md` implement exactly this.
Background and evidence: `registry-design-discussion-2026-10-25-reviewed.md` (bundle folder,
outside the repo) and the repository audit of 2026-11-13 (1,363 tests passing at `783a2de`).

## Intent

Make what the panel sees match what the documents claim. Where the build and the documents
disagree, either the build changes or the document does; each case is decided below.

## Decisions made by the user

| # | Topic | Decision |
|---|---|---|
| U1 | D6 composites | Keep merging on the edge. Rewrite D6, the diagram and the README. The registry's composite code stays, tested and unused, and is described as such. |
| U2 | DP unit | Reword to "example-level DP applied on each client". No mechanism change. |
| U3 | ε budget | ε ≤ 8 (δ = 1e-5) covers the **whole federated run of 2 rounds**, enforced: the edge refuses, the cluster refuses, the registry records. |
| U4 | Security library | Wire in the audit logger, rate limiter, security headers and request validator. Use HMAC twice: on evaluation results sent to `/promote`, and on edge uploads to the cluster. Use JWT for human logins on the demo panel and the edge page. API keys stay in the library, documented as unused. |
| U5 | Identity | People log in with JWT; services identify each other by mTLS certificate. |
| U6 | mTLS default | Compose and the demo scripts turn mTLS on. A service started by hand without certificates still runs plain HTTP with a warning. |
| U7 | D4 re-clustering | A "Re-cluster" button on the demo panel after round 2 runs the **dry run** and shows the proposed grouping. Nothing moves. D4 is reworded: computed after the final round and reported; applying it starts a new run with a new budget. |
| U8 | Who builds | One plan, built in this checkout, across all modules, on branches off `integration/demo`. |
| U9 | Metadata store | SQLite, behind an interface so PostgreSQL is a configuration change. |
| U10 | Activation | Publishing creates an **inactive candidate**. Only PROMOTE makes it live. |
| U11 | Cluster serving | The cluster keeps serving the previous adapter until PROMOTE; a rejected merge is dropped. |
| U12 | Guard | Replace the HumanEval guard with a line-completion guard. A local guard set built from click, jinja and colorama gates promotion; PY150 is run once per experiment and reported without gating. Threshold: guard edit similarity may drop by at most 2 points absolute (0.02). If measured noise is larger than 0.02, stop and ask. |
| U13 | Build shape | One branch and one pull request per piece, merged in order. |

## Calls made by Claude (stated to the user, not objected to, or made after "no more questions")

| # | Call | Reason |
|---|---|---|
| C1 | The three browser pages are served over HTTPS with a server certificate only; people are identified by JWT. | A browser cannot easily present a client certificate. |
| C2 | JWT users come from a JSON file per page (username, scrypt password hash, role) written by a script. No self-registration. | Smallest thing that gives a real login. |
| C3 | HMAC and JWT secrets come from environment variables or key files generated next to the development certificates. | Same distribution path the certificates already use. |
| C4 | The registry imports the security library and SQLAlchemy, so it is no longer standard-library-only. CI installs `security` for the registry job. | Required by U4 and U9. |
| C5 | SQLAlchemy 2.x Core is the database layer. | It is what makes "PostgreSQL is a configuration change" true, and gives a real pool. |
| C6 | A rejected candidate is recorded with a new `PromotionAction.REJECT`. | With U10 nothing is rolled back, so `ROLLBACK` would be a false record. Needs the contracts bump. |
| C7 | The first candidate of a cluster (nothing promoted yet) is compared against the bare base model. | With U10 there is no earlier active version to compare with. |
| C8 | The privacy budget is charged when an adapter is **uploaded**, not when it is trained. | DP cost is incurred by what is released. Rehearsal trainings that are never uploaded cost nothing. |
| C9 | Both rounds use the same noise level, calibrated once so that two rounds together stay within ε 8. | Simple, predictable, and the ledger verifies it by composing the real history. |
| C10 | Certificate names stay as `make_dev_certs.sh` issues them (`cluster`, `demo-ui`, `client-<repo>`); a new `clasp-admin` certificate is added. Roles are mapped from these names in configuration. | Renaming certificates would break every laptop's bundle for no gain. |
| C11 | When a secret is not configured, HMAC checks are skipped with a warning, and when no users file exists a page runs without login with a warning. | Same fall-back behaviour the user chose for mTLS in U6. |
| C12 | Remote access is documented with plain WireGuard; Headscale is named as the alternative. Documentation only. | Three laptops with fixed peers need no control server. |
| C13 | `contracts` goes 1.1.0 → 1.2.0 with optional fields only. | Additive; older payloads still parse. |

## Non-goals

- Client-level DP (rejected in U2).
- Per-developer composites in the registry (rejected in U1).
- Applying a re-clustering inside a run (rejected in U7).
- PostgreSQL, an API gateway, self-registration, API-key authentication.
- Any change to training hyperparameters other than the DP noise level that U3 forces.

## Things only the user can do

- Get all-hands sign-off for contracts 1.2.0 (team rule).
- Run the GPU steps that need the real model and a real cluster round.
- Start Docker Desktop for the compose verification steps.
- Push branches and open pull requests. Nothing in the plans pushes.

## Open risks the plans stop on rather than decide

- Guard noise measured above 0.02 (U12).
- With the noise needed for ε 8 over two rounds, the adapter may learn little. The plan
  measures and reports; it does not change the budget or the round count.
- PY150's licence and format are to be confirmed before it is used.
