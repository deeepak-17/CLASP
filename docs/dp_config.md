# CLASP DP-SGD Configuration (D7)

## Privacy Unit
- **Unit**: Per-sample (record-level). Each individual training example is the
  privacy unit. This means ε bounds the information leakage about any single
  training record.
- **Example-level, applied per client**: CLASP operates at example-level
  privacy, applied on each client: Opacus DP-SGD over 1024- or 512-token
  blocks, on the LoRA parameters only. In federated settings, each client's
  local dataset contributes updates clipped and noised per-sample. This is
  weaker than DP over a developer's whole contribution, because one developer contributes many blocks
  (see `docs/decisions/2026-11-14-amended-decisions.md`, D7).
- **Implication**: The guarantee is (ε, δ)-DP with respect to adding or
  removing a single training example from the dataset.

## The budget covers the whole run

ε ≤ 8 at δ = 1e-5 is the budget of one client's **whole federated run of 2
rounds**, not of one round (decision U3, `docs/decisions/2026-11-14-amended-decisions.md`).

- **Calibration.** Every round of a run trains with the same noise multiplier:
  the smallest σ for which 2 rounds of this size compose to at most 8
  (`security.dp.ledger.calibrate_run_sigma`, PRV accountant). `edge.train_client
  --dp` uses it; `--run-id` names the run and `--rounds-total` its length.
- **Composition.** A run's ε is read from one accountant that replays every
  round's (σ, sample rate, steps). Per-round ε values are never added: two
  rounds at ε 6.67 each compose to 7.90, not 13.3.
- **When it is charged.** At upload, by `edge.upload`, on the laptop's ledger
  (`security.dp.ledger.BudgetLedger`). A training that is never uploaded costs
  nothing; an upload the cluster refused costs nothing; re-sending an adapter
  that was already accepted costs nothing. An upload whose merge is later
  rejected has been spent.
- **Where it is enforced.** On the edge before any network call, and again by
  the cluster on the cumulative ε the upload reports. The registry records it.
- **A new run.** Training with a new `--run-id` starts a new ledger with a new
  budget. That is a new claim: ε ≤ 8 holds per run, not across runs.
- **What is not enforced.** The run is a name the laptop chooses. The cluster
  checks the cumulative ε and round count an upload reports; it does not know
  which run an upload belongs to (the run id is not sent). A developer who starts
  a new run and keeps training on the same promoted cluster adapter therefore
  gets a fresh budget on their own ledger, while the merged adapter now contains
  more than two rounds of their data: the cluster keeps the merge's ε at the
  largest value it has seen, and the registry ledger and the panel show the new
  run's lower figure for that client. ε ≤ 8 is a guarantee per run as declared by
  the client, not per adapter lineage. Closing this needs a decision (see the
  amended-decisions file, D7): for example the cluster refusing a client's upload
  onto a lineage it has already merged two of that client's rounds into.

## Budget Rationale: ε = 8, δ = 1e-5
- **Epsilon (ε = 8)**: 
  - (a) At ≤200 steps per client per round with `batch_size=1` (`sample_rate ≈ 1/60`), the noise needed for smaller ε would destroy utility for code generation.
  - (b) Empirically, code-gen LoRA fine-tuning is sensitive to gradient noise — ε < 4 typically prevents convergence within our step budget.
  - (c) ε = 8 provides a meaningful protection against membership inference while preserving model utility.
- **Delta (δ = 1e-5)**: Chosen as `1/N` where `N ≈ 100,000` (an order-of-magnitude estimate of total training examples across all clients).

## Noise Multiplier Selection
- **σ = 1.0 (Default)**: The noise multiplier relates to ε based on the sample rate `q` and the number of steps `T`. Formula: `ε ≈ (q * sqrt(T * log(1/δ))) / σ`. A noise multiplier of 1.0 provides a balanced starting point for our configuration.

## Gradient Clipping Strategy
- **C = 1.0**: This matches P1's existing `GRAD_CLIP` default. Per-sample clipping is enforced via Opacus, ensuring no individual training example contributes more than `C` to the overall gradient.

## Per-Round ε Computation
- The engine's own accountant (Opacus PRV, the one the noise is calibrated
  with) over the steps actually taken. The RDP bound for the same run is
  recorded beside it in the training manifest; it is looser.
- Formula: `ε(σ, q, T, δ)` where `q = sample_rate` and `T = steps`.

## Cumulative ε Tracking
- `security.dp.ledger.BudgetLedger` keeps one JSON file per client and run with
  each uploaded round's (σ, q, T). The cumulative ε is the composition of all
  of them at δ, and the ledger refuses a round that would exceed 8 or be a
  third round.

## Privacy-Utility Tradeoff Table

| Privacy Budget (ε) | Expected Utility | Risk Level | Use Case |
|--------------------|------------------|------------|----------|
| 2                  | Low              | Very Low   | High-security environments |
| 4                  | Moderate         | Low        | Strict privacy requirements |
| 8                  | High             | Moderate   | Default CLASP configuration |
| ∞ (No DP)          | Maximum          | High       | Public data only / Baselines |

## Configuration Reference
- `epsilon`: Target privacy budget of the run (all of its rounds together).
- `delta`: Target delta (typically `1/N`).
- `noise_multiplier`: Standard deviation of the Gaussian noise added to gradients.
- `max_grad_norm`: Maximum L2 norm for per-sample gradient clipping.
- `poisson_sampling`: Whether to use Poisson sampling for minibatches.

## Integration Guide
The P1 (Edge) service integrates via the `security` library
(`edge.train_client.train_dp`):
```python
from security import DPConfig, calibrate_run_sigma, make_private

# One noise level for every round of the run: 2 rounds compose to at most 8.
sigma = calibrate_run_sigma(steps_per_round=epochs * len(train_loader),
                            sample_rate=1 / len(train_loader))
config = DPConfig(noise_multiplier=sigma, target_epsilon=8.0, delta=1e-5, max_grad_norm=1.0)
model, optimizer, train_loader = make_private(model, optimizer, train_loader, config)
```

`make_private` uses the noise multiplier it is given. The budget is not
enforced by training: it is enforced when the adapter is uploaded, by the
ledger (`edge.upload`, see "The budget covers the whole run" above).
`make_private(..., epochs=)`, which calibrates a single round to the whole
budget, is not used by the edge: a round trained that way leaves no room for a
second one.
