# CLASP DP-SGD Configuration (D7)

## Privacy Unit
- **Unit**: Per-sample (record-level). Each individual training example is the
  privacy unit. This means ε bounds the information leakage about any single
  training record.
- **Client-level vs record-level**: CLASP currently operates at record-level
  privacy. In federated settings, each client's local dataset contributes
  updates clipped and noised per-sample. Client-level DP (bounding leakage
  about an entire client's dataset) is a future extension tracked in
  MASTER_PLAN §2.
- **Implication**: The guarantee is (ε, δ)-DP with respect to adding or
  removing a single training example from the dataset.

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
- Utilizes the RDP (Rényi Differential Privacy) accountant.
- Formula: `ε(σ, q, T, δ)` where `q = sample_rate` and `T = steps`.

## Cumulative ε Tracking
- Tracks composition across rounds. Under RDP, privacy degradation grows sublinearly, allowing for many federated rounds before hitting global privacy budget limits.

## Privacy-Utility Tradeoff Table

| Privacy Budget (ε) | Expected Utility | Risk Level | Use Case |
|--------------------|------------------|------------|----------|
| 2                  | Low              | Very Low   | High-security environments |
| 4                  | Moderate         | Low        | Strict privacy requirements |
| 8                  | High             | Moderate   | Default CLASP configuration |
| ∞ (No DP)          | Maximum          | High       | Public data only / Baselines |

## Configuration Reference
- `epsilon`: Target privacy budget per round.
- `delta`: Target delta (typically `1/N`).
- `noise_multiplier`: Standard deviation of the Gaussian noise added to gradients.
- `max_grad_norm`: Maximum L2 norm for per-sample gradient clipping.
- `poisson_sampling`: Whether to use Poisson sampling for minibatches.

## Integration Guide
The P1 (Edge) service integrates via the `security.dp` wrapper:
```python
from security.dp import DPConfig, make_private

config = DPConfig(target_epsilon=8.0, delta=1e-5, max_grad_norm=1.0)

# Auto-calibrate noise to enforce ε ≤ 8 over the training run:
model, optimizer, train_loader = make_private(
    model, optimizer, train_loader, config, epochs=epochs
)
```

When `epochs` is provided, `make_private` uses Opacus's
`make_private_with_epsilon` internally to calibrate the noise multiplier σ
automatically. Without `epochs`, the caller must set `noise_multiplier`
explicitly and is responsible for the privacy budget.
