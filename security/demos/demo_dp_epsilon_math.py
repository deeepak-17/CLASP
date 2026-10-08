"""CLASP Security — ε math explainer demo (Panel 1 Q&A prep).

Interactive script that computes ε for given (noise_multiplier, steps, δ)
and explains the RDP → (ε,δ)-DP conversion.

Usage:
    python -m security.demos.demo_dp_epsilon_math
    python -m security.demos.demo_dp_epsilon_math --noise-multiplier 0.5 --steps 200
"""
from __future__ import annotations

import argparse
import math

from opacus.accountants import RDPAccountant


def compute_epsilon(noise_multiplier: float, steps: int, sample_rate: float,
                    delta: float) -> float:
    """Compute ε using Opacus RDP accountant."""
    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return accountant.get_epsilon(delta=delta)


def main() -> None:
    ap = argparse.ArgumentParser(description="CLASP ε math explainer")
    ap.add_argument("--noise-multiplier", type=float, default=1.0,
                    help="Gaussian noise σ")
    ap.add_argument("--steps", type=int, default=200,
                    help="Number of optimization steps (D8 cap per client per round)")
    ap.add_argument("--sample-rate", type=float, default=0.016,
                    help="Batch / dataset ratio (batch_size=1, ~60 blocks → 1/60 ≈ 0.016)")
    ap.add_argument("--delta", type=float, default=1e-5,
                    help="δ parameter")
    args = ap.parse_args()

    sigma = args.noise_multiplier
    T = args.steps
    q = args.sample_rate
    delta = args.delta

    print()
    print("=" * 64)
    print("  CLASP Security — Differential Privacy ε Calculator")
    print("=" * 64)
    print()
    print("  Parameters:")
    print(f"    noise_multiplier (σ)  : {sigma}")
    print(f"    optimization steps (T): {T}")
    print(f"    sample rate (q)       : {q:.4f}")
    print(f"    delta (δ)             : {delta}")
    print()

    # Compute ε
    eps = compute_epsilon(sigma, T, q, delta)

    print("  ─── RDP → (ε,δ)-DP Conversion ───")
    print()
    print(f"  Rényi Differential Privacy (RDP) accountant tracks privacy loss")
    print(f"  at multiple orders α, then converts to (ε,δ)-DP by minimizing:")
    print(f"    ε = min_α [ RDP(α) + log(1/δ) / (α-1) ]")
    print()
    print(f"  ─── Result ───")
    print()
    print(f"  ε = {eps:.4f}")
    print(f"  δ = {delta}")
    print(f"  e^ε = {math.exp(eps):.4f}")
    print()
    print(f"  Interpretation:")
    print(f"    The probability ratio of any output being produced with or")
    print(f"    without a single client's data is bounded by e^ε ≈ {math.exp(eps):.2f}.")
    print()

    # Sweep table
    print("  ─── ε Sensitivity Table ───")
    print()
    print(f"  (Fixed: T={T}, q={q:.4f}, δ={delta})")
    print()
    print(f"  {'σ':>8}  {'ε':>10}  {'e^ε':>10}  {'Privacy Level':>20}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*20}")

    for s in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        e = compute_epsilon(s, T, q, delta)
        if e > 100:
            level = "❌ no meaningful DP"
        elif e > 10:
            level = "⚠️ weak"
        elif e > 5:
            level = "moderate"
        elif e > 2:
            level = "✅ good"
        else:
            level = "✅✅ strong"
        print(f"  {s:8.1f}  {e:10.4f}  {math.exp(min(e, 20)):10.2f}  {level:>20}")

    print()
    print("  ─── CLASP Default Configuration ───")
    print()
    print(f"  σ = 1.0, C = 1.0, δ = 1e-5, T ≤ 200 steps/client/round")
    print(f"  Target: ε ≤ 8 per round")
    default_eps = compute_epsilon(1.0, 200, q, 1e-5)
    print(f"  Actual:  ε = {default_eps:.4f} at T=200")
    print(f"  Budget:  {'✅ within budget' if default_eps <= 8.0 else '❌ exceeds budget'}")
    print()


if __name__ == "__main__":
    main()
