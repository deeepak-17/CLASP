"""CLASP Security — DP-SGD toy demo (Panel 1, W5 Mon).

Trains a small 2-layer MLP on synthetic binary-classification data using
Opacus DP-SGD. Prints loss and ε after each epoch to demonstrate that
differential privacy is being applied.

Usage:
    python -m security.demos.demo_dp_toy
    python -m security.demos.demo_dp_toy --noise-multiplier 0.5 --epochs 20
"""
from __future__ import annotations

import argparse

import torch
from opacus import PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_model() -> nn.Module:
    """2-layer MLP, ~1000 params — small enough for CPU, big enough to demo."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


def build_data(n: int = 200, dim: int = 16) -> DataLoader:
    """Synthetic binary classification data."""
    torch.manual_seed(0)
    x = torch.randn(n, dim)
    y = (x[:, 0] + x[:, 1] > 0).long()  # simple linear boundary
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="CLASP DP-SGD toy demo")
    ap.add_argument("--noise-multiplier", type=float, default=1.0,
                    help="Gaussian noise σ (higher = more private)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="Per-sample gradient clipping bound C")
    ap.add_argument("--epochs", type=int, default=10,
                    help="Number of training epochs")
    ap.add_argument("--delta", type=float, default=1e-5,
                    help="δ parameter for (ε,δ)-DP")
    args = ap.parse_args()

    print("=" * 60)
    print("  CLASP Security — DP-SGD Toy Demo")
    print("=" * 60)
    print(f"  noise_multiplier (σ)  : {args.noise_multiplier}")
    print(f"  max_grad_norm (C)     : {args.max_grad_norm}")
    print(f"  delta (δ)             : {args.delta}")
    print(f"  epochs                : {args.epochs}")
    print("=" * 60)

    model = build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loader = build_data()
    criterion = nn.CrossEntropyLoss()

    # Wrap with Opacus DP-SGD
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

    print(f"\n{'Epoch':>5}  {'Loss':>8}  {'ε':>10}  {'δ':>10}")
    print("-" * 40)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        epsilon = privacy_engine.get_epsilon(delta=args.delta)
        print(f"{epoch:5d}  {avg_loss:8.4f}  {epsilon:10.4f}  {args.delta:10.1e}")

    final_eps = privacy_engine.get_epsilon(delta=args.delta)
    print("-" * 40)
    print(f"\n  Final privacy guarantee: (ε={final_eps:.4f}, δ={args.delta})-DP")
    print(f"  Model trained for {args.epochs} epochs with σ={args.noise_multiplier}")
    print("\n  Interpretation: An adversary observing the trained model cannot")
    print("  distinguish whether any single training example was included")
    print(f"  in the dataset with probability better than e^ε ≈ {__import__('math').exp(final_eps):.2f}")
    print()


if __name__ == "__main__":
    main()
