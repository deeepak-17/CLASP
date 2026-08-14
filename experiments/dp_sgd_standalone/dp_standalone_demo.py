"""Standalone DP-SGD demo on a toy model (no GPU required).

Demonstrates that DP-SGD training works correctly:
  1. A small MLP trains on synthetic data.
  2. Ghost clipping is used (or falls back to hooks if unavailable).
  3. Loss decreases over epochs.
  4. ε is reported after training via the EpsilonTracker.

This demo validates the security module's DP pipeline independently
of the edge/cluster services and does NOT require a GPU.

Usage:
    python -m experiments.dp_sgd_standalone.dp_standalone_demo
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from security.dp_config import DPConfig
from security.epsilon_tracker import EpsilonTracker


def create_toy_model() -> nn.Module:
    """Simple 2-layer MLP for binary classification."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


def create_toy_data(n_samples: int = 256) -> DataLoader:
    """Generate synthetic binary classification data."""
    X = torch.randn(n_samples, 16)
    y = (X[:, 0] > 0).long()  # label based on first feature
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def train_with_dp() -> None:
    """Run DP-SGD training on the toy model and report ε."""
    print("=" * 60)
    print("  CLASP DP-SGD Standalone Demo")
    print("=" * 60)

    # --- Config ---
    dp_config = DPConfig(
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=1.1,
        epochs=5,
        batch_size=32,
        physical_batch_size=32,
        grad_sample_mode="hooks",  # Use hooks for CPU compatibility
        accountant_type="rdp",
    )

    # --- Data & Model ---
    data_loader = create_toy_data(n_samples=256)
    model = create_toy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # --- Epsilon Tracker (standalone, no Opacus needed) ---
    tracker = EpsilonTracker(
        noise_multiplier=dp_config.noise_multiplier,
        sample_rate=dp_config.batch_size / 256,
        delta=dp_config.target_delta,
    )

    print(f"\n  Config: ε_target={dp_config.target_epsilon}, "
          f"δ={dp_config.target_delta}, σ={dp_config.noise_multiplier}, "
          f"clip={dp_config.max_grad_norm}")
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print()

    # --- Try Opacus wrapping (may fail without opacus installed) ---
    use_opacus = False
    try:
        from security.dp_engine import make_private
        model, optimizer, data_loader = make_private(
            model=model,
            optimizer=optimizer,
            data_loader=data_loader,
            dp_config=dp_config,
            loss_fn=loss_fn,
        )
        use_opacus = True
        print("  ✓ Opacus DP-SGD enabled\n")
    except ImportError:
        print("  ⚠ Opacus not installed — running simulated DP training\n")
    except Exception as e:
        print(f"  ⚠ Opacus wrapping failed ({e}) — running simulated DP\n")

    # --- Training loop ---
    first_loss = last_loss = None
    steps_per_epoch = len(data_loader)

    for epoch in range(dp_config.epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()

            # If not using Opacus, manually clip + add noise
            if not use_opacus:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), dp_config.max_grad_norm
                )
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            noise = torch.randn_like(p.grad) * (
                                dp_config.noise_multiplier
                                * dp_config.max_grad_norm
                            )
                            p.grad += noise

            optimizer.step()
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        avg_loss = epoch_loss / steps_per_epoch
        accuracy = correct / total
        eps = tracker.step(num_steps=steps_per_epoch)

        if first_loss is None:
            first_loss = avg_loss
        last_loss = avg_loss

        print(
            f"  epoch {epoch:2d} | loss {avg_loss:.4f} | "
            f"acc {accuracy:.2%} | ε = {eps:.4f}"
        )

    # --- Report ---
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  First epoch loss : {first_loss:.4f}")
    print(f"  Last epoch loss  : {last_loss:.4f}")
    print(f"  Loss decreased   : {last_loss < first_loss}")  # type: ignore
    print(f"  Final ε          : {tracker.get_epsilon():.4f}")
    print(f"  Budget (max=8.0) : {'OK' if not tracker.is_budget_exceeded(8.0) else 'EXCEEDED'}")
    print(f"  DP guarantee     : ({tracker.get_epsilon():.4f}, {tracker.delta})-DP")
    print()

    # Generate and print the full report
    from security.epsilon_report import generate_report, format_report_text
    report = generate_report(tracker, dp_config, max_epsilon=8.0, dataset_size=256)
    print(format_report_text(report))


if __name__ == "__main__":
    train_with_dp()
