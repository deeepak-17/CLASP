from __future__ import annotations

import argparse
import logging
import sys

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("Error: PyTorch is required to run this script.", file=sys.stderr)
    sys.exit(1)

from security.dp.config import DPConfig
from security.dp.engine import get_privacy_engine, make_private

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleMLP(nn.Module):
    """A simple 2-layer MLP for binary classification."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="Standalone Toy DP-SGD Demo")
    parser.add_argument("--noise_multiplier", type=float, default=1.0, help="Noise multiplier (sigma) for DP-SGD")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max per-sample gradient norm (C) for clipping")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    args = parser.parse_args()

    # 1. Create Synthetic Data (100 samples, 10 features, binary target)
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    
    batch_size = 10
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model, Optimizer, Loss
    model = SimpleMLP()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"Initialized SimpleMLP with ~{sum(p.numel() for p in model.parameters())} parameters.")

    # 3. Create DP Configuration and Wrap with make_private
    config = DPConfig(
        enabled=True,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        target_epsilon=10.0,
    )
    
    model, optimizer, data_loader = make_private(model, optimizer, data_loader, config)
    privacy_engine = get_privacy_engine(model)
    
    # 4. Train Model
    logger.info("Starting DP-SGD training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(data_loader)
        
        # Calculate spent epsilon
        epsilon = privacy_engine.get_epsilon(delta=config.delta) if privacy_engine else 0.0
        logger.info(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | ε spent: {epsilon:.4f}")
        
    # 5. Final Output
    final_epsilon = privacy_engine.get_epsilon(delta=config.delta) if privacy_engine else 0.0
    logger.info(f"Training completed. Final privacy guarantee: (ε={final_epsilon:.4f}, δ={config.delta})")


if __name__ == "__main__":
    main()
