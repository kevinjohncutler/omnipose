#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import torch


def build_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    )


def run_optimizer(name: str, optimizer_cls, steps: int, seed: int) -> torch.Tensor:
    model = build_model(seed)
    x = torch.randn(64, 16)
    y = torch.randn(64, 4)
    loss_fn = torch.nn.MSELoss()

    opt = optimizer_cls(model.parameters(), lr=1e-3)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        final_loss = loss_fn(model(x), y).detach().cpu()
    return final_loss


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch RAdam vs torch_optimizer.RAdam on a synthetic task."
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of optimization steps.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility.")
    args = parser.parse_args()

    if not hasattr(torch.optim, "RAdam"):
        print("torch.optim.RAdam is not available in this PyTorch version.", file=sys.stderr)
        return 1

    torch_loss = run_optimizer("torch.optim.RAdam", torch.optim.RAdam, args.steps, args.seed)
    print(f"torch.optim.RAdam final loss: {torch_loss:.6f}")

    try:
        import torch_optimizer
    except Exception as exc:
        print(f"torch_optimizer not available: {exc}", file=sys.stderr)
        return 0

    optimizer_loss = run_optimizer("torch_optimizer.RAdam", torch_optimizer.RAdam, args.steps, args.seed)
    print(f"torch_optimizer.RAdam final loss: {optimizer_loss:.6f}")

    delta = (optimizer_loss - torch_loss).abs()
    print(f"abs(delta): {delta:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
