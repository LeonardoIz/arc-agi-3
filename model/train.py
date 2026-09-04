"""The training algorithm: loss functions and the CLI loop. Nothing here is
implemented yet — which objective to train against (behavior cloning,
policy gradient, actor-critic, ...) is a modeling decision, and the loop
itself needs that decision made before it can run.

Usage (once the TODOs below are filled in):
    .venv/bin/python -m model.train --steps 100000
"""
from __future__ import annotations

import argparse
import logging

import torch

from .config import ModelConfig, PolicyOutput, TrainConfig
from .inference import save_checkpoint
from .network import ARCPolicyNetwork

logger = logging.getLogger(__name__)


# ── Losses ───────────────────────────────────────────────────────────────


def policy_loss(output: PolicyOutput, actions: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """actions: [B] long (index into POLICY_ACTIONS). advantages: [B] float
    (return, advantage, or a constant 1.0 for pure behavior cloning).
    Returns a scalar loss.
    """
    raise NotImplementedError("policy_loss is not implemented yet.")


def value_loss(output: PolicyOutput, returns: torch.Tensor) -> torch.Tensor:
    """returns: [B] float target for `output.value`. Returns a scalar loss."""
    raise NotImplementedError("value_loss is not implemented yet.")


# ── CLI loop ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=TrainConfig.num_steps)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--checkpoint-dir", default=TrainConfig.checkpoint_dir)
    p.add_argument("--checkpoint-every", type=int, default=TrainConfig.checkpoint_every)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    torch.manual_seed(args.seed)

    model_config = ModelConfig()
    train_config = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        num_steps=args.steps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        seed=args.seed,
    )

    model = ARCPolicyNetwork(model_config).to(train_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    logger.info(
        "Training on %s for %d steps (checkpoints -> %s)",
        train_config.device,
        train_config.num_steps,
        train_config.checkpoint_dir,
    )

    # TODO: pull batches (see model/data.py), build an Observation, run the
    # model, compute a loss (see policy_loss/value_loss above), and step the
    # optimizer:
    #
    #   for step in range(1, train_config.num_steps + 1):
    #       optimizer.zero_grad()
    #       output = model(obs)
    #       loss = policy_loss(output, actions, advantages)
    #       loss.backward()
    #       optimizer.step()
    #       if step % train_config.checkpoint_every == 0:
    #           save_checkpoint(
    #               f"{train_config.checkpoint_dir}/step_{step}.pt",
    #               model, model_config, step=step,
    #           )
    raise NotImplementedError("The training loop is not implemented yet — see the TODO above.")


if __name__ == "__main__":
    main()
