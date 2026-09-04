"""The training algorithm: loss function and the CLI loop.

The objective is decided (`docs/design.md` §5.2): behavior cloning against
expert trajectories (Algorithm Distillation), not policy gradient or
actor-critic — see `PolicyOutput`'s docstring in `model/config.py` for why
there's no value head to support the latter.

This loop samples one recorded episode at a time (see `model/data.py`,
`model/adapters.py`'s `build_training_example`) — no cross-episode batching
or padding yet. With a single recording available, that makes this exactly
the Fase 2 sanity check from `docs/design.md` §7: can the architecture
overfit one game? Batching many episodes (and the padding that requires,
already supported by `Observation.padding_mask`/`TrainingExample.valid`) is
Fase 3 scope — training at scale over a procedurally generated distribution
— not built here.

Usage:
    .venv/bin/python -m model.train --steps 2000 --recordings-dir recordings
"""
from __future__ import annotations

import argparse
import logging
import random

import torch
from torch.nn import functional as F

from .adapters import build_training_example
from .config import ModelConfig, PolicyOutput, TrainConfig, TrainingExample
from .data import RecordingDataset
from .inference import save_checkpoint
from .network import ARCPolicyNetwork

logger = logging.getLogger(__name__)


# ── Loss ─────────────────────────────────────────────────────────────────


def policy_loss(output: PolicyOutput, example: TrainingExample) -> torch.Tensor:
    """Cross-entropy against the action an expert trajectory actually took
    at every step, plus (x, y) cross-entropy at steps whose target action
    was ACTION6. Masked by `example.valid` (excludes steps whose next
    action was RESET — the agent's decision, never the model's, see
    `TrainingExample`'s docstring) and `example.observation.padding_mask`
    (excludes batch padding, always all-True for a single episode).
    """
    mask = example.valid & example.observation.padding_mask  # [B, T]

    action_ce = F.cross_entropy(
        output.action_logits.transpose(1, 2), example.action_targets, reduction="none"
    )  # [B, T]
    action_denom = mask.sum().clamp(min=1)
    loss = (action_ce * mask).sum() / action_denom

    coord_mask = mask & example.is_complex
    x_ce = F.cross_entropy(output.x_logits.transpose(1, 2), example.x_targets, reduction="none")
    y_ce = F.cross_entropy(output.y_logits.transpose(1, 2), example.y_targets, reduction="none")
    coord_denom = coord_mask.sum().clamp(min=1)
    loss = loss + ((x_ce + y_ce) * coord_mask).sum() / coord_denom

    return loss


# ── CLI loop ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=TrainConfig.num_steps)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size,
                    help="Unused for now — one episode per step. See module docstring.")
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--checkpoint-dir", default=TrainConfig.checkpoint_dir)
    p.add_argument("--checkpoint-every", type=int, default=TrainConfig.checkpoint_every)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--recordings-dir", default="recordings",
                    help="Directory of .recording.jsonl episodes (model/data.py).")
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

    dataset = RecordingDataset(args.recordings_dir)
    if len(dataset) == 0:
        raise SystemExit(
            f"No recordings found in {args.recordings_dir!r}. Record one first, e.g.:\n"
            f"  .venv/bin/python scripts/play_local.py --game ls20 --max-steps 150 --record"
        )
    logger.info(
        "Training on %d recorded episode(s) from %r, device=%s, %d steps (checkpoints -> %s)",
        len(dataset), args.recordings_dir, train_config.device,
        train_config.num_steps, train_config.checkpoint_dir,
    )

    model = ARCPolicyNetwork(model_config).to(train_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    sampler = random.Random(train_config.seed)
    for step in range(1, train_config.num_steps + 1):
        episode = dataset[sampler.randrange(len(dataset))]
        example = build_training_example(episode, model_config, device=train_config.device)

        optimizer.zero_grad()
        output = model(example.observation)
        loss = policy_loss(output, example)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            logger.info("step %d  loss %.4f", step, loss.item())
        if step % train_config.checkpoint_every == 0:
            save_checkpoint(
                f"{train_config.checkpoint_dir}/step_{step}.pt",
                model, model_config, step=step,
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
