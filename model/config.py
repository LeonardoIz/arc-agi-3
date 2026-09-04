"""Configuration and shared data shapes — the schema every other module in
`model/` agrees on.

Two kinds of things live here, together because both are "shape, not
behavior":

  - Config dataclasses (`ModelConfig`, `InferenceConfig`, `TrainConfig`):
    hyperparameters, no `torch.nn` layers.
  - Cross-boundary containers (`Observation`, `PolicyOutput`, `Decision`):
    plain dataclasses around tensors that `model.network` and
    `model.adapters` pass back and forth.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# `arcengine.camera.Camera.MAX_DIMENSION` — every rendered frame is 64x64.
GRID_SIZE = 64
# ARC-AGI's fixed 16-color palette (indices 0-15).
NUM_COLORS = 16
# ACTION1..ACTION7. RESET is excluded: the agent decides when to reset, the
# model only ever picks a gameplay action. See `model/adapters.py`.
NUM_SIMPLE_ACTIONS = 7
# ACTION6 takes (x, y) each in [0, 63] — see `arcengine.enums.ComplexAction`.
COORD_RANGE = 64


@dataclass
class ModelConfig:
    """Architecture hyperparameters for `model.network.ARCPolicyNetwork`.

    Stored inside every checkpoint (see `model/inference.py`) so a saved
    model is self-describing: `ModelPolicy.from_checkpoint` rebuilds the
    exact architecture it was trained with. There's a single backbone
    (`model.network.CNNBackbone`) for now — if you add alternatives later,
    reintroduce a `backbone` field here to select between them.
    """

    embed_dim: int = 256
    grid_size: int = GRID_SIZE
    num_colors: int = NUM_COLORS
    num_simple_actions: int = NUM_SIMPLE_ACTIONS
    coord_range: int = COORD_RANGE
    # How many trailing sub-frames of `FrameData.frame` to stack as temporal
    # context. 1 = current frame only.
    history_len: int = 1
    use_value_head: bool = False


@dataclass
class InferenceConfig:
    """Decoding/runtime behavior — independent of the architecture, so the
    same checkpoint can be replayed deterministically or sampled from
    without retraining.
    """

    device: str = "cpu"
    deterministic: bool = True
    temperature: float = 1.0


@dataclass
class TrainConfig:
    """Training hyperparameters read by `model/train.py`.

    No training algorithm is implemented yet (see that module) — this just
    fixes the shape of the config so the CLI/loop skeleton has something to
    parse and pass around.
    """

    lr: float = 3e-4
    batch_size: int = 64
    num_steps: int = 100_000
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1000
    device: str = "cpu"
    seed: int = 0


@dataclass
class Observation:
    """One model input, already batched (batch dim = 1 for online play).

    Attributes:
        grid: LongTensor [B, T, H, W] of color indices in [0, num_colors),
            T = `ModelConfig.history_len` trailing sub-frames, oldest first.
        action_mask: BoolTensor [B, num_simple_actions], True where that
            action is currently legal (from `FrameData.available_actions`).
    """

    grid: torch.Tensor
    action_mask: torch.Tensor


@dataclass
class PolicyOutput:
    """Raw network output, before any sampling/decoding decision is made."""

    action_logits: torch.Tensor  # [B, num_simple_actions]
    x_logits: torch.Tensor | None = None  # [B, coord_range] — ACTION6 only
    y_logits: torch.Tensor | None = None  # [B, coord_range] — ACTION6 only
    value: torch.Tensor | None = None  # [B] — only if config.use_value_head


@dataclass
class Decision:
    """A fully decoded action, independent of the `arcengine.GameAction`
    enum — kept around for training/logging paths that don't want a
    hard dependency on the game framework.
    """

    action_index: int  # index into model.adapters.POLICY_ACTIONS
    x: int | None = None
    y: int | None = None
