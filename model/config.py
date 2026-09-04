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
    exact architecture it was trained with.

    See `docs/design.md` §4 for why the architecture takes this shape:
    a per-frame Slot Attention encoder (C10 — learned object decomposition,
    no hand-coded segmentation) feeding a causal transformer over the whole
    episode (C4 — memory lives in the model's context, not in agent code).
    """

    grid_size: int = GRID_SIZE
    num_colors: int = NUM_COLORS
    num_simple_actions: int = NUM_SIMPLE_ACTIONS
    coord_range: int = COORD_RANGE

    # ── Slot encoder (per frame) ──
    color_embed_dim: int = 32
    cnn_channels: int = 64
    num_slots: int = 8
    slot_dim: int = 128
    slot_iters: int = 3

    # ── Sequential model (over the episode) ──
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_mult: int = 4
    dropout: float = 0.0

    # Safety cap on episode length, not a capacity decision: the causal
    # transformer uses RoPE, which doesn't impose a hard maximum context on
    # its own. This just bounds memory use if an episode runs away; known
    # public-game baselines top out at 1843 actions (wa30), well under this.
    max_context_len: int = 4096


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
    """One model input: a whole episode-so-far, batched.

    Every tensor's second dimension is T, the number of steps observed so
    far in the episode (grown by one each call during online play; a full,
    padded trajectory length during training). The network predicts an
    action at *every* position in parallel (teacher forcing); callers doing
    online play only look at the last one (see `model.adapters.decode_action`).

    Attributes:
        grid: LongTensor [B, T, H, W] of color indices in [0, num_colors) —
            the settled frame at each step.
        action_mask: BoolTensor [B, T, num_simple_actions], True where that
            action was legal at that step (`FrameData.available_actions`).
        prev_actions: LongTensor [B, T], index into
            `model.adapters.POLICY_ACTIONS` of the action that produced this
            step's frame, or `num_simple_actions` (one past the end) as the
            "start of episode" token where the producing action was RESET.
        level_ups: LongTensor [B, T] of 0/1 — whether the action that
            produced this step's frame increased `levels_completed`. The
            only reward-shaped signal the model ever sees (§2.4).
        padding_mask: BoolTensor [B, T], True where the step is real data.
            Always all-True during online play (one episode, no padding);
            used during batched training so shorter episodes can share a
            batch with longer ones without attending into padding.
    """

    grid: torch.Tensor
    action_mask: torch.Tensor
    prev_actions: torch.Tensor
    level_ups: torch.Tensor
    padding_mask: torch.Tensor


@dataclass
class PolicyOutput:
    """Raw network output at every timestep, before any sampling/decoding
    decision is made. No value head: training is behavior cloning over
    expert trajectories (Algorithm Distillation, `docs/design.md` §5.2),
    which needs no state-value estimate, and C9 (no search at inference)
    means nothing downstream would consume one.
    """

    action_logits: torch.Tensor  # [B, T, num_simple_actions]
    x_logits: torch.Tensor  # [B, T, coord_range] — meaningful only where the decoded action is ACTION6
    y_logits: torch.Tensor  # [B, T, coord_range]


@dataclass
class Decision:
    """A fully decoded action, independent of the `arcengine.GameAction`
    enum — kept around for training/logging paths that don't want a
    hard dependency on the game framework.
    """

    action_index: int  # index into model.adapters.POLICY_ACTIONS
    x: int | None = None
    y: int | None = None


@dataclass
class TrainingExample:
    """One recorded episode turned into teacher-forcing training tensors
    (built by `model.adapters.build_training_example`). Unbatched shapes
    below have an implicit leading batch dim of 1, matching `Observation`.

    T = (number of frames in the episode) - 1: the network predicts a next
    action at every step except the episode's last frame, which has no
    "next" action to predict.

    Attributes:
        observation: inputs for steps [0, T) — see `Observation`.
        action_targets: LongTensor [B, T], index into
            `model.adapters.POLICY_ACTIONS` of the action actually taken
            after each step.
        x_targets / y_targets: LongTensor [B, T] in [0, config.coord_range)
            — meaningful only where `is_complex` is True.
        is_complex: BoolTensor [B, T], True where the target action was
            ACTION6 (so x/y should be part of the loss at that step).
        valid: BoolTensor [B, T], True where the target action was an
            actual policy output. False where it was RESET — RESET is the
            agent's decision (see `agent/my_agent.py`), never the model's,
            so those steps are excluded from the loss the same way padding
            is (combine with `observation.padding_mask`).
    """

    observation: Observation
    action_targets: torch.Tensor
    x_targets: torch.Tensor
    y_targets: torch.Tensor
    is_complex: torch.Tensor
    valid: torch.Tensor
