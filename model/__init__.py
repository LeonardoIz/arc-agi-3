"""PyTorch model package for the ARC-AGI-3 agent.

This package is deliberately independent of `agents.agent.Agent` and the
`arcengine` game runtime except at one seam: `model.adapters`, the only
module that imports `arcengine` (translates `FrameData` <-> tensors both
ways). Everything else — `config`, `network`, `inference`, `data`, `train`
— is pure PyTorch and knows nothing about the game framework, so it can be
unit-tested or trained offline without a live game.

Layout (7 files, one concern each):
  - `config.py`    — hyperparameters + the tensor containers passed between
                      modules (`Observation`, `PolicyOutput`).
  - `network.py`    — the architecture: a per-frame Slot Attention encoder
                      (C10) feeding a causal transformer over the episode
                      (C4), composed into `ARCPolicyNetwork`. See its module
                      docstring for the full design rationale.
  - `adapters.py`    — `FrameData` <-> `Observation`/`GameAction` translation.
  - `inference.py`    — checkpoint save/load + `ModelPolicy`/`Policy`, the
                        interface `agent/my_agent.py` should import from here.
  - `data.py`          — loading recorded episodes + a replay buffer.
  - `train.py`          — loss functions + the training CLI loop.

`agent/my_agent.py` should only ever import `model.inference.Policy` /
`ModelPolicy` (and, for configuration, `model.config`) — never
`model.network` directly. That keeps the architecture swappable without
touching agent code.

The architecture is implemented; the training loop (`model/train.py`) is
not yet — see the `# TODO` markers there. See `docs/model.md` and
`docs/design.md` §4 for the full design.
"""
from __future__ import annotations

from .config import InferenceConfig, ModelConfig, TrainConfig
from .inference import ModelPolicy, Policy, load_checkpoint, save_checkpoint
from .network import ARCPolicyNetwork

__all__ = [
    "ModelConfig",
    "InferenceConfig",
    "TrainConfig",
    "ARCPolicyNetwork",
    "Policy",
    "ModelPolicy",
    "save_checkpoint",
    "load_checkpoint",
]
