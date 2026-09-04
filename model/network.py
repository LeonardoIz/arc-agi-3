"""The architecture: grid encoder -> output heads -> composed network.

Kept as one file but still three clearly-bounded pieces, top to bottom:

  1. `GridBackbone` / `CNNBackbone` — turns stacked color grids into a
     feature vector. This is the piece most likely to change as you
     experiment; swap `CNNBackbone` for a different architecture later by
     adding a new `GridBackbone` subclass and pointing `ARCPolicyNetwork`
     at it. A single CNN is provided as the starting point: ARC grids are
     small (64x64, 16 colors) and spatially local, which is exactly what
     convolutions are good at.
  2. `PolicyHead` / `ValueHead` — project features into action logits (and
     an optional state-value estimate for actor-critic-style training).
  3. `ARCPolicyNetwork` — composes the two. This is the only real logic in
     the file: masking illegal actions is a hard rule of the game, not a
     learned behavior, so it isn't left as a TODO like the rest.

No backbone or head forward pass is implemented yet — see the TODOs below.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from .config import ModelConfig, Observation, PolicyOutput

# ── 1. Backbone ──────────────────────────────────────────────────────────


class GridBackbone(nn.Module, ABC):
    """Encodes a batch of stacked color-index grids into embeddings.

    Input:  LongTensor [B, T, H, W] of color indices in [0, config.num_colors).
    Output: FloatTensor [B, out_dim].
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Dimensionality of the embedding this backbone produces."""

    @abstractmethod
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """grid: [B, T, H, W] long -> returns [B, out_dim] float."""


class CNNBackbone(GridBackbone):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._out_dim = config.embed_dim
        # TODO: build the conv stack, e.g. an `nn.Embedding(config.num_colors,
        # channels)` for the palette followed by conv/pool layers that reduce
        # the config.grid_size x config.grid_size input down to a vector of
        # size `self._out_dim`.

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "CNNBackbone.forward is a structural placeholder — implement "
            "the conv stack here. Input is grid: LongTensor[B, T, H, W]; "
            "return a FloatTensor[B, self.out_dim]."
        )


# ── 2. Heads ─────────────────────────────────────────────────────────────


class PolicyHead(nn.Module):
    """Backbone features -> action logits (+ x/y logits for ACTION6)."""

    def __init__(self, config: ModelConfig, in_dim: int) -> None:
        super().__init__()
        self.config = config
        self.in_dim = in_dim
        # TODO: layers mapping `in_dim` -> `config.num_simple_actions` action
        # logits, and `in_dim` -> `config.coord_range` logits each for x
        # and y (used only when the sampled action is ACTION6).

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """features: [B, in_dim] ->
        (action_logits [B, num_simple_actions], x_logits [B, coord_range],
         y_logits [B, coord_range]).
        """
        raise NotImplementedError(
            "PolicyHead.forward is a structural placeholder — implement "
            "the action/x/y projection layers here."
        )


class ValueHead(nn.Module):
    """Backbone features -> a scalar state-value estimate. Only wired into
    `ARCPolicyNetwork` when `config.use_value_head` is True (needed for
    actor-critic-style training; skip it entirely for behavior cloning).
    """

    def __init__(self, config: ModelConfig, in_dim: int) -> None:
        super().__init__()
        self.config = config
        self.in_dim = in_dim
        # TODO: layers mapping `in_dim` -> a single scalar value estimate.

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, in_dim] -> value [B]."""
        raise NotImplementedError(
            "ValueHead.forward is a structural placeholder — implement the "
            "value projection here."
        )


# ── 3. Composed network ─────────────────────────────────────────────────


class ARCPolicyNetwork(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = CNNBackbone(config)
        self.policy_head = PolicyHead(config, in_dim=self.backbone.out_dim)
        self.value_head = (
            ValueHead(config, in_dim=self.backbone.out_dim)
            if config.use_value_head
            else None
        )

    def forward(self, obs: Observation) -> PolicyOutput:
        features = self.backbone(obs.grid)
        action_logits, x_logits, y_logits = self.policy_head(features)
        action_logits = action_logits.masked_fill(~obs.action_mask, float("-inf"))
        value = self.value_head(features) if self.value_head is not None else None
        return PolicyOutput(
            action_logits=action_logits, x_logits=x_logits, y_logits=y_logits, value=value
        )
