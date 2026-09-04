"""Everything about using a trained model: saving/loading checkpoints, and
`ModelPolicy`, the only class `agent/my_agent.py` should import from
`model/`. Load a checkpoint once in `Agent.__init__`, then call `.act()`
once per `choose_action`.

A checkpoint stores the `ModelConfig` alongside the `state_dict` so it is
self-describing: loading one rebuilds the exact `ARCPolicyNetwork`
architecture it was saved with, without the caller having to know its
hyperparameters in advance.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from arcengine import FrameData, GameAction
from torch import nn

from .adapters import build_observation, decode_action
from .config import InferenceConfig, ModelConfig
from .network import ARCPolicyNetwork

logger = logging.getLogger(__name__)

# Bump this if the checkpoint dict's shape ever changes incompatibly.
CHECKPOINT_FORMAT_VERSION = 1


# ── Checkpoints ──────────────────────────────────────────────────────────


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: ModelConfig,
    *,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "config": dataclasses.asdict(config),
            "state_dict": model.state_dict(),
            "step": step,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    # weights_only=False: our payload also holds a plain dict (the config),
    # not just tensors. It's always a checkpoint you produced yourself via
    # `save_checkpoint`, never untrusted third-party input.
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format_version {payload.get('format_version')!r} is not "
            f"supported (expected {CHECKPOINT_FORMAT_VERSION})."
        )
    return payload


def config_from_checkpoint(payload: dict[str, Any]) -> ModelConfig:
    return ModelConfig(**payload["config"])


# ── Agent-facing policy wrapper ─────────────────────────────────────────


@runtime_checkable
class Policy(Protocol):
    """The interface `agent/my_agent.py` programs against, so the backend
    behind it (the random baseline, `ModelPolicy`, or anything else) is
    swappable without touching agent code.

    `reset()` clears whatever episode-scoped state a policy holds — it's a
    no-op today (`ModelPolicy` carries no memory across calls yet), but the
    call site in `MyAgent.__init__` is here so that once a sequential model
    starts accumulating memory across a game, wiring it in doesn't require
    touching the agent again.
    """

    def act(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction: ...

    def reset(self) -> None: ...


class ModelPolicy:
    """Wraps a loaded `ARCPolicyNetwork` for inference inside the agent loop."""

    def __init__(
        self,
        model: ARCPolicyNetwork,
        model_config: ModelConfig,
        inference_config: InferenceConfig | None = None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.inference_config = inference_config or InferenceConfig()
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        inference_config: InferenceConfig | None = None,
    ) -> "ModelPolicy":
        """Raises if the checkpoint is missing or the architecture it
        describes has no implemented forward pass yet. Prefer
        `try_from_checkpoint` from agent code that needs a graceful
        fallback.
        """
        inference_config = inference_config or InferenceConfig()
        payload = load_checkpoint(checkpoint_path, map_location=inference_config.device)
        model_config = config_from_checkpoint(payload)
        model = ARCPolicyNetwork(model_config)
        model.load_state_dict(payload["state_dict"])
        model.to(inference_config.device)
        return cls(model, model_config, inference_config)

    @classmethod
    def try_from_checkpoint(
        cls,
        checkpoint_path: str | Path | None,
        inference_config: InferenceConfig | None = None,
    ) -> "ModelPolicy | None":
        """Convenience for agent code: returns None instead of raising when
        there's no checkpoint yet (or it fails to load), so callers can fall
        back to a non-model strategy.
        """
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return None
        try:
            return cls.from_checkpoint(checkpoint_path, inference_config)
        except Exception:
            logger.exception("Failed to load model checkpoint from %s", checkpoint_path)
            return None

    @torch.no_grad()
    def act(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Same call signature as `Agent.choose_action` — drop this straight
        into `MyAgent.choose_action` once a checkpoint exists. Only valid to
        call while the game is actually playable (not on NOT_PLAYED /
        GAME_OVER — the agent still owns RESET).
        """
        obs = build_observation(
            frames, latest_frame, self.model_config, device=self.inference_config.device
        )
        output = self.model(obs)
        return decode_action(output, self.inference_config)

    def reset(self) -> None:
        """Clear episode-scoped state. No-op until the network carries
        memory across an episode (see `docs/design.md` §4.4) — kept here so
        `Policy` has one call site in the agent regardless of backend.
        """
