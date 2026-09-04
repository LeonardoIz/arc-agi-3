"""The arcengine <-> tensor boundary.

The only file in `model/` that imports from `arcengine`. Everything else
works purely in tensor space so it can be trained/tested without the game
framework. Two directions, both in this one file:

  - incoming: `arcengine.FrameData` -> `model.config.Observation`
  - outgoing: `model.config.PolicyOutput` -> `arcengine.GameAction`
"""
from __future__ import annotations

import torch
from arcengine import FrameData, GameAction

from .config import InferenceConfig, ModelConfig, Observation, PolicyOutput

# Canonical ordering between action-space indices [0, num_simple_actions)
# and `GameAction` members. RESET is excluded: the agent decides when to
# reset (see `agent/my_agent.py`), the model only ever chooses a gameplay
# action.
POLICY_ACTIONS: list[GameAction] = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
    GameAction.ACTION6,
    GameAction.ACTION7,
]


# ── incoming: FrameData -> Observation ──────────────────────────────────


def frame_to_grid(frame: FrameData) -> torch.Tensor:
    """The most recent sub-frame of `frame.frame` as a [H, W] LongTensor of
    color indices. `FrameData.frame` holds one grid per intermediate
    animation step produced by the last action — we only need the final,
    settled one.
    """
    if frame.is_empty():
        raise ValueError("Cannot build an observation from an empty FrameData.frame")
    return torch.tensor(frame.frame[-1], dtype=torch.long)


def build_observation(
    frames: list[FrameData],
    latest_frame: FrameData,
    config: ModelConfig,
    device: str = "cpu",
) -> Observation:
    """Stack the last `config.history_len` frames (padding by repeating the
    earliest available one if there isn't enough history yet) and build the
    legal-action mask from `latest_frame.available_actions`.

    Mirrors the call signature of `agents.agent.Agent.choose_action`:
    `frames` is the history *not* including `latest_frame`.
    """
    history = [f for f in frames if not f.is_empty()]
    if not latest_frame.is_empty():
        history.append(latest_frame)
    if not history:
        raise ValueError("No non-empty frames available to build an observation")

    history = history[-config.history_len :]
    while len(history) < config.history_len:
        history.insert(0, history[0])

    grid = torch.stack([frame_to_grid(f) for f in history], dim=0)  # [T, H, W]
    grid = grid.unsqueeze(0).to(device)  # [1, T, H, W]

    mask = torch.zeros(1, config.num_simple_actions, dtype=torch.bool, device=device)
    available = set(latest_frame.available_actions)
    for i, action in enumerate(POLICY_ACTIONS):
        if action.value in available:
            mask[0, i] = True

    return Observation(grid=grid, action_mask=mask)


# ── outgoing: PolicyOutput -> GameAction ────────────────────────────────


def _pick_index(logits: torch.Tensor, config: InferenceConfig) -> int:
    """logits: [N] -> a single index, either argmax or sampled."""
    if config.deterministic:
        return int(torch.argmax(logits, dim=-1).item())
    probs = torch.softmax(logits / max(config.temperature, 1e-6), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def decode_action(output: PolicyOutput, config: InferenceConfig) -> GameAction:
    """Pick a `GameAction` (with x/y filled in for ACTION6) from raw network
    output. Illegal actions are expected to already be masked to -inf in
    `output.action_logits` (see `ARCPolicyNetwork.forward`).
    """
    if output.action_logits.shape[0] != 1:
        raise ValueError(
            f"decode_action expects a batch size of 1, got {output.action_logits.shape[0]}"
        )
    idx = _pick_index(output.action_logits[0], config)
    action = POLICY_ACTIONS[idx]
    if action.is_complex():
        if output.x_logits is None or output.y_logits is None:
            raise ValueError(f"{action} is complex but the network produced no x/y logits")
        x = _pick_index(output.x_logits[0], config)
        y = _pick_index(output.y_logits[0], config)
        action.set_data({"x": x, "y": y})
    action.reasoning = {"source": "model"}
    return action
