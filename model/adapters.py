"""The arcengine <-> tensor boundary.

The only file in `model/` that imports from `arcengine`. Everything else
works purely in tensor space so it can be trained/tested without the game
framework. Three directions, all in this one file:

  - incoming (online): `arcengine.FrameData` -> `model.config.Observation`
  - incoming (offline): a recorded episode -> `model.config.TrainingExample`
  - outgoing: `model.config.PolicyOutput` -> `arcengine.GameAction`
"""
from __future__ import annotations

import torch
from arcengine import FrameData, GameAction

from .config import InferenceConfig, ModelConfig, Observation, PolicyOutput, TrainingExample

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


def _prev_action_index(frame: FrameData, config: ModelConfig) -> int:
    """Index into `POLICY_ACTIONS` of the action that produced `frame`, or
    `config.num_simple_actions` (the "start of episode" token) if it was
    RESET — RESET isn't a policy action, so it has no slot in that list.
    """
    action_id = frame.action_input.id
    if action_id in POLICY_ACTIONS:
        return POLICY_ACTIONS.index(action_id)
    return config.num_simple_actions


def _build_step_tensors(
    history: list[FrameData], config: ModelConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The per-step feature extraction shared by `build_observation` (a
    single growing context, online) and `build_training_example` (a whole
    recorded episode, offline). `history` must be non-empty.

    Returns unbatched CPU tensors: (grid [T,H,W] long, action_mask
    [T,num_simple_actions] bool, prev_actions [T] long, level_ups [T] long).
    """
    grid = torch.stack([frame_to_grid(f) for f in history], dim=0)

    t = len(history)
    action_mask = torch.zeros(t, config.num_simple_actions, dtype=torch.bool)
    prev_actions = torch.zeros(t, dtype=torch.long)
    level_ups = torch.zeros(t, dtype=torch.long)

    prev_levels_completed = 0
    for i, f in enumerate(history):
        available = set(f.available_actions)
        for j, action in enumerate(POLICY_ACTIONS):
            if action.value in available:
                action_mask[i, j] = True
        prev_actions[i] = _prev_action_index(f, config)
        level_ups[i] = 1 if f.levels_completed > prev_levels_completed else 0
        prev_levels_completed = f.levels_completed

    return grid, action_mask, prev_actions, level_ups


def build_observation(
    frames: list[FrameData],
    latest_frame: FrameData,
    config: ModelConfig,
    device: str = "cpu",
) -> Observation:
    """Build an `Observation` spanning the whole episode so far — every
    non-empty frame in `frames` plus `latest_frame` — capped to the most
    recent `config.max_context_len` steps as a safety bound (see that
    field's docstring). Mirrors the call signature of
    `agents.agent.Agent.choose_action`: `frames` is the history *not*
    including `latest_frame`.
    """
    history = [f for f in frames if not f.is_empty()]
    if not latest_frame.is_empty():
        history.append(latest_frame)
    if not history:
        raise ValueError("No non-empty frames available to build an observation")

    history = history[-config.max_context_len :]
    grid, action_mask, prev_actions, level_ups = _build_step_tensors(history, config)
    padding_mask = torch.ones(len(history), dtype=torch.bool)

    return Observation(
        grid=grid.unsqueeze(0).to(device),
        action_mask=action_mask.unsqueeze(0).to(device),
        prev_actions=prev_actions.unsqueeze(0).to(device),
        level_ups=level_ups.unsqueeze(0).to(device),
        padding_mask=padding_mask.unsqueeze(0).to(device),
    )


# ── incoming (offline): recorded episode -> TrainingExample ────────────


def build_training_example(
    episode: list[FrameData],
    config: ModelConfig,
    device: str = "cpu",
) -> TrainingExample:
    """Turn one complete recorded episode (as returned by
    `model.data.load_episode`/`RecordingDataset`) into teacher-forcing
    training tensors: an `Observation` over steps [0, T) plus, for each of
    those steps, the action that was actually taken next (see
    `TrainingExample`'s docstring for the target-alignment convention and
    why RESET steps are marked invalid).
    """
    episode = [f for f in episode if not f.is_empty()]
    if len(episode) < 2:
        raise ValueError(
            "Need at least 2 non-empty frames (one transition) to build a training example"
        )
    episode = episode[-(config.max_context_len + 1) :]

    history = episode[:-1]  # inputs: steps [0, T)
    grid, action_mask, prev_actions, level_ups = _build_step_tensors(history, config)
    t = len(history)

    action_targets = torch.zeros(t, dtype=torch.long)
    x_targets = torch.zeros(t, dtype=torch.long)
    y_targets = torch.zeros(t, dtype=torch.long)
    is_complex = torch.zeros(t, dtype=torch.bool)
    valid = torch.zeros(t, dtype=torch.bool)

    for i in range(t):
        next_action_input = episode[i + 1].action_input
        next_action = next_action_input.id
        if next_action in POLICY_ACTIONS:
            action_index = POLICY_ACTIONS.index(next_action)
            # Defensive: `ARCPolicyNetwork.forward` masks illegal actions to
            # -inf, so training against a target the game itself declared
            # illegal at that step (e.g. a teacher trajectory that ignored
            # `available_actions`) would demand -inf log-likelihood. Treat
            # it as an invalid target rather than let the loss go to inf.
            if bool(action_mask[i, action_index]):
                valid[i] = True
                action_targets[i] = action_index
                if next_action.is_complex():
                    is_complex[i] = True
                    x_targets[i] = int(next_action_input.data.get("x", 0))
                    y_targets[i] = int(next_action_input.data.get("y", 0))

    padding_mask = torch.ones(t, dtype=torch.bool)

    observation = Observation(
        grid=grid.unsqueeze(0).to(device),
        action_mask=action_mask.unsqueeze(0).to(device),
        prev_actions=prev_actions.unsqueeze(0).to(device),
        level_ups=level_ups.unsqueeze(0).to(device),
        padding_mask=padding_mask.unsqueeze(0).to(device),
    )
    return TrainingExample(
        observation=observation,
        action_targets=action_targets.unsqueeze(0).to(device),
        x_targets=x_targets.unsqueeze(0).to(device),
        y_targets=y_targets.unsqueeze(0).to(device),
        is_complex=is_complex.unsqueeze(0).to(device),
        valid=valid.unsqueeze(0).to(device),
    )


# ── outgoing: PolicyOutput -> GameAction ────────────────────────────────


def _pick_index(logits: torch.Tensor, config: InferenceConfig) -> int:
    """logits: [N] -> a single index, either argmax or sampled."""
    if config.deterministic:
        return int(torch.argmax(logits, dim=-1).item())
    probs = torch.softmax(logits / max(config.temperature, 1e-6), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def decode_action(output: PolicyOutput, config: InferenceConfig) -> GameAction:
    """Pick a `GameAction` (with x/y filled in for ACTION6) from raw network
    output. `output` holds a prediction at every step of the episode so far
    (see `Observation`'s docstring) — online play only wants the last one.
    Illegal actions are expected to already be masked to -inf in
    `output.action_logits` (see `ARCPolicyNetwork.forward`).
    """
    if output.action_logits.shape[0] != 1:
        raise ValueError(
            f"decode_action expects a batch size of 1, got {output.action_logits.shape[0]}"
        )
    idx = _pick_index(output.action_logits[0, -1], config)
    action = POLICY_ACTIONS[idx]
    if action.is_complex():
        x = _pick_index(output.x_logits[0, -1], config)
        y = _pick_index(output.y_logits[0, -1], config)
        action.set_data({"x": x, "y": y})
    action.reasoning = {"source": "model"}
    return action
