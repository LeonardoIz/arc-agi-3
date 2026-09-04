"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

`scripts/build_notebook.py` splices the contents of this file into the
Kaggle submission notebook, so your local dev loop and your Kaggle
submission stay in lock-step:

    [edit my_agent.py] → [make play-local] → [make submit]

The default body below is a port of the Stochastic Goose / random_agent
sample — a known-good baseline that produces a valid submission and
proves your end-to-end pipeline works. Replace `choose_action` with your
real strategy.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`.
  - Class must be named `MyAgent` (the notebook's __init__.py registers it).
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.

Playing with a trained PyTorch model instead of the random baseline:
  Set `MODEL_CHECKPOINT` below (or the `ARC_MODEL_CHECKPOINT` env var) to a
  checkpoint produced by `model/train.py` (see `docs/model.md` for how the
  model package is laid out and used). If the checkpoint can't be found or
  loaded, this agent silently falls back to the random strategy below — so
  it's always safe to leave `MODEL_CHECKPOINT` set. Model code lives
  entirely under `model/` and is never touched from here.
"""
from __future__ import annotations

import os
import random
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent

try:
    from model.inference import ModelPolicy
except ImportError:
    # torch (or the model/ package) isn't available in this environment —
    # fall back to the random strategy below.
    ModelPolicy = None  # type: ignore[assignment,misc]


class MyAgent(Agent):
    """Picks legal actions uniformly at random. Replace with your strategy."""

    # Upper bound on actions per game; the framework also enforces global limits.
    MAX_ACTIONS = 80

    # Path to a trained model checkpoint. Leave as None (or unset the env
    # var) to always use the random baseline. See the module docstring.
    MODEL_CHECKPOINT: str | None = os.environ.get("ARC_MODEL_CHECKPOINT")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Seed per game_id so replays from the same game are reproducible but
        # different games explore independently.
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

        self._policy = None
        if ModelPolicy is not None and self.MODEL_CHECKPOINT:
            self._policy = ModelPolicy.try_from_checkpoint(self.MODEL_CHECKPOINT)

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER — we want to RESET and retry.
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # First call or after a death → reset the level.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        if self._policy is not None:
            return self._policy.act(frames, latest_frame)

        # ── Per-game strategy fork ───────────────────────────────────────────
        # By default every game uses the same uniformly-random strategy in the
        # `else` branch below. This `if` shows ONE example of giving a single
        # game its own heuristic: on LS20 we bias the random pick so ACTION4
        # is twice as likely as any other action. Add more `elif` branches to
        # specialize other games.
        #
        # `self.game_id` is set by the framework. It may be the short id
        # ("ls20") or include a version suffix ("ls20-9607627b"), so we
        # compare on the prefix to be safe.
        candidate_actions = [a for a in GameAction if a is not GameAction.RESET]
        if self.game_id.split("-")[0] == "ls20":
            weights = [2 if a is GameAction.ACTION4 else 1 for a in candidate_actions]
            action = random.choices(candidate_actions, weights=weights, k=1)[0]
        else:
            action = random.choice(candidate_actions)
        # ────────────────────────────────────────────────────────────────────

        if action.is_complex():
            # ACTION6 takes (x, y) coordinates on a 64×64 grid.
            action.set_data(
                {"x": random.randint(0, 63), "y": random.randint(0, 63)}
            )
            action.reasoning = {"why": "random complex action"}
        else:
            action.reasoning = f"random simple action: {action.value}"
        return action
