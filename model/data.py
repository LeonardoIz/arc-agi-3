"""Everything about getting training data in: loading recorded episodes off
disk, and a generic buffer for replay-style sampling. Both are I/O and data
structure plumbing (not modeling decisions), so both are fully implemented
— what to do with a sampled batch (the loss) is `model/train.py`.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from arcengine import FrameData

RECORDING_SUFFIX = ".recording.jsonl"


# ── Loading recorded episodes ────────────────────────────────────────────
# Produced by `agents.recorder.Recorder` whenever an agent runs with
# `record=True` — one `.recording.jsonl` file per episode.


def find_recordings(recordings_dir: str | Path) -> list[Path]:
    recordings_dir = Path(recordings_dir)
    if not recordings_dir.is_dir():
        return []
    return sorted(recordings_dir.glob(f"*{RECORDING_SUFFIX}"))


def load_episode(path: str | Path) -> list[FrameData]:
    """One `.recording.jsonl` file -> the ordered list of `FrameData` it
    recorded (the trailing scorecard event, if present, is skipped since it
    doesn't validate as `FrameData`).
    """
    frames: list[FrameData] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            try:
                frames.append(FrameData.model_validate(event["data"]))
            except Exception:
                continue  # non-frame event (e.g. the trailing scorecard record)
    return frames


class RecordingDataset:
    """Iterable over recorded episodes in a directory, each already parsed
    into a `list[FrameData]`. Kept deliberately dumb — batching, framing
    into (Observation, action, reward) tuples, and any reward shaping
    belongs in the training loop.
    """

    def __init__(self, recordings_dir: str | Path) -> None:
        self.paths = find_recordings(recordings_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> list[FrameData]:
        return load_episode(self.paths[index])

    def __iter__(self):
        for path in self.paths:
            yield load_episode(path)


# ── Replay buffer ────────────────────────────────────────────────────────
# For off-policy / RL-style training that samples random past transitions
# rather than replaying whole episodes in order.


@dataclass
class Transition:
    grid: torch.Tensor  # [T, H, W] long — one Observation's worth, unbatched
    action_mask: torch.Tensor  # [num_simple_actions] bool
    action_index: int  # index into model.adapters.POLICY_ACTIONS
    x: int | None
    y: int | None
    reward: float
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data: list[Transition] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self._data)

    def push(self, transition: Transition) -> None:
        if len(self._data) < self.capacity:
            self._data.append(transition)
        else:
            self._data[self._next] = transition
            self._next = (self._next + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self._data):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of {len(self._data)}"
            )
        return random.sample(self._data, batch_size)
