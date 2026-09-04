"""`GameSpec`: everything needed to reconstruct one generated game — pure
data, no code. Sampling one is randomized; replaying it (same seed) is
deterministic, which is what lets `procedural/solver.py` validate a spec
before it's ever written out as a game file.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .layout import Layout, generate_layout
from .mechanics import HAZARD_MECHANICS, PRIMARY_MECHANICS

# A palette of (name, color_index) roles a game can draw from. Indices are
# ARC-AGI's fixed 16-color palette; 0 is reserved for "floor/background".
FLOOR_COLOR = 0
ROLE_COLOR_POOL = list(range(1, 16))


@dataclass(frozen=True)
class GameSpec:
    seed: int
    width: int
    height: int
    primary_key: str
    primary_params: dict
    hazard_key: str | None
    hazard_params: dict
    num_levels: int
    role_colors: dict[str, int]  # entity role name -> color index, this spec's palette

    @property
    def directional(self) -> bool:
        return PRIMARY_MECHANICS[self.primary_key].directional


def sample_spec(rng: random.Random, seed: int) -> GameSpec:
    primary_key = rng.choice(list(PRIMARY_MECHANICS.keys()))
    mechanic = PRIMARY_MECHANICS[primary_key]

    width = rng.randint(6, 12)
    height = rng.randint(6, 12)

    primary_params = dict(mechanic.default_params)
    # Light randomization of numeric params, kept within sane bounds so a
    # sampled instance is still solvable-sized (checked for real by the
    # solver later, not assumed here).
    if "num_items" in primary_params:
        primary_params["num_items"] = rng.randint(2, 4)
    if "num_states" in primary_params:
        primary_params["num_states"] = rng.randint(2, 4)
    if "min_steps" in primary_params:
        primary_params["min_steps"] = rng.randint(8, 20)
    if "num_checkpoints" in primary_params:
        primary_params["num_checkpoints"] = rng.randint(2, 4)
    if "num_targets" in primary_params:
        primary_params["num_targets"] = rng.randint(2, 5)
    if "num_pairs" in primary_params:
        primary_params["num_pairs"] = rng.randint(1, 3)

    hazard_key = None
    hazard_params: dict = {}
    if mechanic.directional and rng.random() < 0.4:
        hazard_key = rng.choice(list(HAZARD_MECHANICS.keys()))
        hazard_params = {"num_hazards": rng.randint(1, 3), "patrol_length": rng.randint(4, 8)}

    num_levels = rng.randint(1, 3)

    roles = set(mechanic.roles)
    colors = rng.sample(ROLE_COLOR_POOL, min(len(roles) + 2, len(ROLE_COLOR_POOL)))
    role_colors = {role: colors[i % len(colors)] for i, role in enumerate(sorted(roles))}
    role_colors["actor"] = colors[-1]
    role_colors["hazard"] = colors[-2] if len(colors) > 1 else colors[0]

    return GameSpec(
        seed=seed,
        width=width,
        height=height,
        primary_key=primary_key,
        primary_params=primary_params,
        hazard_key=hazard_key,
        hazard_params=hazard_params,
        num_levels=num_levels,
        role_colors=role_colors,
    )


def build_layout(spec: GameSpec) -> Layout:
    rng = random.Random(spec.seed)
    return generate_layout(spec.width, spec.height, open_fraction=0.55, rng=rng)


# Shared by `procedural/engine.py` (which builds all `spec.num_levels` real
# levels) and `procedural/solver.py` (which must validate the same levels)
# so the two never derive per-level seeds differently.
LEVEL_SEED_STRIDE = 7919


def level_seed(spec: GameSpec, level_index: int) -> int:
    return spec.seed + level_index * LEVEL_SEED_STRIDE


def build_level_layout(spec: GameSpec, level_index: int) -> Layout:
    rng = random.Random(level_seed(spec, level_index))
    return generate_layout(spec.width, spec.height, open_fraction=0.55, rng=rng)
