"""Generation-time solvability check: BFS over `SimState` using the exact
same mechanic `step` functions the real game uses (see `mechanics.py`'s
module docstring for why that matters).

This is *not* search at inference — C5/C9 forbid the deployed agent from
simulating/planning. This runs once, offline, while generating training
environments, to reject specs that turn out to be unsolvable or degenerate,
the same way a level designer playtests a level before shipping it. It's
squarely inside C6 ("entrenamiento ≠ inferencia"), and arguably even more
permissive than the "privileged simulator teacher" C6 already allows, since
it never touches the agent's own decision-making at all.
"""
from __future__ import annotations

import random
from collections import deque

from .layout import Cell, generate_layout
from .mechanics import HAZARD_MECHANICS, PRIMARY_MECHANICS, ActionEvent
from .spec import GameSpec, level_seed
from .state import SimState

MAX_BFS_STATES = 20_000
MAX_DEPTH = 150


def _state_key(state: SimState) -> tuple:
    # `cell_state` keys are usually `Cell` tuples, but `state_match_goal`
    # uses the string key "actor_state" — and a hazard like `decaying_tiles`
    # can add `Cell`-keyed entries on top of any primary's cell_state, so a
    # single instance can mix key types. Sort by `repr` (always comparable)
    # rather than the keys themselves.
    cell_items = tuple(sorted(state.cell_state.items(), key=lambda kv: repr(kv[0])))
    return (
        state.actor,
        state.collected,
        state.open_doors,
        cell_items,
        state.step_count,
        state.carrying,
        state.checkpoint_index,
    )


def _candidate_actions(spec: GameSpec, entities: dict) -> list[ActionEvent]:
    mechanic = PRIMARY_MECHANICS[spec.primary_key]
    if mechanic.directional:
        ids = [1, 2, 3, 4] + ([5] if spec.primary_key == "state_match_goal" else [])
        return [ActionEvent(a) for a in ids]
    cells: set[Cell] = set()
    for role in ("targets", "keys", "doors"):
        value = entities.get(role)
        if isinstance(value, list):
            cells.update(value)  # type: ignore[arg-type]
    return [ActionEvent(6, xy=c) for c in sorted(cells)]


def solve(spec: GameSpec, seed_for_level: int | None = None) -> list[ActionEvent] | None:
    """Returns a winning action sequence (shortest, since BFS) for one
    level, or None if no solution was found within the search bounds —
    either genuinely unsolvable or too large to prove so cheaply; both are
    treated as "reject this spec" by the generator.

    `seed_for_level` defaults to level 0's seed (`spec.seed` itself); pass
    `procedural.spec.level_seed(spec, i)` to validate level `i` of a
    multi-level spec — see `solve_all_levels`, which every generated game
    should actually be checked with, not this function directly.
    """
    if seed_for_level is None:
        seed_for_level = level_seed(spec, 0)
    layout = generate_layout(spec.width, spec.height, open_fraction=0.55, rng=random.Random(seed_for_level))
    mechanic = PRIMARY_MECHANICS[spec.primary_key]
    entities = mechanic.place(layout, random.Random(seed_for_level ^ 0x9E3779B1), spec.primary_params)
    init_state = mechanic.init_state(layout, entities, spec.primary_params)

    hazard = HAZARD_MECHANICS[spec.hazard_key] if spec.hazard_key else None
    hazard_cells: list[Cell] = []
    if hazard is not None:
        hazard_cells = hazard.place(
            layout, random.Random(seed_for_level ^ 0x2545F491), spec.hazard_params, entities
        )

    def apply(state: SimState, event: ActionEvent) -> SimState:
        new_state = mechanic.step(state, layout, entities, spec.primary_params, event)
        if hazard is not None:
            new_state = hazard.apply(new_state, layout, hazard_cells, spec.hazard_params)
        return new_state

    candidates = _candidate_actions(spec, entities)
    if init_state.terminal == "WIN":
        return []

    visited = {_state_key(init_state)}
    frontier: deque[tuple[SimState, list[ActionEvent]]] = deque([(init_state, [])])
    expanded = 0
    while frontier:
        state, path = frontier.popleft()
        if len(path) >= MAX_DEPTH:
            continue
        expanded += 1
        if expanded > MAX_BFS_STATES:
            return None
        for event in candidates:
            nxt = apply(state, event)
            # Check WIN before the visited-dedup lookup, unconditionally: a
            # dedup key that omits `terminal` (see `_state_key`) could
            # otherwise let an earlier non-winning path to the "same" state
            # claim the key first and shadow a winning one reached via a
            # different action order to that same key.
            if nxt.terminal == "WIN":
                return path + [event]
            if nxt.terminal == "LOSE":
                continue
            key = _state_key(nxt)
            if key in visited:
                continue
            visited.add(key)
            frontier.append((nxt, path + [event]))
    return None


def solve_all_levels(spec: GameSpec) -> list[list[ActionEvent]] | None:
    """The real generator-time filter: a spec is only accepted if *every*
    one of its `spec.num_levels` levels is solvable — `solve()` alone only
    checks level 0, and `procedural/engine.py` builds every level
    independently at the same per-level seeds this iterates over.
    """
    solutions = []
    for i in range(spec.num_levels):
        sol = solve(spec, seed_for_level=level_seed(spec, i))
        if sol is None:
            return None
        solutions.append(sol)
    return solutions
