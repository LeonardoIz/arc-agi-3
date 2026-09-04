"""The mechanic library: composable, pure state-transition rules.

A `Mechanic` bundles three pure functions — `place` (choose entity
positions in a `Layout`), `init_state` (build the starting `SimState`), and
`step` (one action -> a new `SimState`) — plus whether it's directional
(ACTION1-4 movement) or click-based (ACTION6). Nothing here touches
`arcengine`; `procedural/engine.py` is the only place that renders a
`SimState` into real sprites, and `procedural/solver.py` calls these same
functions headless for BFS solvability checks. One implementation, two
consumers — the solver can't drift from what the real game does.

Two families:
  - **Primary mechanics** (`PRIMARY_MECHANICS`) — each defines a complete
    core loop (how input moves/edits the world + what winning means).
    Directional primaries have an `actor`; click primaries manipulate grid
    state directly with no actor to endanger.
  - **Hazard modifiers** (`HAZARD_MECHANICS`) — optional, stack onto any
    *directional* primary; they only add a lose-condition, they don't
    change the core loop.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .layout import Layout, reachable_from
from .state import Cell, SimState

# ACTION1-4 -> (dx, dy). Arbitrary but fixed; the model has to learn it like
# it learns everything else about an unfamiliar game (C1/C3).
DIRECTIONS: dict[int, Cell] = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


@dataclass(frozen=True)
class ActionEvent:
    """Decoupled from `arcengine.GameAction` so this module stays pure
    Python — `procedural/engine.py` translates both directions.
    """

    action_id: int  # 1..7 (never RESET/0 — the agent/engine handles that)
    xy: Cell | None = None  # only set for ACTION6


Entities = dict[str, object]  # role name -> Cell, or list[Cell] for multi-instance roles
PlaceFn = Callable[[Layout, random.Random, dict], Entities]
InitFn = Callable[[Layout, Entities, dict], SimState]
StepFn = Callable[[SimState, Layout, Entities, dict, ActionEvent], SimState]


@dataclass(frozen=True)
class Mechanic:
    key: str
    directional: bool  # False = click-based (ACTION6 only)
    roles: tuple[str, ...]  # entity roles `place` is expected to fill
    place: PlaceFn
    init_state: InitFn
    step: StepFn
    default_params: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.default_params is None:
            object.__setattr__(self, "default_params", {})


def _sample_cells(layout: Layout, rng: random.Random, n: int, exclude: set[Cell]) -> list[Cell]:
    pool = [c for c in layout.open_cells if c not in exclude]
    if len(pool) < n:
        raise ValueError(f"Layout has only {len(pool)} free open cells, need {n}")
    return rng.sample(pool, n)


def _farthest_reachable_pair(layout: Layout, rng: random.Random) -> tuple[Cell, Cell]:
    """A start cell and a goal cell that's actually reachable from it — the
    goal is sampled from the far half of the BFS-distance-sorted reachable
    set, so games aren't trivially "goal is one step away".
    """
    start = rng.choice(layout.open_cells)
    reachable = sorted(reachable_from(layout, start) - {start})
    if not reachable:
        return start, start
    far_half = reachable[len(reachable) // 2 :] or reachable
    goal = rng.choice(far_half)
    return start, goal


def _clamp_move(cell: Cell, d: Cell, layout: Layout, blocked: set[Cell]) -> Cell:
    nxt = (cell[0] + d[0], cell[1] + d[1])
    if layout.is_open(nxt) and nxt not in blocked:
        return nxt
    return cell


# ── Directional primaries ───────────────────────────────────────────────


def _walker_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    start, goal = _farthest_reachable_pair(layout, rng)
    return {"actor": start, "goal": goal}


def _walker_init(layout: Layout, entities: Entities, params: dict) -> SimState:
    return SimState(actor=entities["actor"])  # type: ignore[arg-type]


def _walker_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    d = DIRECTIONS.get(event.action_id)
    if d is None or state.actor is None:
        return state
    new_pos = _clamp_move(state.actor, d, layout, blocked=set())
    terminal = "WIN" if new_pos == entities["goal"] else None
    return state.replace(actor=new_pos, step_count=state.step_count + 1, terminal=terminal)


def _sokoban_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    start = rng.choice(layout.open_cells)
    reachable = reachable_from(layout, start) - {start}
    block = rng.choice(sorted(reachable)) if reachable else start
    reachable2 = reachable_from(layout, block) - {block, start}
    target = rng.choice(sorted(reachable2)) if reachable2 else block
    return {"actor": start, "block": block, "target": target}


def _sokoban_init(layout: Layout, entities: Entities, params: dict) -> SimState:
    return SimState(actor=entities["actor"], cell_state={entities["block"]: 1})  # type: ignore[dict-item]


def _sokoban_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    d = DIRECTIONS.get(event.action_id)
    if d is None or state.actor is None:
        return state
    blocks = {c for c, v in state.cell_state.items() if v == 1}
    dest = (state.actor[0] + d[0], state.actor[1] + d[1])
    if not layout.is_open(dest):
        return state
    if dest in blocks:
        push_to = (dest[0] + d[0], dest[1] + d[1])
        if not layout.is_open(push_to) or push_to in blocks:
            return state  # blocked push
        new_cell_state = {c: v for c, v in state.cell_state.items() if c != dest}
        new_cell_state[push_to] = 1
        terminal = "WIN" if push_to == entities["target"] else None
        return state.replace(
            actor=dest, cell_state=new_cell_state, step_count=state.step_count + 1, terminal=terminal
        )
    return state.replace(actor=dest, step_count=state.step_count + 1)


def _ice_slide_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    return _walker_place(layout, rng, params)


def _ice_slide_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    d = DIRECTIONS.get(event.action_id)
    if d is None or state.actor is None:
        return state
    pos = state.actor
    while True:
        nxt = (pos[0] + d[0], pos[1] + d[1])
        if not layout.is_open(nxt):
            break
        pos = nxt
        if pos == entities["goal"]:
            break
    terminal = "WIN" if pos == entities["goal"] else None
    return state.replace(actor=pos, step_count=state.step_count + 1, terminal=terminal)


def _teleport_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    start, goal = _farthest_reachable_pair(layout, rng)
    pair = _sample_cells(layout, rng, 2, exclude={start, goal})
    return {"actor": start, "goal": goal, "teleport_a": pair[0], "teleport_b": pair[1]}


def _teleport_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    d = DIRECTIONS.get(event.action_id)
    if d is None or state.actor is None:
        return state
    new_pos = _clamp_move(state.actor, d, layout, blocked=set())
    if new_pos == entities["teleport_a"]:
        new_pos = entities["teleport_b"]  # type: ignore[assignment]
    elif new_pos == entities["teleport_b"]:
        new_pos = entities["teleport_a"]  # type: ignore[assignment]
    terminal = "WIN" if new_pos == entities["goal"] else None
    return state.replace(actor=new_pos, step_count=state.step_count + 1, terminal=terminal)


def _one_way_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    start, goal = _farthest_reachable_pair(layout, rng)
    gates = _sample_cells(layout, rng, min(3, max(1, len(layout.open_cells) // 8)), exclude={start, goal})
    gate_dirs = {c: rng.choice(list(DIRECTIONS.values())) for c in gates}
    return {"actor": start, "goal": goal, "gates": gate_dirs}  # type: ignore[dict-item]


def _one_way_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    d = DIRECTIONS.get(event.action_id)
    if d is None or state.actor is None:
        return state
    gates: dict[Cell, Cell] = entities["gates"]  # type: ignore[assignment]
    if state.actor in gates and gates[state.actor] != d:
        return state.replace(step_count=state.step_count + 1)  # wrong-direction crossing blocked
    new_pos = _clamp_move(state.actor, d, layout, blocked=set())
    terminal = "WIN" if new_pos == entities["goal"] else None
    return state.replace(actor=new_pos, step_count=state.step_count + 1, terminal=terminal)


def _collect_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    start, goal = _farthest_reachable_pair(layout, rng)
    n = params.get("num_items", 3)
    items = _sample_cells(layout, rng, n, exclude={start, goal})
    return {"actor": start, "goal": goal, "items": items}


def _collect_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    d = DIRECTIONS.get(event.action_id)
    if d is None or state.actor is None:
        return state
    new_pos = _clamp_move(state.actor, d, layout, blocked=set())
    collected = state.collected
    if new_pos in entities["items"] and new_pos not in collected:  # type: ignore[operator]
        collected = collected | {new_pos}
    terminal = None
    if new_pos == entities["goal"] and len(collected) == len(entities["items"]):  # type: ignore[arg-type]
        terminal = "WIN"
    return state.replace(actor=new_pos, collected=collected, step_count=state.step_count + 1, terminal=terminal)


def _state_match_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    start, goal = _farthest_reachable_pair(layout, rng)
    n_states = params.get("num_states", 3)
    target_state = rng.randrange(n_states)
    return {"actor": start, "goal": goal, "target_state": target_state, "num_states": n_states}


def _state_match_init(layout: Layout, entities: Entities, params: dict) -> SimState:
    return SimState(actor=entities["actor"], cell_state={"actor_state": 0})  # type: ignore[dict-item]


def _state_match_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    actor_state = state.cell_state.get("actor_state", 0)  # type: ignore[call-overload]
    if event.action_id == 5:  # ACTION5 cycles the actor's own state
        actor_state = (actor_state + 1) % entities["num_states"]  # type: ignore[operator]
        # Check WIN here too, not just on movement: the actor may already be
        # standing on the goal and only need its state to match.
        terminal = "WIN" if state.actor == entities["goal"] and actor_state == entities["target_state"] else None
        return state.replace(
            cell_state={**state.cell_state, "actor_state": actor_state},
            step_count=state.step_count + 1,
            terminal=terminal,
        )
    d = DIRECTIONS.get(event.action_id)
    if d is None or state.actor is None:
        return state
    new_pos = _clamp_move(state.actor, d, layout, blocked=set())
    terminal = "WIN" if new_pos == entities["goal"] and actor_state == entities["target_state"] else None
    return state.replace(actor=new_pos, step_count=state.step_count + 1, terminal=terminal)


def _survive_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    start, goal = _farthest_reachable_pair(layout, rng)
    return {"actor": start, "goal": goal, "min_steps": params.get("min_steps", 15)}


def _survive_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    d = DIRECTIONS.get(event.action_id)
    if d is None or state.actor is None:
        return state
    new_pos = _clamp_move(state.actor, d, layout, blocked=set())
    step_count = state.step_count + 1
    terminal = "WIN" if new_pos == entities["goal"] and step_count >= entities["min_steps"] else None
    return state.replace(actor=new_pos, step_count=step_count, terminal=terminal)


def _checkpoints_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    start = rng.choice(layout.open_cells)
    n = params.get("num_checkpoints", 3)
    order = _sample_cells(layout, rng, n, exclude={start})
    return {"actor": start, "checkpoints": order}


def _checkpoints_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    d = DIRECTIONS.get(event.action_id)
    if d is None or state.actor is None:
        return state
    new_pos = _clamp_move(state.actor, d, layout, blocked=set())
    checkpoints: list[Cell] = entities["checkpoints"]  # type: ignore[assignment]
    idx = state.checkpoint_index
    if idx < len(checkpoints) and new_pos == checkpoints[idx]:
        idx += 1
    terminal = "WIN" if idx == len(checkpoints) else None
    return state.replace(
        actor=new_pos, checkpoint_index=idx, step_count=state.step_count + 1, terminal=terminal
    )


# ── Click-based primaries (no actor; the whole grid is the puzzle) ──────


def _paint_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    n_targets = params.get("num_targets", max(3, len(layout.open_cells) // 6))
    targets = _sample_cells(layout, rng, min(n_targets, len(layout.open_cells)), exclude=set())
    kernel = params.get("kernel", [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)])
    return {"targets": targets, "kernel": kernel}


def _paint_init(layout: Layout, entities: Entities, params: dict) -> SimState:
    return SimState(cell_state={c: 0 for c in layout.open_cells})


def _paint_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    if event.action_id != 6 or event.xy is None or not layout.is_open(event.xy):
        return state
    kernel: list[Cell] = entities["kernel"]  # type: ignore[assignment]
    cell_state = dict(state.cell_state)
    for dx, dy in kernel:
        c = (event.xy[0] + dx, event.xy[1] + dy)
        if layout.is_open(c):
            cell_state[c] = 1 - cell_state.get(c, 0)
    targets: list[Cell] = entities["targets"]  # type: ignore[assignment]
    terminal = "WIN" if all(cell_state.get(c, 0) == 1 for c in targets) else None
    return state.replace(cell_state=cell_state, step_count=state.step_count + 1, terminal=terminal)


def _rotate_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    n_states = params.get("num_states", 3)
    n_targets = params.get("num_targets", max(2, len(layout.open_cells) // 8))
    targets = _sample_cells(layout, rng, min(n_targets, len(layout.open_cells)), exclude=set())
    goal_states = {c: rng.randrange(1, n_states) for c in targets}  # never already-satisfied (0)
    return {"targets": targets, "goal_states": goal_states, "num_states": n_states}


def _rotate_init(layout: Layout, entities: Entities, params: dict) -> SimState:
    return SimState(cell_state={c: 0 for c in entities["targets"]})  # type: ignore[union-attr]


def _rotate_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    if event.action_id != 6 or event.xy is None or event.xy not in state.cell_state:
        return state
    cell_state = dict(state.cell_state)
    cell_state[event.xy] = (cell_state[event.xy] + 1) % entities["num_states"]  # type: ignore[operator]
    goal_states: dict[Cell, int] = entities["goal_states"]  # type: ignore[assignment]
    terminal = "WIN" if all(cell_state[c] == g for c, g in goal_states.items()) else None
    return state.replace(cell_state=cell_state, step_count=state.step_count + 1, terminal=terminal)


def _key_lock_place(layout: Layout, rng: random.Random, params: dict) -> Entities:
    n_pairs = params.get("num_pairs", 2)
    cells = _sample_cells(layout, rng, n_pairs * 2, exclude=set())
    keys = cells[:n_pairs]
    doors = cells[n_pairs:]
    return {"keys": keys, "doors": doors}


def _key_lock_step(
    state: SimState, layout: Layout, entities: Entities, params: dict, event: ActionEvent
) -> SimState:
    if event.action_id != 6 or event.xy is None:
        return state
    keys: list[Cell] = entities["keys"]  # type: ignore[assignment]
    doors: list[Cell] = entities["doors"]  # type: ignore[assignment]
    carrying = state.carrying
    open_doors = state.open_doors
    if event.xy in keys and event.xy not in state.collected:
        carrying = f"key_{keys.index(event.xy)}"
        return state.replace(
            carrying=carrying, collected=state.collected | {event.xy}, step_count=state.step_count + 1
        )
    if event.xy in doors and carrying == f"key_{doors.index(event.xy)}":
        open_doors = open_doors | {event.xy}
        carrying = None
    terminal = "WIN" if len(open_doors) == len(doors) else None
    return state.replace(
        carrying=carrying, open_doors=open_doors, step_count=state.step_count + 1, terminal=terminal
    )


PRIMARY_MECHANICS: dict[str, Mechanic] = {
    "walker_goal": Mechanic("walker_goal", True, ("actor", "goal"), _walker_place, _walker_init, _walker_step),
    "sokoban": Mechanic("sokoban", True, ("actor", "block", "target"), _sokoban_place, _sokoban_init, _sokoban_step),
    "ice_slide": Mechanic("ice_slide", True, ("actor", "goal"), _ice_slide_place, _walker_init, _ice_slide_step),
    "teleport_maze": Mechanic(
        "teleport_maze", True, ("actor", "goal", "teleport_a", "teleport_b"),
        _teleport_place, _walker_init, _teleport_step,
    ),
    "one_way_maze": Mechanic(
        "one_way_maze", True, ("actor", "goal", "gates"), _one_way_place, _walker_init, _one_way_step
    ),
    "collect_then_exit": Mechanic(
        "collect_then_exit", True, ("actor", "goal", "items"),
        _collect_place, _walker_init, _collect_step, {"num_items": 3},
    ),
    "state_match_goal": Mechanic(
        "state_match_goal", True, ("actor", "goal", "target_state"),
        _state_match_place, _state_match_init, _state_match_step, {"num_states": 3},
    ),
    "survive_then_exit": Mechanic(
        "survive_then_exit", True, ("actor", "goal", "min_steps"),
        _survive_place, _walker_init, _survive_step, {"min_steps": 15},
    ),
    "checkpoints_in_order": Mechanic(
        "checkpoints_in_order", True, ("actor", "checkpoints"),
        _checkpoints_place, _walker_init, _checkpoints_step, {"num_checkpoints": 3},
    ),
    "paint_propagate": Mechanic(
        "paint_propagate", False, ("targets", "kernel"), _paint_place, _paint_init, _paint_step
    ),
    "rotate_to_match": Mechanic(
        "rotate_to_match", False, ("targets", "goal_states"),
        _rotate_place, _rotate_init, _rotate_step, {"num_states": 3},
    ),
    "key_lock": Mechanic(
        "key_lock", False, ("keys", "doors"), _key_lock_place, _paint_init, _key_lock_step, {"num_pairs": 2}
    ),
}


# ── Hazard modifiers (stack onto any directional primary) ──────────────
# A hazard only ever adds a LOSE condition on top of whatever the primary
# mechanic already computed; it never overrides a WIN the primary reports.


@dataclass(frozen=True)
class HazardMechanic:
    key: str
    place: Callable[[Layout, random.Random, dict, Entities], list[Cell]]
    apply: Callable[[SimState, Layout, list[Cell], dict], SimState]


def _static_hazard_place(layout: Layout, rng: random.Random, params: dict, entities: Entities) -> list[Cell]:
    exclude = {v for v in entities.values() if isinstance(v, tuple)}
    n = params.get("num_hazards", 2)
    return _sample_cells(layout, rng, n, exclude=exclude)  # type: ignore[arg-type]


def _static_hazard_apply(state: SimState, layout: Layout, cells: list[Cell], params: dict) -> SimState:
    if state.terminal == "WIN" or state.actor is None:
        return state
    if state.actor in cells:
        return state.replace(terminal="LOSE")
    return state


def _decaying_apply(state: SimState, layout: Layout, cells: list[Cell], params: dict) -> SimState:
    """Cells in `cells` collapse (become lethal) once visited once already."""
    if state.terminal == "WIN" or state.actor is None:
        return state
    visited_key = "decayed"
    decayed = {c for c, v in state.cell_state.items() if v == -1}
    if state.actor in cells:
        if state.actor in decayed:
            return state.replace(terminal="LOSE")
        new_cell_state = {**state.cell_state, state.actor: -1}
        return state.replace(cell_state=new_cell_state)
    return state


def _moving_hazard_apply(state: SimState, layout: Layout, cells: list[Cell], params: dict) -> SimState:
    """`cells` is a fixed patrol path; the hazard occupies
    `cells[step_count % len(cells)]` — deterministic, so the solver can
    plan around it exactly like the real game will.
    """
    if state.terminal == "WIN" or state.actor is None or not cells:
        return state
    pos = cells[state.step_count % len(cells)]
    if state.actor == pos:
        return state.replace(terminal="LOSE")
    return state


def _patrol_place(layout: Layout, rng: random.Random, params: dict, entities: Entities) -> list[Cell]:
    start = rng.choice(layout.open_cells)
    path = [start]
    cur = start
    length = params.get("patrol_length", 6)
    for _ in range(length - 1):
        options = [c for c in layout.neighbors4(cur) if layout.is_open(c)]
        if not options:
            break
        cur = rng.choice(options)
        path.append(cur)
    return path


HAZARD_MECHANICS: dict[str, HazardMechanic] = {
    "static_hazard": HazardMechanic("static_hazard", _static_hazard_place, _static_hazard_apply),
    "decaying_tiles": HazardMechanic("decaying_tiles", _static_hazard_place, _decaying_apply),
    "moving_hazard": HazardMechanic("moving_hazard", _patrol_place, _moving_hazard_apply),
}
