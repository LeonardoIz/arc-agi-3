"""`ProceduralGame`: the one `arcengine.ARCBaseGame` subclass every
generated game file instantiates (with a different `SPEC` class attribute).

This is the *only* file in `procedural/` that imports `arcengine` — same
seam discipline as `model/adapters.py`. It's a pure rendering/translation
layer: all actual game logic lives in `mechanics.py` as pure `SimState`
transitions; this class's `step()` just applies one action via the
mechanic (+ hazard, if any) and syncs sprite position/color to match the
resulting `SimState`. Every piece of state that affects the win condition
is rendered as a real color/position change — a model that only ever sees
pixels (C1) still gets a fair, learnable game.

Sprites are tracked by *identity*, not by a shared "sprite at this cell"
dict: the actor can walk onto a cell that already holds a static marker
(a goal, an item, a checkpoint), and a shared dict keyed by current
position would silently lose one of the two references the moment that
happens. `actor_sprite` and `block_sprite` (Sokoban) are tracked directly;
`static_sprites` is keyed by each marker's *fixed* spawn cell, which never
collides with anything since only the actor and the block ever move.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, Level, Sprite

from .layout import Layout, generate_layout
from .mechanics import HAZARD_MECHANICS, PRIMARY_MECHANICS, ActionEvent, Entities
from .spec import GameSpec, level_seed
from .state import Cell, SimState

FLOOR_COLOR = 0


def _is_cell(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value)


@dataclass
class LevelContext:
    layout: Layout
    entities: Entities
    hazard_cells: list[Cell]
    init_state: SimState
    actor_sprite: Sprite | None
    block_sprite: Sprite | None  # Sokoban only
    static_sprites: dict[Cell, Sprite]  # keyed by each marker's fixed spawn cell
    hazard_sprite: Sprite | None  # single reusable sprite for a *moving* hazard only


class ProceduralGame(ARCBaseGame):
    """Subclassed per generated game file, which sets `SPEC` as a class
    attribute (a `GameSpec`) before defining nothing else.
    """

    SPEC: GameSpec

    def __init__(self, seed: int = 0) -> None:
        spec = self.SPEC
        self._mechanic = PRIMARY_MECHANICS[spec.primary_key]
        self._hazard = HAZARD_MECHANICS[spec.hazard_key] if spec.hazard_key else None
        self._cell_px = max(1, 64 // max(spec.width, spec.height))
        self._sim_state: SimState | None = None
        self._level_ctx: list[LevelContext] = []

        levels: list[Level] = []
        for i in range(spec.num_levels):
            ctx, level = self._build_level(spec, level_seed(spec, i))
            self._level_ctx.append(ctx)
            levels.append(level)

        if self._mechanic.directional:
            action_ids = [1, 2, 3, 4] + ([5] if spec.primary_key == "state_match_goal" else [])
        else:
            action_ids = [6]

        camera = Camera(
            width=spec.width * self._cell_px,
            height=spec.height * self._cell_px,
            background=FLOOR_COLOR,
            letter_box=FLOOR_COLOR,
        )
        super().__init__(
            game_id=spec.primary_key,
            levels=levels,
            camera=camera,
            win_score=spec.num_levels,
            available_actions=action_ids,
            seed=seed,
        )

    # ── level construction ──────────────────────────────────────────────

    def _square(self, color: int, cell: Cell, layer: int) -> Sprite:
        px = self._cell_px
        pixels = np.full((px, px), color, dtype=np.int8)
        return Sprite(
            pixels=pixels, x=cell[0] * px, y=cell[1] * px, layer=layer, blocking=BlockingMode.NOT_BLOCKED
        )

    def _build_level(self, spec: GameSpec, seed_for_level: int) -> tuple[LevelContext, Level]:
        layout = generate_layout(spec.width, spec.height, open_fraction=0.55, rng=random.Random(seed_for_level))
        entities = self._mechanic.place(layout, random.Random(seed_for_level ^ 0x9E3779B1), spec.primary_params)
        init_state = self._mechanic.init_state(layout, entities, spec.primary_params)

        hazard_cells: list[Cell] = []
        if self._hazard is not None:
            hazard_cells = self._hazard.place(
                layout, random.Random(seed_for_level ^ 0x2545F491), spec.hazard_params, entities
            )

        sprites: list[Sprite] = [self._square(spec.role_colors.get("wall", 1), c, 0) for c in layout.walls]

        static_sprites: dict[Cell, Sprite] = {}
        actor_sprite: Sprite | None = None
        block_sprite: Sprite | None = None

        if spec.primary_key == "paint_propagate":
            # The whole interactive grid is the puzzle, not just the target
            # cells — every open cell needs to show its own toggle state,
            # and none of them ever coincide with a moving actor (there is
            # none in this mechanic), so a fixed-cell dict is safe here too.
            for c in layout.open_cells:
                s = self._square(FLOOR_COLOR, c, 1)
                sprites.append(s)
                static_sprites[c] = s
        else:
            for role, value in entities.items():
                color = spec.role_colors.get(role, spec.role_colors.get("actor", 8))
                if role == "actor" and _is_cell(value):
                    actor_sprite = self._square(color, value, 2)  # type: ignore[arg-type]
                    sprites.append(actor_sprite)
                elif role == "block" and _is_cell(value):
                    block_sprite = self._square(color, value, 2)  # type: ignore[arg-type]
                    sprites.append(block_sprite)
                elif _is_cell(value):
                    s = self._square(color, value, 1)  # type: ignore[arg-type]
                    sprites.append(s)
                    static_sprites[value] = s  # type: ignore[index]
                elif isinstance(value, list) and value and _is_cell(value[0]):
                    for c in value:  # type: ignore[assignment]
                        s = self._square(color, c, 1)
                        sprites.append(s)
                        static_sprites[c] = s
                elif isinstance(value, dict) and value and _is_cell(next(iter(value))):
                    for c in value:  # type: ignore[assignment]
                        s = self._square(color, c, 1)
                        sprites.append(s)
                        static_sprites[c] = s

        hazard_sprite: Sprite | None = None
        hazard_color = spec.role_colors.get("hazard", 2)
        if self._hazard is not None:
            if spec.hazard_key == "moving_hazard":
                start = hazard_cells[0] if hazard_cells else next(iter(layout.open_cells))
                hazard_sprite = self._square(hazard_color, start, 2)
                sprites.append(hazard_sprite)
            else:
                for c in hazard_cells:
                    s = self._square(hazard_color, c, 1)
                    sprites.append(s)
                    static_sprites[c] = s

        level = Level(sprites=sprites, grid_size=(spec.width * self._cell_px, spec.height * self._cell_px))
        ctx = LevelContext(
            layout=layout,
            entities=entities,
            hazard_cells=hazard_cells,
            init_state=init_state,
            actor_sprite=actor_sprite,
            block_sprite=block_sprite,
            static_sprites=static_sprites,
            hazard_sprite=hazard_sprite,
        )
        return ctx, level

    def on_set_level(self, level: Level) -> None:
        # Fires on initial load, level transitions, and resets (both reclone
        # the level from its pristine copy — see `handle_reset` — so this is
        # the one place to reset the abstract SimState back to its start).
        if not self._level_ctx:
            return  # first call, from ARCBaseGame.__init__ before our lists exist
        ctx = self._level_ctx[self._current_level_index]
        self._sim_state = ctx.init_state

    # ── per-action simulation ───────────────────────────────────────────

    def step(self) -> None:
        spec = self.SPEC
        ctx = self._level_ctx[self._current_level_index]
        state = self._sim_state
        assert state is not None

        action = self.action
        if action.id.value == 6:
            data = action.data or {}
            xy = (int(data.get("x", 0)) // self._cell_px, int(data.get("y", 0)) // self._cell_px)
            event = ActionEvent(6, xy=xy)
        else:
            event = ActionEvent(action.id.value)

        new_state = self._mechanic.step(state, ctx.layout, ctx.entities, spec.primary_params, event)
        if self._hazard is not None:
            new_state = self._hazard.apply(new_state, ctx.layout, ctx.hazard_cells, spec.hazard_params)

        self._sync_sprites(spec, ctx, state, new_state)
        self._sim_state = new_state

        if new_state.terminal == "WIN":
            self.next_level()
        elif new_state.terminal == "LOSE":
            self.lose()
        self.complete_action()

    @staticmethod
    def _recolor_by_state(sprite: Sprite, base_color: int, state_value: int) -> None:
        color = (base_color - 1 + state_value) % 15 + 1
        sprite.color_remap(None, color)

    def _sync_sprites(self, spec: GameSpec, ctx: LevelContext, old: SimState, new: SimState) -> None:
        px = self._cell_px

        if new.actor is not None and new.actor != old.actor and ctx.actor_sprite is not None:
            ctx.actor_sprite.set_position(new.actor[0] * px, new.actor[1] * px)

        if spec.primary_key == "state_match_goal" and ctx.actor_sprite is not None:
            actor_state = new.cell_state.get("actor_state", 0)
            self._recolor_by_state(ctx.actor_sprite, spec.role_colors.get("actor", 8), actor_state)  # type: ignore[arg-type]
            goal_sprite = ctx.static_sprites.get(ctx.entities["goal"])  # type: ignore[arg-type]
            if goal_sprite is not None:
                self._recolor_by_state(
                    goal_sprite, spec.role_colors.get("goal", 3), ctx.entities["target_state"]  # type: ignore[arg-type]
                )

        if spec.primary_key == "sokoban" and ctx.block_sprite is not None:
            new_block = next((c for c, v in new.cell_state.items() if v == 1), None)
            old_block = next((c for c, v in old.cell_state.items() if v == 1), None)
            if new_block is not None and new_block != old_block:
                ctx.block_sprite.set_position(new_block[0] * px, new_block[1] * px)  # type: ignore[index]

        for c in new.collected - old.collected:
            if c in ctx.static_sprites:
                ctx.static_sprites[c].color_remap(None, FLOOR_COLOR)
        for c in new.open_doors - old.open_doors:
            if c in ctx.static_sprites:
                ctx.static_sprites[c].color_remap(None, FLOOR_COLOR)
        if spec.primary_key == "checkpoints_in_order" and new.checkpoint_index > old.checkpoint_index:
            checkpoints: list[Cell] = ctx.entities["checkpoints"]  # type: ignore[assignment]
            for c in checkpoints[old.checkpoint_index : new.checkpoint_index]:
                if c in ctx.static_sprites:
                    ctx.static_sprites[c].color_remap(None, FLOOR_COLOR)

        if spec.primary_key == "paint_propagate":
            for c, v in new.cell_state.items():
                if old.cell_state.get(c) != v and c in ctx.static_sprites:
                    color = spec.role_colors.get("targets", 4) if v else FLOOR_COLOR
                    ctx.static_sprites[c].color_remap(None, color)  # type: ignore[arg-type]
        elif spec.primary_key == "rotate_to_match":
            for c, v in new.cell_state.items():
                if old.cell_state.get(c) != v and c in ctx.static_sprites:
                    self._recolor_by_state(ctx.static_sprites[c], spec.role_colors.get("targets", 4), v)  # type: ignore[arg-type]

        if spec.hazard_key == "decaying_tiles":
            for c, v in new.cell_state.items():
                if v == -1 and old.cell_state.get(c) != -1 and c in ctx.static_sprites:
                    ctx.static_sprites[c].color_remap(None, spec.role_colors.get("hazard", 2))  # type: ignore[arg-type]

        if spec.hazard_key == "moving_hazard" and ctx.hazard_sprite is not None and ctx.hazard_cells:
            pos = ctx.hazard_cells[new.step_count % len(ctx.hazard_cells)]
            ctx.hazard_sprite.set_position(pos[0] * px, pos[1] * px)
