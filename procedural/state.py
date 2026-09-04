"""The abstract simulation state every mechanic reads and writes.

One flexible struct instead of one bespoke state class per mechanic: a
walker uses `actor`/`step_count`, Sokoban uses `actor`/`cell_state` (block
positions), a click-toggle puzzle uses `cell_state`, key-and-lock uses
`carrying`/`open_doors`, collect-N uses `collected`, and so on. Mechanics
only touch the fields they need.

This is deliberately *not* tied to `arcengine.Sprite`/pixels — it's the
model each mechanic's logic is defined against. `procedural/engine.py` is
the only place that renders a `SimState` into real sprites; the BFS solver
in `procedural/solver.py` operates on `SimState` directly, headless. Both
call the exact same mechanic `step` function, so the solver can never drift
from what the real game actually does.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

Cell = tuple[int, int]
Terminal = Literal["WIN", "LOSE"] | None


@dataclass(frozen=True)
class SimState:
    actor: Cell | None = None
    collected: frozenset[Cell] = field(default_factory=frozenset)
    open_doors: frozenset[Cell] = field(default_factory=frozenset)
    # Generic per-cell integer state: toggle on/off, color-cycle index,
    # "still solid" flag for decaying tiles, block positions marked by a
    # sentinel value, etc. — meaning is mechanic-specific.
    cell_state: dict[Cell, int] = field(default_factory=dict)
    step_count: int = 0
    carrying: str | None = None
    checkpoint_index: int = 0
    terminal: Terminal = None

    def replace(self, **kwargs: object) -> "SimState":
        return dataclasses.replace(self, **kwargs)  # type: ignore[arg-type]
