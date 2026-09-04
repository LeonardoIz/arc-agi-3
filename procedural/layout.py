"""Procedural room/maze layout generation: which cells are walls vs open.

Deliberately simple (a randomized-walk carver, not a full maze-generation
algorithm) — connectivity is guaranteed by construction (every carved cell
is adjacent to an already-open one), which is what mechanics need to place
entities that are actually reachable from each other.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .state import Cell


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    walls: frozenset[Cell]
    open_cells: tuple[Cell, ...]  # deterministic order, for reproducible sampling

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_open(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.walls

    def neighbors4(self, cell: Cell) -> list[Cell]:
        x, y = cell
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in candidates if self.in_bounds(c)]


def generate_layout(
    width: int, height: int, open_fraction: float, rng: random.Random
) -> Layout:
    """Carve `open_fraction` of the grid open via a randomized walk that
    always extends from the current open set, so the result is a single
    connected region by construction.
    """
    target_open = max(4, int(width * height * open_fraction))
    start = (rng.randrange(width), rng.randrange(height))
    open_cells: set[Cell] = {start}
    frontier: set[Cell] = set()

    def add_neighbors_to_frontier(cell: Cell) -> None:
        x, y = cell
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in open_cells:
                frontier.add((nx, ny))

    add_neighbors_to_frontier(start)
    while len(open_cells) < target_open and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.discard(cell)
        open_cells.add(cell)
        add_neighbors_to_frontier(cell)

    all_cells = {(x, y) for x in range(width) for y in range(height)}
    walls = frozenset(all_cells - open_cells)
    ordered_open = tuple(sorted(open_cells))
    return Layout(width=width, height=height, walls=walls, open_cells=ordered_open)


def reachable_from(layout: Layout, start: Cell) -> set[Cell]:
    """BFS over open cells only (4-connected) — used both to sanity-check a
    layout and as a building block for entity placement (e.g. "place the
    goal far from the actor, but still reachable").
    """
    seen = {start}
    frontier = [start]
    while frontier:
        cell = frontier.pop()
        for nb in layout.neighbors4(cell):
            if layout.is_open(nb) and nb not in seen:
                seen.add(nb)
                frontier.append(nb)
    return seen
