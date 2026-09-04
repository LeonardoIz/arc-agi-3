"""Procedural game generation for training-distribution diversity (Fase 1,
`docs/design.md` §5.1, §7). Independent of `model/` and `agent/` — this
package only produces `arcengine`-playable game files; nothing here is
imported by the agent or the model.

Layout:
  - `state.py`      — `SimState`, the abstract state every mechanic reads/writes.
  - `layout.py`      — room/maze layout generation (walls vs open cells).
  - `mechanics.py`    — the mechanic library: pure `place`/`init_state`/`step`
                        functions, directional and click-based.
  - `spec.py`          — `GameSpec` (what defines one generated game) + sampling.
  - `solver.py`         — BFS solvability check over `SimState`, headless.
  - `engine.py`          — `ProceduralGame(arcengine.ARCBaseGame)`, the sprite-
                           rendering adapter every generated game file instantiates.
  - `generate.py`         — CLI: sample, filter by solvability, write game
                            files, report diversity (`python -m procedural.generate`).

See `mechanics.py` and `engine.py`'s module docstrings for the key design
decision: game logic is pure `SimState` transitions, shared unchanged
between the solver (headless) and the real game (rendered) — the solver
can't drift from what the deployed game actually does.
"""
