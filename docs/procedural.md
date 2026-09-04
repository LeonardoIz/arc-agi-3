# The `procedural/` package

A procedural generator for ARC-AGI-3-style minigames, independent of
`model/` and `agent/`. Its only job is producing `arcengine`-playable game
files — a generated game is indistinguishable, from the agent/model's
point of view, from a real public game.

> **Why this exists:** `docs/design.md` §5.1 — Algorithm Distillation needs
> a broad distribution of tasks to learn a general exploration strategy
> from; with only the 25 public games, the model memorizes how to approach
> those 25 instead. §8 flags this generator's diversity as the highest-risk
> variable in the whole project, more than the network architecture.

## Design decision: hand-authored mechanics, not an LLM

Considered generating each game's source with an LLM (more potential
variety per game, but hard to validate in volume — broken code,
unsolvable puzzles — and expensive at scale). Went with a deterministic
engine instead: a fixed, tested library of composable mechanics, combined
and parameterized randomly. LLM-based generation stays an option to revisit
if this library's diversity turns out insufficient once training starts.

## How a game is generated

```
sample_spec()                    — pick 1 primary mechanic + 0-1 hazard
                                    modifier + grid size/palette/levels
        │  procedural/spec.py
        ▼
GameSpec                          — pure data, no code
        │
        ├─ procedural/solver.py   — BFS over SimState (headless, same pure
        │                           mechanic functions the real game uses)
        │                           validates EVERY level is solvable
        │
        └─ procedural/engine.py   — ProceduralGame(arcengine.ARCBaseGame)
                                     renders the spec into real sprites
        ▼
a `.py` file arc_agi's local loader `exec()`s exactly like a public game
```

Only solvable specs (every level, not just the first) get written out.
`procedural/generate.py` is the CLI that does the sample → validate → write
loop and produces the diversity report.

## The mechanic library

Each mechanic (`procedural/mechanics.py`) is a bundle of three **pure**
functions — `place` (choose entity positions), `init_state` (starting
`SimState`), `step` (one action → a new `SimState`) — operating on a single
flexible state struct (`procedural/state.py`), not on `arcengine` sprites
directly. `procedural/solver.py` and `procedural/engine.py` both call the
exact same `step` function — one implementation, two consumers, so the
solver can never validate a game the engine actually renders differently.

**Primary mechanics** (one per game, defines the core loop):

| Directional (ACTION1-4) | Click-based (ACTION6) |
|---|---|
| `walker_goal` — reach the goal cell | `paint_propagate` — click toggles cells via a kernel; match a target pattern |
| `sokoban` — push a block onto a target | `rotate_to_match` — click cycles a cell's state; match target states |
| `ice_slide` — moves until hitting a wall | `key_lock` — click a key, then its matching door, to unlock all doors |
| `teleport_maze` — walker + teleporter pair |
| `one_way_maze` — walker + one-way gates |
| `collect_then_exit` — collect all items before the goal unlocks |
| `state_match_goal` — cycle your own state (ACTION5) to match the goal's |
| `survive_then_exit` — goal unlocks after N steps |
| `checkpoints_in_order` — visit cells in a specific order |

**Hazard modifiers** (optional, stack onto any directional primary — pure
lose-condition add-ons, never change the core loop): `static_hazard`,
`moving_hazard` (deterministic patrol path), `decaying_tiles` (collapse
after one visit).

Every mechanic renders every piece of state that affects the win condition
as a real color/position change (see `procedural/engine.py`'s module
docstring) — a model that only ever sees pixels still gets a fair,
learnable game (C1).

## Solvability validation

`procedural/solver.py` does breadth-first search over `SimState`, using
the mechanic's real `step` function, to find a winning action sequence for
every level of a spec before it's ever written out. This is **not** search
at inference — C5/C9 forbid the *deployed agent* from planning; this runs
once, offline, at generation time, closer to a level designer playtesting a
level than anything the agent contract restricts (arguably more permissive
than the "privileged simulator teacher" C6 already allows, since it never
touches the agent's decision-making at all).

Solve rates vary genuinely by mechanic — most solve 90-100% of sampled
instances, `sokoban` solves roughly half (random block/target placement is
often unsolvable by construction, a real property of Sokoban puzzles, not
a bug). Unsolvable specs are just rejected and resampled.

## Usage

```bash
.venv/bin/python -m procedural.generate --count 200
.venv/bin/python -m procedural.generate --count 20 --play-check 5
```

`--play-check N` loads N generated games through `arc_agi`'s real local
loader (not the generator's own test harness) and plays random legal
actions through each — the same path `scripts/play_local.py`/
`scripts/evaluate.py` use for public games.

Output lands in `procedural_games/<game_id>/<class_name>.py` (gitignored,
same treatment as `environment_files/`) plus a `manifest.json` with every
generated spec and its solution lengths, used by the diversity report.

### Playing a generated game with the real agent

Nothing in `agent/my_agent.py` or `model/` needs to know a game is
procedural — construct an `arc_agi.models.EnvironmentInfo` from a
`manifest.json` entry and an `arc_agi.local_wrapper.LocalEnvironmentWrapper`
from that, and pass it as `arc_env` to `MyAgent` exactly like
`scripts/play_local.py` does for public games. (A CLI wrapper around this,
mirroring `play_local.py`/`evaluate.py`, is a natural next addition once
Fase 3 needs to actually train against these in bulk — not built yet since
nothing has consumed it that way so far.)

## What's next (Fase 3 scope, not this package's current gap)

- Expanding the mechanic library further if the measured diversity proves
  insufficient once training starts (§8's risk).
- Turning generated games into actual training trajectories — running a
  teacher policy (the BFS solutions themselves are one candidate) through
  many generated games and recording them for `model/train.py`.
- The LLM-based generation fallback, if this library's ceiling is reached.
