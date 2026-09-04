"""Offline evaluator: run `agent/my_agent.py` against real games and report
the official scorecard formula, broken down per level / per game / total —
not just the final aggregate number `scripts/play_local.py` prints.

Score formula (see `docs/design.md` for the full citation trail into
`arc_agi.scorecard`):
  - per level: a quadratic efficiency score capped at 115
  - per game (`EnvironmentScoreList.score`): **max**, not average, over
    repeated runs/attempts of the same game_id
  - total (`score`): simple mean over games

Usage:
    .venv/bin/python scripts/evaluate.py --game ls20 --max-steps 200
    .venv/bin/python scripts/evaluate.py --game ls20,vc33 --runs 3
    .venv/bin/python scripts/evaluate.py --offline   # after games are cached
    .venv/bin/python scripts/evaluate.py --list
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode


def load_my_agent_class():
    """Import MyAgent from agent/my_agent.py via importlib."""
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise SystemExit("agent/my_agent.py must define a class named `MyAgent`")
    return module.MyAgent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default=None,
                   help="Game id to evaluate. If omitted, evaluates ALL available games. "
                        "Comma-separated list also accepted, e.g. ls20,vc33.")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Per-attempt cap on actions (overrides MyAgent.MAX_ACTIONS).")
    p.add_argument("--runs", type=int, default=1,
                   help="Attempts per game. The official per-game score is the MAX "
                        "across attempts, so >1 measures best-of-N, not average.")
    p.add_argument("--offline", action="store_true",
                   help="Use OperationMode.OFFLINE: no network calls, only games "
                        "already cached in environment_files/ (populated by a prior "
                        "NORMAL run). Fails on uncached games instead of fetching them.")
    p.add_argument("--list", action="store_true",
                   help="List available games and exit.")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    mode = OperationMode.OFFLINE if args.offline else OperationMode.NORMAL
    arc = arc_agi.Arcade(operation_mode=mode)
    all_envs = arc.get_environments()

    if args.list:
        print(f"{len(all_envs)} environments:")
        for e in all_envs:
            print(f"  {e.game_id}: {getattr(e, 'title', '?')}")
        return

    if args.game:
        wanted = {g.strip().split("-")[0] for g in args.game.split(",")}
        game_ids = [e.game_id.split("-")[0] for e in all_envs
                    if e.game_id.split("-")[0] in wanted]
        missing = wanted - set(game_ids)
        if missing:
            raise SystemExit(f"Unknown game id(s): {sorted(missing)}. Run --list.")
    else:
        game_ids = [e.game_id.split("-")[0] for e in all_envs]
        print(f"No --game specified; evaluating all {len(game_ids)} games.\n")

    MyAgentCls = load_my_agent_class()
    if hasattr(MyAgentCls, "MAX_ACTIONS"):
        MyAgentCls.MAX_ACTIONS = min(MyAgentCls.MAX_ACTIONS, args.max_steps)

    for game_id in game_ids:
        for attempt in range(1, args.runs + 1):
            label = f"{game_id}" if args.runs == 1 else f"{game_id} (attempt {attempt}/{args.runs})"
            print(f"=== {label} ===")
            env = arc.make(game_id)
            if env is None:
                print(f"  could not create env for {game_id!r}, skipping")
                continue
            agent = MyAgentCls(
                card_id="eval",
                game_id=game_id,
                agent_name=f"MyAgent.eval.{game_id}",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=env,
                tags=["eval"],
            )
            agent.main()
            final = agent.frames[-1]
            print(f"  state={final.state}, levels_completed={final.levels_completed}, "
                  f"actions={agent.action_counter}")

    sc = arc.get_scorecard()
    if sc is None:
        raise SystemExit("No scorecard produced.")

    print("\n========= SCORE BREAKDOWN =========")
    for env_score_list in sc.environments:
        print(f"\n{env_score_list.id}  →  score={env_score_list.score:.2f}  "
              f"(max over {len(env_score_list.runs)} run(s))  "
              f"levels_completed={env_score_list.levels_completed}")
        for i, run in enumerate(env_score_list.runs, 1):
            levels = run.level_scores or []
            levels_str = ", ".join(f"{s:.1f}" for s in levels) if levels else "(no levels reached)"
            print(f"  run {i}: score={run.score:.2f}  state={run.state}  "
                  f"actions={run.actions}  level_scores=[{levels_str}]")

    print("\n========= TOTAL =========")
    print(f"games evaluated: {len(sc.environments)}")
    print(f"score_total (mean over games): {sc.score:.2f}")


if __name__ == "__main__":
    main()
