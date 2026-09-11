#!/usr/bin/env python3
"""Run and record the native five-robot preliminary prototype."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.best_mission import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/best_design/mission")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--maximum-seconds", type=float, default=120.)
    args = vars(parser.parse_args())
    result = run(**args)
    print(json.dumps({key: result[key] for key in ("success", "declaration_seconds", "deployment_seconds", "failures", "wall_seconds")}, indent=2))
    print("score", result["score_at_declaration"]["task_score"])
