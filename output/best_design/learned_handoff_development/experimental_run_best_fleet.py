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
    parser.add_argument("--time-penalty-per-second", type=float, default=.5)
    parser.add_argument("--drive-limits", type=float, nargs=2, default=(.35, 2.5), metavar=("M_S", "RAD_S"))
    parser.add_argument("--drive-policy", help="exported weights.npz or its model directory")
    parser.add_argument("--learned-precision-distance", type=float, default=.030, metavar="M",
                        help="switch learned travel to geometric docking within this distance")
    parser.add_argument("--learned-precision-heading", type=float, default=.08, metavar="RAD",
                        help="switch learned turns to geometric alignment within this angle")
    parser.add_argument("--lab-drive-limits", type=float, nargs=2, metavar=("M_S", "RAD_S"))
    parser.add_argument("--green-upper-first", action="store_true")
    args = vars(parser.parse_args())
    result = run(**args)
    print(json.dumps({key: result[key] for key in ("success", "declaration_seconds", "deployment_seconds", "failures", "wall_seconds")}, indent=2))
    print("score", result["score_at_declaration"]["task_score"])
