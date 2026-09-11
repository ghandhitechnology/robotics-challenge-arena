#!/usr/bin/env python3
"""Select fleet speed limits with matched native missions, then test frozen seeds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_mujoco.best_mission import mission_sources, run

SPEEDS = ((.35, 2.5), (.40, 2.75), (.42, 3.), (.42, 3.5), (.45, 3.5), (.48, 4.))
PROFILES = [{"name": f"uniform_{speed[0]:g}_{speed[1]:g}", "limits": list(speed),
             "lab_limits": None, "green_upper_first": False} for speed in SPEEDS]
PROFILES += [{"name": f"fast_couriers_{speed[0]:g}_{speed[1]:g}", "limits": list(speed),
              "lab_limits": [.35, 2.5], "green_upper_first": True}
             for speed in ((.40, 2.75), (.45, 3.5), (.48, 4.))]


def sources():
    return {**mission_sources(), "scripts/optimize_best_fleet.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def episode(task):
    profile, seed = task
    limits = profile["limits"]
    try:
        report = run(seed=seed, randomize=True, drive_limits=limits,
                     lab_drive_limits=profile["lab_limits"], green_upper_first=profile["green_upper_first"])
        initial = report["score_at_declaration"]
        final = report["score_after_five_seconds"]
        return {
            "profile": profile["name"], "limits": list(limits), "seed": seed, "success": report["success"],
            "score": min(initial["score"], final["score"]),
            "task_score": min(initial["task_score"], final["task_score"]),
            "declaration_seconds": report["declaration_seconds"],
            "deployment_seconds": report["deployment_seconds"],
            "failures": report["failures"], "completed_robots": report["completed_robots"],
            "unfinished_robots": report.get("unfinished_robots", []),
            "hold": report.get("five_second_hold", {}),
            "max_torque_nm": report["maximum_torque_nm"],
            "max_tilt_rad": report["maximum_tilt_rad"],
            "physics_warning_count": final["physics_warning_count"],
            "wall_seconds": report["wall_seconds"],
            "exception": None,
        }
    except Exception as error:
        # A failed simulator episode stays in the trial count and earns zero.
        return {"profile": profile["name"], "limits": list(limits), "seed": seed, "success": False,
                "score": 0, "task_score": 0, "declaration_seconds": 120.,
                "failures": {}, "exception": f"{type(error).__name__}: {error}"}


def summary(rows):
    n = len(rows)
    p = sum(row["success"] for row in rows) / n
    z = 1.96
    lower = (p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
    times = [row["declaration_seconds"] for row in rows if row["success"]]
    return {
        "episodes": n, "full_success_rate": p, "wilson_lower_95": float(lower),
        "mean_score": float(np.mean([row["score"] for row in rows])),
        "minimum_score": min(row["score"] for row in rows),
        "mean_task_objective": float(np.mean([10 * row["score"] - .5 * row["declaration_seconds"] for row in rows])),
        "mean_success_seconds": float(np.mean(times)) if times else None,
        "p95_success_seconds": float(np.quantile(times, .95)) if times else None,
        "maximum_tilt_degrees": float(np.degrees(max(
            (max(row.get("max_tilt_rad", {}).values(), default=0.)
             if isinstance(row.get("max_tilt_rad"), dict) else row.get("max_tilt_rad", 0.))
            for row in rows))),
        "exceptions": sum(row.get("exception") is not None for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tune", "test"))
    parser.add_argument("--output", type=Path, default=ROOT / "output/best_design/search")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--episodes", type=int, help="defaults: 6 per profile for tuning, 100 for test")
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or (args.episodes is not None and args.episodes < 1):
        parser.error("workers and episodes must be positive")
    source_hashes = sources()
    count = args.episodes or (6 if args.mode == "tune" else 100)
    if args.mode == "tune":
        profiles = PROFILES
        seeds = np.random.default_rng(202609115).integers(0, 1_000_000_000, count).tolist()
        selection = None
    else:
        selection_path = args.selection or args.output / "selection.json"
        selection = json.loads(selection_path.read_text())
        if selection["source_sha256"] != source_hashes:
            raise ValueError("Mission sources differ from speed selection; create a new tuning run")
        profiles = (selection["selected_profile"],)
        seeds = np.random.default_rng(202609116).integers(1_000_000_000, 2_000_000_000, count).tolist()
    started = time.perf_counter()
    tasks = [(profile, seed) for profile in profiles for seed in seeds]
    rows = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [pool.submit(episode, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            save(args.output / f"{args.mode}_progress.json", {
                "mode": args.mode, "completed": len(rows), "total": len(tasks),
                "elapsed_seconds": time.perf_counter() - started, "last_episode": row})
            print(json.dumps({"completed": len(rows), "total": len(tasks), **row}), flush=True)
    if sources() != source_hashes:
        raise RuntimeError("Mission sources changed during evaluation")
    rows.sort(key=lambda row: (row["profile"], row["seed"]))
    candidates = [{**profile, **summary([row for row in rows if row["profile"] == profile["name"]])}
                  for profile in profiles]
    result = {
        "method": "matched native mission parameter search" if args.mode == "tune" else "frozen held-out native mission test",
        "source_sha256": source_hashes,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seeds": seeds, "randomized_physics": True, "candidates": candidates,
        "elapsed_seconds": time.perf_counter() - started, "episodes": rows,
        "controller": "geometric task planner and motor feedback",
    }
    save(args.output / f"{args.mode}.json", result)
    if args.mode == "tune":
        selected = max(candidates, key=lambda item: (
            item["full_success_rate"], item["mean_score"], item["mean_task_objective"]))
        save(args.output / "selection.json", {
            "selected_limits": selected["limits"], "source_sha256": source_hashes,
            "selected_profile": next(profile for profile in profiles if profile["name"] == selected["name"]),
            "selection_rule": "maximize full success rate, then mean official score, then score minus elapsed-time cost",
            "validation": selected, "tuning_seeds": seeds,
            "test_evaluation": "not run"})
    print(json.dumps({"mode": args.mode, "candidates": candidates}), flush=True)


if __name__ == "__main__":
    main()
