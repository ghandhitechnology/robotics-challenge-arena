#!/usr/bin/env python3
"""Measure the actual robot-contact model before choosing a training batch."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from arena_mujoco.swarm_env import SwarmVectorEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robots", type=int, default=40)
    p.add_argument("--objects", type=int, default=4)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--backend", choices=["native", "warp"], default="warp")
    p.add_argument("--output", default="output/swarm/benchmark.json")
    a = p.parse_args()
    device = "cuda:0" if a.backend == "warp" else "cpu"
    start = time.perf_counter()
    print("BENCHMARK_START", json.dumps(vars(a)), flush=True)
    env = SwarmVectorEnv(num_envs=a.envs, num_robots=a.robots, num_objects=a.objects,
                         device=device, backend=a.backend)
    prepared = time.perf_counter() - start
    print("BENCHMARK_MODEL_READY", json.dumps({"nq": env.model.nq, "nv": env.model.nv,
            "geoms": env.model.ngeom, "preparation_seconds": prepared}), flush=True)
    warm = time.perf_counter()
    env.step(env.teacher_action())
    if a.backend == "warp": torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - warm
    print("BENCHMARK_COMPILED", compile_seconds, flush=True)
    start = time.perf_counter()
    total_done = 0
    max_collision = 0
    max_clearance = 0.
    for step in range(a.steps):
        obs, reward, done, truncated, info = env.step(env.teacher_action())
        if (step + 1) % 25 == 0:
            print("BENCHMARK_PROGRESS", json.dumps({"step": step + 1,
                    "phase": info["phase"][0].tolist(), "overflow": int(info["overflow"].max())}), flush=True)
        total_done += int(done.sum())
        max_collision = max(max_collision, int(info["collisions"].max()))
        max_clearance = max(max_clearance, float(info["clearance"].max()))
        if bool((done | truncated).any()): env.reset_done(done | truncated)
    if a.backend == "warp": torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    report = {**vars(a), "device": device,
              "gpu": torch.cuda.get_device_name(0) if a.backend == "warp" else None,
              "torch": torch.__version__, "preparation_seconds": prepared,
              "compile_seconds": compile_seconds, "measured_seconds": elapsed,
              "world_control_steps_per_second": a.envs * a.steps / elapsed,
              "world_physics_steps_per_second": a.envs * a.steps * env.substeps / elapsed,
              "robot_control_steps_per_second": a.envs * a.steps * a.robots / elapsed,
              "terminations": total_done, "maximum_simultaneous_robot_collisions": max_collision,
              "maximum_object_clearance_m": max_clearance,
              "xml_sha256": hashlib.sha256(env.xml.encode()).hexdigest(),
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if a.backend == "warp" else None,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("BENCHMARK_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
