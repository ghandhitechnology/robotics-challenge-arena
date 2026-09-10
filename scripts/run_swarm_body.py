#!/usr/bin/env python3
"""Record native magnetic-body actions and physics for independent replay.

A teacher rollout is a diagnostic and requires --allow-teacher. A recorder report
never certifies a final neural proof; run verify_swarm_body.py on the artifacts.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_mujoco.swarm_body_env import FEATURES
from arena_mujoco.swarm_flow import connected_components
from arena_mujoco.swarm_policy import NumpySwarmPolicy
from scripts.verify_swarm_body import (
    ARTIFACTS, CONTRACT, ENV_FACTORY, FORCE_THRESHOLD_N, TASK, OBSERVATIONS,
    audit_body_task, audit_training, capture_frame, digest, make_environment, source_hashes,
)


def record_body(directory, *, policy_type="neural", weights=None, training_report=None,
                allow_teacher=False, robots=40, objects=2, seed=20260911,
                episode_seconds=70., hold_seconds=5., timestep=.002, max_steps=None, verbose=True):
    if robots != 40 or objects != 2:
        raise ValueError("The body proof records exactly forty modules and two payloads")
    if policy_type not in {"neural", "teacher"} or (policy_type == "teacher" and not allow_teacher):
        raise ValueError("Teacher diagnostics require --allow-teacher; final recordings use neural actions")
    if not (0 < episode_seconds <= 3600 and 5 <= hold_seconds <= 60) or timestep not in (.001, .002):
        raise ValueError("Use a positive episode duration, a hold of at least five seconds, and a 1 or 2 ms timestep")
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    policy = None
    checkpoint_errors = []
    artifacts = set(ARTIFACTS)
    report = {
        "schema_version": 1, "task": TASK, "env_factory": ENV_FACTORY,
        "official_competition_score": None, "policy": policy_type,
        "robot_count": robots, "object_count": objects, "seed": seed, "difficulty": 1.,
        "episode_seconds": float(episode_seconds), "requested_hold_seconds": float(hold_seconds),
        "physics_timestep_s": float(timestep), "control_timestep_s": .02,
        "simulator": "native_MuJoCo", "mujoco": mujoco.__version__,
        "local_feature_names": FEATURES, "policy_contract": CONTRACT,
        "magnetic_force_threshold_n": FORCE_THRESHOLD_N,
        "magnetic_sample_timing": "Force arrays are the actual final substep loads, sampled before its integration",
        "runtime": {"python": platform.python_version(), "system": platform.system(),
                    "machine": platform.machine(), "numpy": np.__version__, "torch": torch.__version__},
        "weights_sha256": None, "training_report_sha256": None,
        "final_neural_proof": False, "verification_required": True,
    }
    if policy_type == "neural":
        weights = Path(weights) if weights else ROOT/"output/swarm/body_policy/weights.npz"
        training_report = Path(training_report) if training_report else weights.parent/"training.json"
        if not weights.is_file() or not training_report.is_file():
            raise ValueError("Neural recording needs weights.npz and its matching training.json")
        for source, name in ((weights, "weights.npz"), (training_report, "training.json")):
            if source.resolve() != (directory/name).resolve():
                shutil.copyfile(source, directory/name)
            artifacts.add(name)
        policy = NumpySwarmPolicy(directory/"weights.npz")
        if any(getattr(policy.config, key) != value for key, value in CONTRACT.items()):
            raise ValueError("Checkpoint must use 40 local, 12 neighbor, 88 global features and four actions")
        report["weights_sha256"] = digest(directory/"weights.npz")
        report["training_report_sha256"] = digest(directory/"training.json")
        audit_training(json.loads((directory/"training.json").read_text()), report["weights_sha256"],
                       vars(policy.config), checkpoint_errors)
    report["checkpoint_acceptance_passed"] = policy_type == "neural" and not checkpoint_errors
    report["checkpoint_acceptance_errors"] = checkpoint_errors
    env, observation = make_environment(report)
    start = time.perf_counter()
    frames, trace = [], {key: [] for key in ("times", "actions", *OBSERVATIONS)}
    failure, completion = None, None
    info = None
    budget = env.max_steps+round(hold_seconds/env.dt)
    if max_steps is not None:
        budget = min(budget, max_steps)
    try:
        (directory/"scene.xml").write_text(env.xml)
        (directory/"metadata.json").write_text(json.dumps(env.metadata, indent=2, allow_nan=False)+"\n")
        frames.append(capture_frame(env))
        for step in range(budget):
            numpy_obs = {key: observation[key].detach().cpu().numpy().copy() for key in OBSERVATIONS}
            action = env.teacher_action().numpy() if policy_type == "teacher" else policy(numpy_obs)
            trace["times"].append(float(env.native_data[0].time))
            trace["actions"].append(action[0].copy())
            for key in OBSERVATIONS:
                trace[key].append(numpy_obs[key][0].copy())
            try:
                observation, _, _, truncated, info = env.step(torch.as_tensor(action, dtype=torch.float32))
            except (RuntimeError, ValueError, FloatingPointError) as error:
                failure = f"Control step failed: {type(error).__name__}: {error}"
                break
            frame = capture_frame(env)
            frames.append(frame)
            now = float(frame["times"])
            largest = connected_components(frame["magnetic_graph"])[1][0]
            candidate = (bool(info["success"][0]) and largest == 40 and frame["substep_min_largest_component"] == 40
                         and not np.any(frame["payload_robot_contacts"]))
            if candidate:
                if completion is None:
                    completion = now
                if now-completion >= hold_seconds-1e-8:
                    break
            else:
                completion = None
            if bool(info["failure"][0]):
                failure = "Native environment reports field exit, excessive tilt, or failed physics"
                break
            if bool(truncated[0]) and completion is None:
                failure = "Episode time budget exhausted before a complete body-and-payload hold"
                break
            if verbose and step % 250 == 0:
                print(json.dumps({"time_s": round(now, 2), "body_phase": int(frame["body_phase"]),
                                  "object_phases": frame["object_phases"].tolist(),
                                  "largest_force_connected_component": largest}), flush=True)
        if failure is None and (completion is None or float(frames[-1]["times"])-completion < hold_seconds-1e-8):
            failure = "Recording step budget exhausted before the final hold completed"
        trajectory = {key: np.asarray([frame[key] for frame in frames]) for key in frames[0]}
        trace_arrays = {key: np.asarray(value) for key, value in trace.items()}
        np.savez_compressed(directory/"trajectory.npz", **trajectory)
        np.savez_compressed(directory/"policy_trace.npz", **trace_arrays)
        report.update(completion_time_s=completion,
                      final_hold_s=0. if completion is None else float(frames[-1]["times"])-completion,
                      failure=failure, goals_m=env.goals[0].tolist(),
                      completed_objects=int(info["delivered"][0]) if info is not None else 0,
                      recorded_control_steps=len(trace["actions"]), recorded_frames=len(frames),
                      neural_calls=len(trace["actions"]) if policy_type == "neural" else 0,
                      wall_seconds=time.perf_counter()-start,
                      sources_stable_during_recording=hashes == source_hashes(), source_hashes=hashes,
                      source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
        task_errors = []
        if len(frames) == len(trace["actions"])+1:
            report["task_checks"] = audit_body_task(trajectory, trace_arrays, env.model, env.metadata, report, task_errors)
        else:
            task_errors.append("Incomplete physics recording after a failed control step")
        report["success"] = not task_errors
        report["task_errors"] = task_errors
        report["final_hold_valid"] = not task_errors and report["final_hold_s"] >= hold_seconds-1e-7
        report["artifacts"] = {name: digest(directory/name) for name in sorted(artifacts)}
        (directory/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        if verbose:
            print(json.dumps({"report": str(directory/"report.json"), "task_passed": report["success"],
                              "final_neural_proof": False, "failure": failure,
                              "control_steps": report["recorded_control_steps"]}), flush=True)
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("neural", "teacher"), default="neural")
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--weights", type=Path, default=ROOT/"output/swarm/body_policy/weights.npz")
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"output/swarm/body_proof")
    parser.add_argument("--robots", type=int, default=40)
    parser.add_argument("--objects", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--episode-seconds", type=float, default=70.)
    parser.add_argument("--hold-seconds", type=float, default=5.)
    parser.add_argument("--timestep", type=float, default=.002)
    parser.add_argument("--max-steps", type=int, help="Limit a diagnostic recording; the full proof gates still apply")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        report = record_body(args.output, policy_type=args.policy, weights=args.weights,
                             training_report=args.training_report, allow_teacher=args.allow_teacher,
                             robots=args.robots, objects=args.objects, seed=args.seed,
                             episode_seconds=args.episode_seconds, hold_seconds=args.hold_seconds,
                             timestep=args.timestep, max_steps=args.max_steps)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
