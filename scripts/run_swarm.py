#!/usr/bin/env python3
"""Record an actual native swarm rollout, neural actions, and a final hold."""
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
from arena_mujoco.swarm_env import SwarmVectorEnv, FEATURES
from arena_mujoco.swarm_policy import NumpySwarmPolicy, ZeroSwarmPolicy

ROOT = Path(__file__).resolve().parents[1]
PHASES = ["approach", "grip and lift", "transport", "lower", "release", "delivered"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["neural", "teacher", "zero"], default="neural")
    p.add_argument("--weights", type=Path, default=ROOT / "output/swarm/policy/weights.npz")
    p.add_argument("--output", type=Path, default=ROOT / "output/swarm/proof")
    p.add_argument("--robots", type=int, default=40)
    p.add_argument("--objects", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--difficulty", type=float, default=1.)
    p.add_argument("--episode-seconds", type=float, default=60.)
    p.add_argument("--hold-seconds", type=float, default=5.)
    p.add_argument("--timestep", type=float, default=.002)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    env = SwarmVectorEnv(num_envs=1, num_robots=args.robots, num_objects=args.objects,
                         seed=args.seed, episode_seconds=args.episode_seconds, timestep=args.timestep)
    env.set_curriculum({"num_active_robots": args.robots, "num_active_objects": args.objects,
                        "difficulty": args.difficulty})
    obs = env.reset()
    policy = NumpySwarmPolicy(args.weights) if args.policy == "neural" else ZeroSwarmPolicy()
    (out / "scene.xml").write_text(env.xml)
    (out / "metadata.json").write_text(json.dumps(env.metadata, indent=2) + "\n")
    times, poses, phase_labels, object_phases, contacts, floor_contacts = [], [], [], [], [], []
    observation_trace = {key: [] for key in ("local", "neighbors", "neighbor_mask", "active")}
    actions, action_times = [], []
    robot_travel = np.zeros(args.robots)
    object_carried = np.zeros(args.objects)
    contact_samples = np.zeros(args.objects)
    supported_samples = np.zeros(args.objects)
    collision_steps = 0
    successful_at = None
    hold_valid = True
    failure = None
    start = time.perf_counter()
    previous_robot = env._state()[0][0, :, :2].numpy().copy()
    previous_object = env._state()[1][0, :, :2].numpy().copy()

    def frame():
        data = env.native_snapshot()
        times.append(float(data.time))
        poses.append(data.qpos.copy())
        object_phases.append(env.phase[0].numpy().copy())
        contacts.append(env.pad_contact[0].numpy().astype(np.uint8).copy())
        floor_contacts.append(env.object_floor_contact[0].numpy().astype(np.uint8).copy())
        counts = np.bincount(env.phase[0].numpy(), minlength=6)
        phase_labels.append(" | ".join(f"{count} {PHASES[i]}" for i, count in enumerate(counts) if count))

    frame()
    info = None
    for step in range(env.max_steps + round(args.hold_seconds / env.dt)):
        numpy_obs = {key: value.detach().cpu().numpy().copy() for key, value in obs.items()}
        action = env.teacher_action().numpy() if args.policy == "teacher" else policy(numpy_obs)
        for key in observation_trace:
            observation_trace[key].append(numpy_obs[key][0])
        actions.append(action[0].copy())
        action_times.append(float(env.native_data[0].time))
        try:
            obs, reward, done, truncated, info = env.step(torch.as_tensor(action, dtype=torch.float32))
        except (ValueError, FloatingPointError) as error:
            failure = str(error)
            break
        state = env._state()
        current_robot = state[0][0, :, :2].numpy()
        current_object = state[1][0, :, :2].numpy()
        robot_travel += np.linalg.norm(current_robot - previous_robot, axis=-1)
        clear = info["clearance"][0].numpy()
        both = env.pad_contact[0, :args.objects * 2].numpy().reshape(args.objects, 2).min(-1) > 0
        supported = both & (clear > .004) & (env.object_floor_contact[0].numpy() == 0)
        object_carried += np.linalg.norm(current_object - previous_object, axis=-1) * supported
        contact_samples += both
        supported_samples += supported
        previous_robot, previous_object = current_robot.copy(), current_object.copy()
        collision_steps += int(info["collisions"][0] > 0)
        frame()
        if bool(info["failure"][0]):
            failure = "Field exit, excessive body tilt, nonfinite state, or physics overflow"
            break
        if bool(info["success"][0]) and successful_at is None:
            successful_at = float(env.native_data[0].time)
        if successful_at is not None:
            hold_valid &= bool(info["success"][0])
            if float(env.native_data[0].time) >= successful_at + args.hold_seconds - 1e-8:
                break
        elif bool(truncated[0]):
            failure = "Episode time budget exhausted"
            break
        if step % 250 == 0:
            print(json.dumps({"step": step, "time": times[-1], "phase": object_phases[-1].tolist(),
                              "delivered": int(info["delivered"][0]), "clearance_m": clear.tolist()}), flush=True)
    np.savez_compressed(out / "trajectory.npz", times=np.asarray(times), qpos=np.asarray(poses),
                        phases=np.asarray(phase_labels), object_phases=np.asarray(object_phases),
                        pad_contacts=np.asarray(contacts), object_floor_contacts=np.asarray(floor_contacts))
    np.savez_compressed(out / "policy_trace.npz", times=np.asarray(action_times), actions=np.asarray(actions),
                        **{key: np.asarray(value) for key, value in observation_trace.items()})
    held = 0. if successful_at is None else times[-1] - successful_at
    success = successful_at is not None and hold_valid and held >= args.hold_seconds - 1e-7 and failure is None
    sources = ["arena_mujoco/swarm_env.py", "arena_mujoco/swarm_robot.py", "arena_mujoco/swarm_policy.py",
               "arena_mujoco/builder.py", "arena_mujoco/materials.py", "arena_spec.json", "scripts/run_swarm.py"]
    report = {"success": success, "task": "cooperative_transport_benchmark", "official_competition_score": None,
              "policy": args.policy, "robot_count": args.robots, "object_count": args.objects,
              "completed_objects": int(info["delivered"][0]) if info is not None else 0,
              "completion_time_s": successful_at, "final_hold_s": held, "final_hold_valid": bool(hold_valid),
              "simulator": "native_MuJoCo", "mujoco": mujoco_version(), "physics_timestep_s": args.timestep,
              "control_timestep_s": env.dt, "seed": args.seed, "difficulty": args.difficulty,
              "episode_seconds": args.episode_seconds, "requested_hold_seconds": args.hold_seconds,
              "wall_seconds": time.perf_counter() - start, "failure": failure,
              "robot_travel_m": robot_travel.tolist(), "object_carried_distance_m": object_carried.tolist(),
              "object_supported_control_steps": supported_samples.tolist(),
              "collision_control_steps": collision_steps, "neural_calls": len(actions) if args.policy == "neural" else 0,
              "weights_sha256": sha(args.weights) if args.policy == "neural" else None,
              "local_feature_names": FEATURES, "goals_m": env.goals[0].tolist(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_hashes": {path: sha(ROOT / path) for path in sources},
              "artifacts": {path: sha(out / path) for path in ("scene.xml", "metadata.json", "trajectory.npz", "policy_trace.npz")}}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if success else 1


def mujoco_version():
    import mujoco
    return mujoco.__version__


if __name__ == "__main__":
    raise SystemExit(main())
