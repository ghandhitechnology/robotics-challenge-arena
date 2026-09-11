#!/usr/bin/env python3
"""Score the exported drive policy on the frozen and native held-out tests."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena_mujoco.best_drive import (
    BestDrivePolicy,
    DriveConfig,
    atomic_json,
    evaluate_drive_policy,
    load_drive_split,
    sha256_file,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=ROOT / "output/best_drive/dataset")
    result.add_argument("--model", type=Path, default=ROOT / "output/best_drive/model")
    result.add_argument("--output", type=Path)
    result.add_argument("--episodes", type=int,
                        help="bounded prefix of frozen held-out episode seeds")
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--latency-iterations", type=int, default=10000)
    result.add_argument("--min-reach-rate", type=float)
    result.add_argument("--max-final-distance-m", type=float)
    return result


def action_metrics(prediction: np.ndarray, target: np.ndarray, config: DriveConfig) -> dict:
    error = np.abs(prediction - target)
    physical = error * np.array((config.max_forward_m_s, config.max_yaw_rad_s))
    return {
        "normalized_mae": float(error.mean()),
        "normalized_p95": float(np.quantile(error, 0.95)),
        "forward_mae_m_s": float(physical[:, 0].mean()),
        "forward_p95_m_s": float(np.quantile(physical[:, 0], 0.95)),
        "yaw_mae_rad_s": float(physical[:, 1].mean()),
        "yaw_p95_rad_s": float(np.quantile(physical[:, 1], 0.95)),
    }


def closed_loop_metrics(statuses: list[dict]) -> dict:
    success = np.array([status["success"] for status in statuses], dtype=bool)
    distances = np.array([status["distance_m"] for status in statuses])
    headings = np.array([status["heading_error_rad"] for status in statuses])
    success_seconds = [status["seconds"] for status in statuses if status["success"]]
    probability = float(success.mean())
    count = len(statuses)
    z = 1.96
    lower = (probability + z*z/(2*count) - z*np.sqrt(
        probability*(1-probability)/count + z*z/(4*count*count))) / (1 + z*z/count)
    return {
        "episodes": count, "complete": True, "reach_rate": probability,
        "reach_wilson_lower_95": float(lower),
        "mean_final_distance_m": float(distances.mean()),
        "p95_final_distance_m": float(np.quantile(distances, 0.95)),
        "mean_final_heading_error_rad": float(headings.mean()),
        "p95_final_heading_error_rad": float(np.quantile(headings, 0.95)),
        "mean_episode_seconds": float(np.mean([status["seconds"] for status in statuses])),
        "mean_success_seconds": float(np.mean(success_seconds)) if success_seconds else None,
        "maximum_actual_torque_nm": float(max(status["maximum_actual_torque_nm"] for status in statuses)),
        "maximum_tilt_rad": float(max(status["peak_tilt_rad"] for status in statuses)),
        "terminal_failures": int(sum(status.get("failure_reason") is not None for status in statuses)),
        "primitive_reach_rate": {
            primitive: float(np.mean([status["success"] for status in statuses
                                      if status.get("primitive") == primitive]))
            for primitive in sorted({status.get("primitive") for status in statuses})},
    }


def main() -> None:
    args = parser().parse_args()
    if args.workers <= 0 or args.latency_iterations <= 0:
        raise ValueError("workers and latency iterations must be positive")
    weights = args.model / "weights.npz" if args.model.is_dir() else args.model
    model_dir = weights.parent
    training_path = model_dir / "training.json"
    if not training_path.is_file():
        raise FileNotFoundError(f"No training report beside {weights}")
    training = json.loads(training_path.read_text())
    if sha256_file(weights) != training["artifact_sha256"]:
        raise ValueError("Drive weights do not match the training report")
    dataset_manifest_path = args.dataset / "manifest.json"
    if sha256_file(dataset_manifest_path) != training["dataset_manifest_sha256"]:
        raise ValueError("Drive dataset manifest does not match training")
    manifest = json.loads(dataset_manifest_path.read_text())
    test_record = manifest["splits"]["test"]
    if sha256_file(args.dataset / test_record["file"]) != test_record["sha256"]:
        raise ValueError("Frozen drive test split hash mismatch")
    policy = BestDrivePolicy(weights)
    observation, target = load_drive_split(args.dataset, "test")
    prediction = policy.normalized(observation)
    inference_started = time.perf_counter_ns()
    sample = observation[0]
    for _ in range(args.latency_iterations):
        policy.normalized(sample)
    inference_ns = time.perf_counter_ns() - inference_started
    seeds = test_record["episode_seeds"]
    if args.episodes is not None:
        if not 1 <= args.episodes <= len(seeds):
            raise ValueError("--episodes must be within the frozen test episode count")
        seeds = seeds[:args.episodes]
    physics_started = time.perf_counter()
    learned_statuses = evaluate_drive_policy(
        weights, seeds, policy.config, args.workers,
        domain_seed=int(test_record["seed"]) + 30_000, policy_kind="learned")
    teacher_statuses = evaluate_drive_policy(
        weights, seeds, policy.config, args.workers,
        domain_seed=int(test_record["seed"]) + 30_000, policy_kind="teacher")
    zero_statuses = evaluate_drive_policy(
        weights, seeds, policy.config, args.workers,
        domain_seed=int(test_record["seed"]) + 30_000, policy_kind="zero")
    physics_seconds = time.perf_counter() - physics_started
    learned = closed_loop_metrics(learned_statuses)
    learned["physics_seconds_including_baselines"] = physics_seconds
    result = {
        "schema_version": 1,
        "evaluation_role": "independent final test after validation selection",
        "model_sha256": training["artifact_sha256"],
        "dataset_manifest_sha256": training["dataset_manifest_sha256"],
        "test_split_sha256": test_record["sha256"],
        "test_seed": test_record["seed"],
        "frozen_action_examples": len(observation),
        "frozen_action_metrics": action_metrics(prediction, target, policy.config),
        "native_closed_loop": {
            **learned,
            "episode_seeds_sha256": __import__("hashlib").sha256(
                np.asarray(seeds, dtype=np.int64).tobytes()).hexdigest(),
        },
        "native_baselines_same_seeds": {
            "teacher": closed_loop_metrics(teacher_statuses),
            "zero": closed_loop_metrics(zero_statuses),
        },
        "inference": {
            "iterations": args.latency_iterations,
            "mean_microseconds": inference_ns / args.latency_iterations / 1000,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
        },
        "claim_scope": "This test covers one unloaded lab robot in the empty center corridor. It does not score grasping or the full mission.",
    }
    failures = []
    reach_rate = learned["reach_rate"]
    p95_distance = learned["p95_final_distance_m"]
    if args.min_reach_rate is not None and reach_rate < args.min_reach_rate:
        failures.append(f"reach rate {reach_rate:.6f} < {args.min_reach_rate:.6f}")
    if args.max_final_distance_m is not None and p95_distance > args.max_final_distance_m:
        failures.append(f"p95 distance {p95_distance:.6f}m > {args.max_final_distance_m:.6f}m")
    result["requested_gates"] = {
        "minimum_reach_rate": args.min_reach_rate,
        "maximum_p95_final_distance_m": args.max_final_distance_m,
        "passed": not failures, "failures": failures,
    }
    output = args.output or model_dir / "test.json"
    atomic_json(output, result)
    print("DRIVE_TEST_REPORT " + json.dumps(result, allow_nan=False), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
