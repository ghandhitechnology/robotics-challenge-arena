#!/usr/bin/env python3
"""Finish a deferred swarm training audit with native CPU evaluation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_swarm_policy import (
    acceptance_passed, file_sha256, load_exported_actor, run_final_evaluations,
    save_json, training_source_provenance,
)


BODY_CONTRACT = {"local_dim": 42, "neighbor_dim": 12, "global_dim": 92, "action_dim": 4}
BODY_FACTORY = "arena_mujoco.swarm_body_env:SwarmBodyEnv"
ARTIFACTS = ("weights.npz", "checkpoint.pt", "warmstart.npz")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_immutable_inputs(directory, pending):
    expected_sources = {"commit": pending.get("source_commit"),
                        "files": pending.get("source_hashes")}
    require(training_source_provenance() == expected_sources,
            "Current source commit or hashes differ from deferred training")
    expected_artifacts = pending.get("artifact_hashes")
    require(isinstance(expected_artifacts, dict) and set(expected_artifacts) == set(ARTIFACTS),
            "Deferred training artifact manifest is incomplete")
    actual_artifacts = {name: file_sha256(directory / name) for name in ARTIFACTS}
    require(actual_artifacts == expected_artifacts,
            "Deferred training artifacts changed before evaluation completed")
    require(pending.get("weights_sha256") == actual_artifacts["weights.npz"],
            "Deferred report does not bind the final actor weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_dir", type=Path)
    args = parser.parse_args()
    directory = args.policy_dir.resolve()
    pending_path = directory / "training.pending.json"
    if (directory / "training.json").exists():
        parser.error("training.json already exists; keep the completed audit immutable")
    if not pending_path.is_file():
        parser.error(f"Missing deferred report: {pending_path}")
    pending = json.loads(pending_path.read_text())
    require(pending.get("evaluation_status") == "pending"
            and pending.get("acceptance_passed") is False,
            "Deferred report is not in the pending evaluation state")
    stored_args = pending.get("args", {})
    require(stored_args.get("defer_evaluation") is True
            and stored_args.get("env_factory") == BODY_FACTORY
            and stored_args.get("backend") == "native"
            and stored_args.get("robots") == 40 and stored_args.get("objects") == 2
            and stored_args.get("cpu_smoke") is False,
            "Deferred report is not a full native magnetic-body training run")
    require("A100" in str(pending.get("gpu", ""))
            and pending.get("cuda_optimization") is True
            and str(pending.get("policy_device", "")).startswith("cuda")
            and isinstance(pending.get("cuda"), str) and pending["cuda"],
            "Deferred report lacks A100 CUDA training provenance")
    verify_immutable_inputs(directory, pending)

    evaluation_args = SimpleNamespace(**stored_args)
    evaluation_args.output = str(directory)
    evaluation_args.eval_episodes = 32
    device = torch.device("cpu")
    torch.set_num_threads(2)
    model, config = load_exported_actor(directory / "weights.npz", device)
    require(asdict(config) == pending.get("config")
            and all(getattr(config, key) == value for key, value in BODY_CONTRACT.items()),
            "Deferred actor configuration differs from the 42-feature body contract")

    started = time.monotonic()
    final_eval, zero_eval, warmstart_eval = run_final_evaluations(
        model, config, evaluation_args, device)
    require(warmstart_eval is not None, "Deferred audit requires the saved warmstart baseline")
    require(all(result.get("complete") is True
                for result in (final_eval, zero_eval, warmstart_eval)),
            "All three 32-episode evaluations must complete before sealing training.json")
    verify_immutable_inputs(directory, pending)

    passed = acceptance_passed(
        evaluation_args, pending.get("final_stage"),
        pending.get("cumulative_ppo_actor_update_l2", 0.), final_eval, zero_eval)
    report = dict(pending)
    report.update({
        "heldout": final_eval,
        "zero_baseline": zero_eval,
        "warmstart_baseline": warmstart_eval,
        "acceptance_passed": passed,
        "evaluation_status": "complete",
        "evaluation_seconds": time.monotonic() - started,
        "evaluation_provenance": {
            "device": "cpu",
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "episodes_per_policy": 32,
            "seed": evaluation_args.seed + 200000,
            "native_workers": evaluation_args.native_workers,
        },
    })
    save_json(directory / "training.json", report)
    save_json(directory / "progress.json", {
        "phase": "complete", "acceptance_passed": passed,
        "heldout": final_eval, "evaluation_seconds": report["evaluation_seconds"],
    })
    print("TRAINING_REPORT " + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
