#!/usr/bin/env python3
"""Collect native wheel-contact rollouts and train the best-fleet drive policy."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena_mujoco.best_drive import (
    BestDriveMLP,
    DriveConfig,
    atomic_json,
    collect_dagger_round,
    collect_drive_dataset,
    collect_ppo_rollouts,
    evaluate_drive_policy,
    export_drive_policy,
    load_drive_split,
    sha256_file,
)


SOURCE_FILES = (
    "arena_spec.json", "best_design.json", "profiles/default.json",
    "arena_mujoco/builder.py", "arena_mujoco/materials.py",
    "arena_mujoco/competition_robot.py", "arena_mujoco/best_fleet.py",
    "arena_mujoco/best_drive.py", "scripts/train_best_drive.py",
    "scripts/test_best_drive.py", "requirements-best-design.txt",
)


def source_provenance() -> dict:
    files = {name: sha256_file(ROOT / name) for name in SOURCE_FILES if (ROOT / name).is_file()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"commit": commit, "files": files}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=ROOT / "output/best_drive/dataset")
    result.add_argument("--output", type=Path, default=ROOT / "output/best_drive/model")
    result.add_argument("--train-episodes", type=int, default=720)
    result.add_argument("--validation-episodes", type=int, default=120)
    result.add_argument("--test-episodes", type=int, default=120)
    result.add_argument("--episode-seconds", type=float, default=12.0)
    result.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    result.add_argument("--seed", type=int, default=20260911)
    result.add_argument("--force-generate", action="store_true")
    result.add_argument("--generate-only", action="store_true")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--cpu-smoke", action="store_true")
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--patience", type=int, default=18)
    result.add_argument("--batch-size", type=int, default=4096)
    result.add_argument("--learning-rate", type=float, default=2e-3)
    result.add_argument("--compile", action="store_true")
    result.add_argument("--dagger-rounds", type=int, default=2)
    result.add_argument("--dagger-episodes", type=int, default=180)
    result.add_argument("--closed-loop-validation-episodes", type=int, default=40)
    result.add_argument("--ppo-updates", type=int, default=20)
    result.add_argument("--ppo-episodes-per-update", type=int, default=24)
    result.add_argument("--ppo-epochs", type=int, default=4)
    result.add_argument("--ppo-minibatch", type=int, default=4096)
    result.add_argument("--ppo-action-std", type=float, default=0.08)
    result.add_argument("--ppo-clip", type=float, default=0.2)
    result.add_argument("--ppo-target-kl", type=float, default=0.02)
    result.add_argument("--time-penalty-per-second", type=float, default=0.5)
    result.add_argument("--comparison-time-penalty-per-second", type=float, default=0.05)
    return result


def validate_args(args: argparse.Namespace) -> None:
    if min(args.train_episodes, args.validation_episodes, args.test_episodes) <= 0:
        raise ValueError("Every drive split needs at least one episode")
    if (args.workers <= 0 or args.episode_seconds <= 0 or args.epochs <= 0 or args.patience <= 0
            or args.dagger_rounds < 2 or args.dagger_episodes <= 0
            or args.closed_loop_validation_episodes <= 0 or args.ppo_updates <= 0
            or args.ppo_episodes_per_update <= 0 or args.ppo_epochs <= 0
            or args.ppo_minibatch <= 0 or args.ppo_action_std <= 0
            or not 0 < args.ppo_clip < 1 or args.ppo_target_kl <= 0
            or args.time_penalty_per_second < 0 or args.comparison_time_penalty_per_second < 0):
        raise ValueError("workers, duration, epochs, and patience must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full drive run. Use --device cpu --cpu-smoke for development.")
    if args.device == "cpu" and not args.cpu_smoke:
        raise RuntimeError("CPU training is limited to --cpu-smoke")
    if args.cpu_smoke and (args.train_episodes > 16 or args.validation_episodes > 6 or args.test_episodes > 6):
        raise ValueError("CPU smoke runs are capped at 16/6/6 episodes")
    if args.cpu_smoke and args.dagger_episodes > 6:
        raise ValueError("CPU smoke DAgger rounds are capped at 6 episodes")
    if args.cpu_smoke and (args.ppo_updates > 3 or args.ppo_episodes_per_update > 4):
        raise ValueError("CPU smoke PPO is capped at 3 updates and 4 episodes per update")
    if args.compile and args.device != "cuda":
        raise ValueError("--compile requires CUDA")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def action_metrics(prediction: torch.Tensor, target: torch.Tensor, config: DriveConfig) -> dict:
    error = (prediction - target).abs()
    physical = error * error.new_tensor((config.max_forward_m_s, config.max_yaw_rad_s))
    return {
        "normalized_mae": float(error.mean()),
        "normalized_p95": float(torch.quantile(error.flatten(), 0.95)),
        "forward_mae_m_s": float(physical[:, 0].mean()),
        "forward_p95_m_s": float(torch.quantile(physical[:, 0], 0.95)),
        "yaw_mae_rad_s": float(physical[:, 1].mean()),
        "yaw_p95_rad_s": float(torch.quantile(physical[:, 1], 0.95)),
    }


@torch.inference_mode()
def validate(model, observation: torch.Tensor, target: torch.Tensor,
             config: DriveConfig) -> tuple[float, dict]:
    model.eval()
    prediction = model(observation)
    loss = F.mse_loss(prediction, target)
    return float(loss), action_metrics(prediction, target, config)


def closed_loop_summary(statuses: list[dict]) -> dict:
    success = np.asarray([status["success"] for status in statuses], dtype=bool)
    distances = np.asarray([status["distance_m"] for status in statuses])
    headings = np.asarray([status["heading_error_rad"] for status in statuses])
    success_seconds = [status["seconds"] for status in statuses if status["success"]]
    probability = float(success.mean())
    count = len(statuses)
    z = 1.96
    lower = (probability + z*z/(2*count) - z*np.sqrt(
        probability*(1-probability)/count + z*z/(4*count*count))) / (1 + z*z/count)
    return {
        "episodes": count, "reach_rate": probability,
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


def train_phase(model, training_model, train_observation_np: np.ndarray,
                train_target_np: np.ndarray, validation_observation: torch.Tensor,
                validation_target: torch.Tensor, config: DriveConfig,
                args: argparse.Namespace, device: torch.device, round_index: int,
                history: list[dict]) -> tuple[float, dict, int, float]:
    dataset = TensorDataset(torch.from_numpy(train_observation_np), torch.from_numpy(train_target_np))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 11 + round_index),
        num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-5, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    phase_best_path = args.output / f"action_best_round_{round_index}.pt"
    phase_best, stale, completed, transfer_seconds = float("inf"), 0, 0, 0.0
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        training_model.train()
        losses = []
        for observation, target in loader:
            before_transfer = time.perf_counter()
            observation = observation.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            transfer_seconds += time.perf_counter() - before_transfer
            distance = observation[:, :2].norm(dim=1) * config.goal_scale_m
            weight = 1 + 2 * (distance < 0.12).float() + 4 * (distance < 0.02).float()
            prediction = training_model(observation)
            loss = ((prediction - target).square().mean(1) * weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation_loss, metrics = validate(model, validation_observation, validation_target, config)
        record = {"dagger_round": round_index, "epoch": epoch + 1,
                  "training_transitions": len(dataset),
                  "train_weighted_mse": float(np.mean(losses)),
                  "validation_mse": validation_loss, **metrics,
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "seconds": time.perf_counter() - epoch_started}
        history.append(record)
        completed = epoch + 1
        if validation_loss < phase_best - 1e-7:
            phase_best, stale = validation_loss, 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "validation_mse": phase_best}, phase_best_path)
        else:
            stale += 1
        (args.output / "history.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in history))
        atomic_json(args.output / "progress.json", {
            "phase": "training", **record, "phase_best_validation_mse": phase_best,
            "stale_epochs": stale})
        if epoch % 10 == 0 or stale == 0:
            print(json.dumps(record, allow_nan=False), flush=True)
        if stale >= args.patience:
            break
    checkpoint = torch.load(phase_best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    validation_loss, metrics = validate(model, validation_observation, validation_target, config)
    return validation_loss, metrics, completed, transfer_seconds


class DriveCritic(nn.Module):
    def __init__(self, config: DriveConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.observation_dim, config.hidden_dim), nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation).squeeze(-1)


def normal_log_probability(action: torch.Tensor, mean: torch.Tensor, std: float) -> torch.Tensor:
    variance = std * std
    return (-0.5 * ((action - mean).square() / variance +
                    np.log(2 * np.pi * variance))).sum(-1)


def generalized_advantage(rewards: torch.Tensor, dones: torch.Tensor,
                          values: torch.Tensor, gamma: float = 0.995,
                          gae_lambda: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    carry = rewards.new_zeros(())
    next_value = rewards.new_zeros(())
    for index in reversed(range(len(rewards))):
        continuation = (~dones[index]).float()
        delta = rewards[index] + gamma * next_value * continuation - values[index]
        carry = delta + gamma * gae_lambda * continuation * carry
        advantages[index] = carry
        next_value = values[index]
    return advantages, advantages + values


def ppo_finetune(actor: BestDriveMLP, base_state: dict, config: DriveConfig,
                 args: argparse.Namespace, device: torch.device,
                 time_penalty: float, label: str) -> tuple[dict, list[dict]]:
    """Fine-tune one time-cost setting on rewards executed in native physics."""
    torch.manual_seed(args.seed + 900_301)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 900_301)
    actor.load_state_dict(base_state)
    critic = DriveCritic(config).to(device)
    optimizer = torch.optim.AdamW([
        {"params": actor.parameters(), "lr": args.learning_rate * 0.02},
        {"params": critic.parameters(), "lr": args.learning_rate * 0.25},
    ], weight_decay=1e-5, fused=device.type == "cuda")
    updates = []
    rollout_rng = np.random.default_rng(args.seed + 700_301)
    current_path = args.output / f"ppo_{label}_current.npz"
    for update in range(args.ppo_updates):
        export_drive_policy(actor, config, current_path)
        seeds = rollout_rng.integers(
            0, np.iinfo(np.int32).max, args.ppo_episodes_per_update).tolist()
        collection_started = time.perf_counter()
        rollout = collect_ppo_rollouts(
            current_path, seeds, config, args.workers,
            args.seed + 800_301 + update * 1000,
            args.ppo_action_std, time_penalty)
        collection_seconds = time.perf_counter() - collection_started
        observation = torch.from_numpy(rollout["observations"]).to(device)
        action = torch.from_numpy(rollout["actions"]).to(device)
        rewards = torch.from_numpy(rollout["rewards"]).to(device)
        dones = torch.from_numpy(rollout["dones"]).to(device)
        with torch.no_grad():
            old_mean = actor(observation)
            old_log_probability = normal_log_probability(action, old_mean, args.ppo_action_std)
            old_values = critic(observation)
            advantages, returns = generalized_advantage(rewards, dones, old_values)
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        sample_count = len(observation)
        stopped_for_kl = False
        observed_kl = 0.0
        rejected_kl = None
        update_started = time.perf_counter()
        for _ in range(args.ppo_epochs):
            actor_before_epoch = {name: value.detach().clone()
                                  for name, value in actor.state_dict().items()}
            indices = torch.randperm(sample_count, device=device)
            for start in range(0, sample_count, args.ppo_minibatch):
                batch = indices[start:start + args.ppo_minibatch]
                mean = actor(observation[batch])
                log_probability = normal_log_probability(action[batch], mean, args.ppo_action_std)
                ratio = (log_probability - old_log_probability[batch]).exp()
                unclipped = ratio * advantages[batch]
                clipped = ratio.clamp(1 - args.ppo_clip, 1 + args.ppo_clip) * advantages[batch]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                value = critic(observation[batch])
                critic_loss = F.mse_loss(value, returns[batch])
                loss = actor_loss + 0.5 * critic_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 2.0)
                optimizer.step()
            with torch.no_grad():
                new_log_probability = normal_log_probability(
                    action, actor(observation), args.ppo_action_std)
                log_ratio = new_log_probability - old_log_probability
                observed_kl = float(((log_ratio.exp() - 1) - log_ratio).mean())
            if observed_kl > args.ppo_target_kl:
                rejected_kl = observed_kl
                actor.load_state_dict(actor_before_epoch)
                for parameter in actor.parameters():
                    optimizer.state.pop(parameter, None)
                with torch.no_grad():
                    new_log_probability = normal_log_probability(
                        action, actor(observation), args.ppo_action_std)
                    log_ratio = new_log_probability - old_log_probability
                    observed_kl = float(((log_ratio.exp() - 1) - log_ratio).mean())
                stopped_for_kl = True
                break
        status = closed_loop_summary(rollout["status"])
        reward_components = {
            name: float(np.mean([episode["reward_components"][name]
                                 for episode in rollout["status"]]))
            for name in ("goal_progress", "time", "success", "timeout", "failure")
        }
        record = {
            "penalty_label": label, "time_penalty_per_second": time_penalty,
            "update": update + 1, "episodes": len(seeds), "transitions": sample_count,
            "mean_return": float(np.mean([episode["return"] for episode in rollout["status"]])),
            "reward_components_mean_per_episode": reward_components,
            "first_timestamp_s": float(rollout["timestamps"].min()),
            "last_timestamp_s": float(rollout["timestamps"].max()),
            "rollout_reach_rate": status["reach_rate"],
            "rollout_mean_success_seconds": status["mean_success_seconds"],
            "approximate_kl": observed_kl, "kl_stopped": stopped_for_kl,
            "rejected_approximate_kl": rejected_kl,
            "collection_seconds": collection_seconds,
            "optimization_seconds": time.perf_counter() - update_started,
        }
        updates.append(record)
        atomic_json(args.output / "progress.json", {"phase": "ppo", **record})
        print(json.dumps({"phase": "ppo", **record}, allow_nan=False), flush=True)
    artifact = args.output / f"ppo_{label}.npz"
    artifact_hash = export_drive_policy(actor, config, artifact)
    return {"artifact": artifact.name, "artifact_sha256": artifact_hash,
            "time_penalty_per_second": time_penalty,
            "actor_state": {name: value.detach().cpu().clone()
                            for name, value in actor.state_dict().items()}}, updates


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    seed_all(args.seed)
    overall_started = time.perf_counter()
    config = DriveConfig(episode_seconds=args.episode_seconds)
    split_episodes = {"train": args.train_episodes, "validation": args.validation_episodes,
                      "test": args.test_episodes}
    split_seeds = {"train": args.seed + 301, "validation": args.seed + 100_301,
                   "test": args.seed + 200_301}
    collection_started = time.perf_counter()
    manifest = collect_drive_dataset(
        args.dataset, config, split_episodes, split_seeds, args.workers,
        source_provenance(), force=args.force_generate)
    collection_seconds = time.perf_counter() - collection_started
    print(json.dumps({"phase": "drive_dataset_ready", "seconds_this_run": collection_seconds,
                      "transitions": {name: record["transitions"]
                                      for name, record in manifest["splits"].items()}}), flush=True)
    if args.generate_only:
        return
    if manifest["splits"]["validation"]["teacher_reach_rate"] < 0.9:
        raise RuntimeError("The held-out validation teacher reached fewer than 90% of drive goals")

    args.output.mkdir(parents=True, exist_ok=True)
    run_config = {
        "drive_config": asdict(config), "dataset": str(args.dataset.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset / "manifest.json"),
        "seed": args.seed, "device": args.device, "epochs": args.epochs,
        "patience": args.patience, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "compile": args.compile,
        "dagger_rounds": args.dagger_rounds,
        "dagger_episodes": args.dagger_episodes,
        "closed_loop_validation_episodes": args.closed_loop_validation_episodes,
        "ppo_updates": args.ppo_updates,
        "ppo_episodes_per_update": args.ppo_episodes_per_update,
        "ppo_epochs": args.ppo_epochs,
        "ppo_clip": args.ppo_clip,
        "ppo_target_kl": args.ppo_target_kl,
        "time_penalties_per_second": [args.comparison_time_penalty_per_second,
                                      args.time_penalty_per_second],
    }
    atomic_json(args.output / "config.json", run_config)
    train_observation_np, train_target_np = load_drive_split(args.dataset, "train")
    validation_observation_np, validation_target_np = load_drive_split(args.dataset, "validation")
    device = torch.device(args.device)
    validation_observation = torch.from_numpy(validation_observation_np).to(device)
    validation_target = torch.from_numpy(validation_target_np).to(device)
    model = BestDriveMLP(config).to(device)
    training_model = model
    compile_seconds = 0.0
    if args.compile:
        started = time.perf_counter()
        training_model = torch.compile(model)
        training_model(torch.zeros((args.batch_size, config.observation_dim), device=device))
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - started
    history, candidates, dagger_records = [], [], []
    selected_score = None
    selected_round = None
    selected_path = args.output / "closed_loop_best.pt"
    validation_seeds = manifest["splits"]["validation"]["episode_seeds"][
        :min(args.closed_loop_validation_episodes, args.validation_episodes)]
    transfer_seconds = 0.0
    training_started = time.perf_counter()
    for round_index in range(args.dagger_rounds + 1):
        if round_index:
            dagger_rng = np.random.default_rng(args.seed + 400_301 + round_index)
            dagger_seeds = dagger_rng.integers(
                0, np.iinfo(np.int32).max, args.dagger_episodes).tolist()
            dagger_started = time.perf_counter()
            dagger_observation, dagger_target, dagger_status = collect_dagger_round(
                args.output / f"candidate_round_{round_index - 1}.npz", dagger_seeds,
                config, args.workers, args.seed + 500_301 + round_index * 1000)
            train_observation_np = np.concatenate((train_observation_np, dagger_observation))
            train_target_np = np.concatenate((train_target_np, dagger_target))
            dagger_path = args.output / f"dagger_round_{round_index}.npz"
            temporary = dagger_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, observations=dagger_observation,
                                    targets=dagger_target,
                                    episode_seeds=np.asarray(dagger_seeds, dtype=np.int64))
            temporary.replace(dagger_path)
            dagger_record = {
                "round": round_index, "episodes": len(dagger_seeds),
                "transitions": len(dagger_observation),
                "learner_reach_rate_before_update": float(np.mean(
                    [status["success"] for status in dagger_status])),
                "terminal_failures": int(sum(
                    status.get("failure_reason") is not None for status in dagger_status)),
                "sha256": sha256_file(dagger_path),
                "seconds": time.perf_counter() - dagger_started,
            }
            dagger_records.append(dagger_record)
            print(json.dumps({"phase": "dagger_collection", **dagger_record}), flush=True)
        validation_loss, metrics, epochs_completed, phase_transfer = train_phase(
            model, training_model, train_observation_np, train_target_np,
            validation_observation, validation_target, config, args, device,
            round_index, history)
        transfer_seconds += phase_transfer
        candidate_path = args.output / f"candidate_round_{round_index}.npz"
        candidate_hash = export_drive_policy(model, config, candidate_path)
        closed_loop_started = time.perf_counter()
        statuses = evaluate_drive_policy(
            candidate_path, validation_seeds, config, args.workers,
            domain_seed=args.seed + 600_301, policy_kind="learned")
        closed_loop = closed_loop_summary(statuses)
        closed_loop["seconds"] = time.perf_counter() - closed_loop_started
        candidate = {"round": round_index, "artifact": candidate_path.name,
                     "artifact_sha256": candidate_hash,
                     "epochs_completed": epochs_completed,
                     "validation_action_mse": validation_loss,
                     "validation_action_metrics": metrics,
                     "closed_loop_validation": closed_loop}
        candidates.append(candidate)
        score = (closed_loop["reach_rate"], -closed_loop["p95_final_distance_m"],
                 -closed_loop["p95_final_heading_error_rad"], -validation_loss)
        if selected_score is None or score > selected_score:
            selected_score, selected_round = score, round_index
            torch.save({"model": model.state_dict(), "round": round_index,
                        "candidate": candidate}, selected_path)
        atomic_json(args.output / "progress.json", {
            "phase": "closed_loop_validation", "candidate": candidate,
            "selected_round": selected_round})
        print(json.dumps({"phase": "closed_loop_validation", **candidate}, allow_nan=False), flush=True)
    checkpoint = torch.load(selected_path, map_location=device, weights_only=False)
    base_state = checkpoint["model"]
    ppo_candidates, ppo_updates = [], []
    selected_bc_candidate = next(item for item in candidates if item["round"] == selected_round)
    base_summary = selected_bc_candidate["closed_loop_validation"]
    base_completion = base_summary["mean_success_seconds"]
    selected_final_score = (
        base_summary["reach_rate"],
        -(base_completion if base_completion is not None else float("inf")),
        -base_summary["p95_final_distance_m"],
        -base_summary["p95_final_heading_error_rad"],
    )
    selected_final_state = base_state
    selected_final = {"kind": "bc_dagger", "round": selected_round}
    penalties = (("baseline", args.comparison_time_penalty_per_second),
                 ("requested", args.time_penalty_per_second))
    if penalties[0][1] == penalties[1][1]:
        penalties = penalties[:1]
    for label, penalty in penalties:
        ppo_candidate, updates = ppo_finetune(
            model, base_state, config, args, device, penalty, label)
        actor_state = ppo_candidate.pop("actor_state")
        candidate_path = args.output / ppo_candidate["artifact"]
        validation_started = time.perf_counter()
        statuses = evaluate_drive_policy(
            candidate_path, validation_seeds, config, args.workers,
            domain_seed=args.seed + 600_301, policy_kind="learned")
        summary = closed_loop_summary(statuses)
        summary["seconds"] = time.perf_counter() - validation_started
        ppo_candidate["closed_loop_validation"] = summary
        ppo_candidate["updates"] = len(updates)
        ppo_candidates.append(ppo_candidate)
        ppo_updates.extend(updates)
        completion = summary["mean_success_seconds"]
        score = (summary["reach_rate"], -(completion if completion is not None else float("inf")),
                 -summary["p95_final_distance_m"],
                 -summary["p95_final_heading_error_rad"])
        if score > selected_final_score:
            selected_final_score = score
            selected_final_state = actor_state
            selected_final = {"kind": "ppo", "time_penalty": label,
                              "time_penalty_per_second": penalty}
        print(json.dumps({"phase": "ppo_validation", **ppo_candidate}, allow_nan=False), flush=True)
    model.load_state_dict(selected_final_state)
    training_seconds = time.perf_counter() - training_started
    validation_loss, validation_metrics = validate(
        model, validation_observation, validation_target, config)
    weights_path = args.output / "weights.npz"
    artifact_hash = export_drive_policy(model, config, weights_path)
    report = {
        "schema_version": 1,
        "artifact": weights_path.name, "artifact_sha256": artifact_hash,
        "architecture": [config.observation_dim, config.hidden_dim,
                         config.hidden_dim, config.hidden_dim // 2, 2],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "api_output": "normalized forward and yaw, scaled to 0.35 m/s and 2.5 rad/s",
        "drive_config": asdict(config), "seed": args.seed,
        "torch_version": torch.__version__, "mujoco_version": mujoco.__version__,
        "device": str(device),
        "gpu": ({"name": torch.cuda.get_device_name(device),
                 "capability": list(torch.cuda.get_device_capability(device)),
                 "cuda_runtime": torch.version.cuda,
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
                if device.type == "cuda" else None),
        "cuda_optimizations": {"fused_adamw": device.type == "cuda",
                               "tf32_allowed": True, "torch_compile": args.compile,
                               "compile_seconds": compile_seconds,
                               "host_to_device_seconds": transfer_seconds},
        "timing": {"dataset_seconds_this_run": collection_seconds,
                   "training_seconds": training_seconds,
                   "total_seconds": time.perf_counter() - overall_started},
        "dataset_manifest_sha256": run_config["dataset_manifest_sha256"],
        "dataset_split_sha256": {split: record["sha256"]
                                 for split, record in manifest["splits"].items()},
        "source_provenance": source_provenance(),
        "selection": {"criterion": "highest fixed-seed reach rate, then lowest successful completion time and p95 pose errors",
                      "selected_bc_dagger_round": selected_round,
                      "selected_final": selected_final,
                      "validation_action_mse": validation_loss,
                      "bc_dagger_candidates": candidates,
                      "ppo_time_penalty_candidates": ppo_candidates,
                      "patience_per_round": args.patience},
        "dagger": {"rounds": args.dagger_rounds,
                   "episodes_per_round": args.dagger_episodes,
                   "records": dagger_records},
        "ppo": {"updates_per_candidate": args.ppo_updates,
                "episodes_per_update": args.ppo_episodes_per_update,
                "clip_ratio": args.ppo_clip,
                "target_kl": args.ppo_target_kl,
                "action_std": args.ppo_action_std,
                "reward": {"goal_progress_per_meter": 20.0,
                           "heading_progress_per_radian": 0.35,
                           "success_bonus": 10.0,
                           "timeout_penalty": -20.0,
                           "terminal_failure_penalty": -25.0,
                           "time_penalties_per_second": [value for _, value in penalties]},
                "updates": ppo_updates},
        "validation_action_metrics": validation_metrics,
        "validation_teacher_physics": {
            "episodes": manifest["splits"]["validation"]["episodes"],
            "reach_rate": manifest["splits"]["validation"]["teacher_reach_rate"],
            "maximum_actual_torque_nm": manifest["splits"]["validation"]["maximum_actual_torque_nm"],
            "maximum_tilt_rad": manifest["splits"]["validation"]["maximum_tilt_rad"],
        },
        "test_evaluation": "Not read during training. Run scripts/test_best_drive.py after selection.",
        "claim_scope": "The drive model covers precision line and turn primitives in the empty center corridor. The geometric route planner, gripping, and full mission remain separate.",
    }
    atomic_json(args.output / "training.json", report)
    atomic_json(args.output / "progress.json", {"phase": "complete", "artifact": str(weights_path),
                                                 "artifact_sha256": artifact_hash,
                                                 "selected_bc_dagger_round": selected_round})
    print("DRIVE_TRAINING_REPORT " + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
