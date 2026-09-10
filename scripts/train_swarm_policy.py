#!/usr/bin/env python3
"""Train shared attention MAPPO on measured MuJoCo swarm transitions."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_mujoco.swarm_policy import PolicyConfig, export_policy, make_actor_critic


STAGES = [
    {"num_active_robots": 2, "num_active_objects": 1, "difficulty": 0.0},
    {"num_active_robots": 4, "num_active_objects": 1, "difficulty": 0.25},
    {"num_active_robots": 8, "num_active_objects": 2, "difficulty": 0.55},
    {"num_active_robots": 40, "num_active_objects": 4, "difficulty": 1.0},
]

ACTION_NAMES = ("left_wheel", "right_wheel", "lift", "magnet_enable")
TRAINING_SOURCES = (
    "arena_mujoco/swarm_env.py", "arena_mujoco/swarm_body_env.py",
    "arena_mujoco/swarm_flow.py", "arena_mujoco/swarm_magnets.py",
    "arena_mujoco/swarm_robot.py", "arena_mujoco/swarm_policy.py",
    "arena_mujoco/builder.py", "arena_mujoco/materials.py", "arena_spec.json",
    "scripts/train_swarm_policy.py", "scripts/evaluate_swarm_training.py",
)


def tensor_obs(obs, device):
    return {key: torch.as_tensor(value, device=device, dtype=torch.float32) for key, value in obs.items()}


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def compute_gae(reward, value, next_value, terminated, truncated, gamma=.99, gae_lambda=.95):
    """Bootstrap time limits using terminal states, never the next episode."""
    advantage = torch.zeros_like(reward)
    carry = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * next_value[t] * (~terminated[t]).unsqueeze(-1) - value[t]
        continuation = (~(terminated[t] | truncated[t])).unsqueeze(-1)
        carry = delta + gamma * gae_lambda * continuation * carry
        advantage[t] = carry
    return advantage, advantage + value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_source_provenance():
    """Identify the exact code that generates training and terminal evaluations."""
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": {name: file_sha256(ROOT / name) for name in TRAINING_SOURCES},
    }


def progress(args, record):
    save_json(Path(args.output) / "progress.json", record)
    print(json.dumps(record, allow_nan=False), flush=True)


def make_env(args, device, seed, num_envs=None):
    module, factory = args.env_factory.split(":")
    return getattr(importlib.import_module(module), factory)(
        num_envs=num_envs or args.num_envs, num_robots=args.robots,
        num_objects=args.objects, device="cpu" if args.backend == "native" else str(device), backend=args.backend, seed=seed,
        episode_seconds=args.episode_seconds,
        native_workers=args.native_workers,
    )


def stage_spec(stage, args):
    spec = dict(STAGES[stage])
    spec["num_active_robots"] = min(args.robots, spec["num_active_robots"])
    spec["num_active_objects"] = min(args.objects, spec["num_active_objects"])
    return spec


def full_stage_validation_pass(result, stage, success_threshold):
    return (stage == 3 and result["complete"] and result["episodes"] >= 32
            and result["success_rate"] >= success_threshold
            and result["success_wilson_lower_95"] >= .6)


def acceptance_passed(args, stage, cumulative_actor_update, final_eval, zero_eval):
    return (not args.cpu_smoke and args.robots == 40 and args.objects >= 2 and stage == 3
            and cumulative_actor_update > 1e-6
            and final_eval["complete"] and final_eval["episodes"] >= 32
            and final_eval["success_rate"] >= args.final_success
            and final_eval["success_wilson_lower_95"] >= .6
            and final_eval["success_rate"] > zero_eval["success_rate"] + .2)


@torch.no_grad()
def evaluate(model, env, stage, args, device, policy="learned"):
    required_episodes = args.eval_episodes if stage == 3 else args.stage_eval_episodes
    env.set_curriculum(stage_spec(stage, args))
    obs = tensor_obs(env.reset(), device)
    completed, successes, deliveries, returns = [], [], [], []
    world_count = obs["active"].shape[0]
    # Fix each world's sample size before rollout so fast successes cannot
    # replace slower failures in the validation sample.
    quotas = [required_episodes // world_count + int(world < required_episodes % world_count)
              for world in range(world_count)]
    counted = [0] * world_count
    episode_return = torch.zeros(obs["active"].shape[0], device=device)
    episode_control_steps = torch.zeros(world_count, dtype=torch.long, device=device)
    peak_component = torch.full((world_count,), -float("inf"), device=device)
    body_episodes = defaultdict(list)
    # A bounded evaluation must not hang if an environment fails to terminate.
    batches = (required_episodes + obs["active"].shape[0] - 1) // obs["active"].shape[0]
    limit = args.eval_max_steps if args.eval_max_steps > 0 else batches * env.max_steps
    for step in range(limit):
        if step % 128 == 0:
            progress(args, {"phase": "evaluation", "policy": policy, "stage": stage,
                            "control_step": step, "episodes": len(completed), "required_episodes": required_episodes})
        if policy == "zero":
            actions = torch.zeros((*obs["active"].shape, model.config.action_dim), device=device)
        elif policy == "teacher":
            actions = torch.as_tensor(env.teacher_action(), device=device, dtype=torch.float32)
        else:
            actions = model.act(obs, deterministic=True)[0]
        active = obs["active"]
        raw_obs, reward, terminated, truncated, info = env.step(actions)
        obs = tensor_obs(raw_obs, device)
        reward = torch.as_tensor(reward, device=device)
        done = torch.as_tensor(terminated, device=device).bool() | torch.as_tensor(truncated, device=device).bool()
        episode_return += (reward * active).sum(-1) / active.sum(-1).clamp_min(1)
        episode_control_steps += 1
        component = None
        if "largest_magnetic_component" in info:
            component = torch.as_tensor(info["largest_magnetic_component"], device=device)
            peak_component = torch.maximum(peak_component, component)
        if done.any():
            success = torch.as_tensor(info["success"], device=device)
            delivered = torch.as_tensor(info["delivered"], device=device)
            terminal_metrics = {
                "connected_control_steps": info.get("connected_control_steps"),
                "terminal_all_robot_min_travel_m": info.get("all_robot_min_travel"),
                "magnetic_link_formations": info.get("magnetic_link_formations"),
                "magnetic_link_releases": info.get("magnetic_link_releases"),
                "initial_connected_seconds": info.get("initial_connected_seconds"),
                "middle_connected_control_fraction": info.get("middle_connected_control_fraction"),
                "new_magnetic_neighbor_pairs": info.get("new_magnetic_neighbor_pairs"),
            }
            terminal_metrics = {name: torch.as_tensor(value, device=device)
                                for name, value in terminal_metrics.items() if value is not None}
            for idx in torch.where(done)[0].tolist():
                if counted[idx] >= quotas[idx]:
                    continue
                counted[idx] += 1
                completed.append(idx)
                successes.append(float(success[idx]))
                deliveries.append(float(delivered[idx]))
                returns.append(float(episode_return[idx]))
                if component is not None:
                    body_episodes["peak_largest_magnetic_component"].append(float(peak_component[idx]))
                for name, values in terminal_metrics.items():
                    body_episodes[name].append(float(values[idx]))
                if "connected_control_steps" in terminal_metrics:
                    connected_steps = float(terminal_metrics["connected_control_steps"][idx])
                    body_episodes["connected_control_fraction"].append(
                        connected_steps / max(int(episode_control_steps[idx]), 1))
            if counted == quotas:
                break
            episode_return[done] = 0
            episode_control_steps[done] = 0
            peak_component[done] = -float("inf")
            obs = tensor_obs(env.reset_done(done), device)
    if not completed:
        raise RuntimeError(f"No evaluation episode terminated within {limit} physics control steps")
    p = float(np.mean(successes))
    n = len(completed)
    # Wilson lower bound prevents a single lucky episode from passing the gate.
    z = 1.96
    lower = (p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
    result = {"policy": policy, "stage": stage, "episodes": n, "success_rate": p,
              "success_wilson_lower_95": float(lower), "mean_delivered": float(np.mean(deliveries)),
              "mean_return": float(np.mean(returns)), "complete": counted == quotas,
              "required_episodes": required_episodes, "episode_quotas": quotas,
              "episodes_per_world": counted}
    if body_episodes:
        result["magnetic_body"] = {
            "episode_counts": {name: len(values) for name, values in body_episodes.items()},
            **{f"mean_episode_{name}": float(np.mean(values))
               for name, values in body_episodes.items()},
        }
    return result


def load_exported_actor(path, device):
    """Load an actor-only NPZ into a Torch model for deterministic evaluation."""
    with np.load(path, allow_pickle=False) as archive:
        config = PolicyConfig(**json.loads(str(archive["config"])))
        state = {key: torch.as_tensor(archive[key], device=device)
                 for key in archive.files if key != "config"}
    model = make_actor_critic(config).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith("critic_") for name in missing):
        raise ValueError("Exported actor weights do not match their policy configuration")
    model.eval()
    return model, config


def run_final_evaluations(model, config, args, device):
    """Run held-out, zero-action, and saved warmstart evaluations."""
    audit_env = make_env(args, device, args.seed + 200000, min(args.num_envs, args.eval_episodes))
    try:
        final_eval = evaluate(model, audit_env, 3, args, device)
    finally:
        if hasattr(audit_env, "close"):
            audit_env.close()
    baseline_env = make_env(args, device, args.seed + 200000, min(args.num_envs, args.eval_episodes))
    try:
        zero_eval = evaluate(model, baseline_env, 3, args, device, policy="zero")
    finally:
        if hasattr(baseline_env, "close"):
            baseline_env.close()
    warmstart_eval = None
    warmstart_path = Path(args.output) / "warmstart.npz"
    if warmstart_path.exists():
        warmstart_model, warmstart_config = load_exported_actor(warmstart_path, device)
        if warmstart_config != config:
            raise ValueError("Warmstart and final actor configurations differ")
        warmstart_env = make_env(args, device, args.seed + 200000,
                                 min(args.num_envs, args.eval_episodes))
        try:
            warmstart_eval = evaluate(warmstart_model, warmstart_env, 3, args, device)
        finally:
            if hasattr(warmstart_env, "close"):
                warmstart_env.close()
        warmstart_eval["policy"] = "warmstart"
    return final_eval, zero_eval, warmstart_eval


MAX_DEMO_SAMPLES = 2_000_000


def demo_weights(data, balance_roles=False):
    weight = data.get("learning_weight", torch.ones(len(data["local"]), device=data["local"].device)).clone()
    if balance_roles:
        if data["local"].shape[-1] not in (32, 40, 42) or "phase" not in data:
            raise ValueError("Role balancing requires a swarm observation and phase labels")
        carrier = data["local"][:, 15] > .5
        weight.zero_()
        if carrier.any():
            phases = data["phase"][carrier].long()
            _, inverse, counts = torch.unique(phases, return_inverse=True, return_counts=True)
            weight[carrier] = 1. / (len(counts) * counts[inverse].float())
        if (~carrier).any():
            weight[~carrier] = 1. / (~carrier).sum()
    elif "phase" in data:
        phases = data["phase"].long()
        counts = torch.bincount(phases, minlength=int(phases.max()) + 1).clamp_min(1)
        weight *= len(phases) / (len(counts) * counts[phases])
    return weight / weight.mean().clamp_min(1e-8)


@torch.no_grad()
def collect_demonstrations(model, env, args, device, steps, teacher_prob=1., dagger_round=None):
    """Label real visited states; only collection may execute teacher actions."""
    obs = tensor_obs(env.reset(), device)
    samples, targets, successes, deliveries = [], [], [], []
    teacher_world_steps = 0
    noise = args.demo_noise if dagger_round is None else 0.
    phase = "demonstrations" if dagger_round is None else "dagger_collection"
    for step in range(steps):
        if step % 128 == 0:
            progress(args, {"phase": phase, "round": dagger_round, "control_step": step,
                            "total_control_steps": steps, "teacher_probability": teacher_prob,
                            "action_noise_std": noise})
        target = torch.as_tensor(env.teacher_action(), device=device, dtype=torch.float32)
        active = obs["active"].bool()
        samples.append({key: value[active].clone() for key, value in obs.items() if key != "global"})
        targets.append(target[active].clone())
        worlds = active.shape[0]
        if teacher_prob == 1:
            action = target
            teacher_world_steps += worlds
        else:
            learner = torch.tanh(model.mean(obs)) * active.unsqueeze(-1)
            if teacher_prob == 0:
                action = learner
            else:
                use_teacher = torch.rand((worlds, 1, 1), device=device) < teacher_prob
                teacher_world_steps += int(use_teacher.sum())
                action = torch.where(use_teacher, target, learner)
        if dagger_round is None or noise:
            action = (action + torch.randn_like(action) * noise).clamp(-1, 1)
        next_obs, _, terminated, truncated, info = env.step(action)
        done = torch.as_tensor(terminated, device=device).bool() | torch.as_tensor(truncated, device=device).bool()
        if done.any():
            successes.extend(torch.as_tensor(info["success"], device=device)[done].tolist())
            deliveries.extend(torch.as_tensor(info["delivered"], device=device)[done].tolist())
        obs = tensor_obs(env.reset_done(done) if done.any() else next_obs, device)
    if not samples:
        return None
    data = {key: torch.cat([sample[key] for sample in samples]) for key in samples[0]}
    target = torch.cat(targets)
    report = {"physics_control_steps": steps * worlds, "agent_examples": len(target),
              "completed_episodes": len(successes), "successful_episodes": int(sum(successes)),
              "success_rate": float(np.mean(successes)) if successes else None,
              "mean_delivered": float(np.mean(deliveries)) if deliveries else None,
              "teacher_probability": teacher_prob, "teacher_world_steps": teacher_world_steps,
              "learner_world_steps": steps * worlds - teacher_world_steps,
              "learner_deterministic": True, "action_noise_std": noise}
    return {"obs": data, "target": target, "weight": demo_weights(data, args.balance_demo_roles), "report": report}


@torch.no_grad()
def demonstration_errors(model, dataset, batch_size):
    """Measure all retained samples, separating carriers' phases from formation."""
    totals = {}
    weighted_error = torch.zeros((), device=dataset["target"].device)
    for start in range(0, len(dataset["target"]), batch_size):
        batch = {key: value[start:start + batch_size] for key, value in dataset["obs"].items()}
        error = (torch.tanh(model.mean(batch)) - dataset["target"][start:start + batch_size]).square()
        weight = dataset["weight"][start:start + batch_size]
        weighted_error += (error.mean(-1) * weight).sum()
        groups = {"all": torch.ones(len(error), dtype=torch.bool, device=error.device)}
        if batch["local"].shape[-1] in (32, 40, 42) and "phase" in batch:
            carrier = batch["local"][:, 15] > .5
            groups["formation"] = ~carrier
            for phase in torch.unique(batch["phase"][carrier]).tolist():
                groups[f"carrier_phase_{int(phase)}"] = carrier & (batch["phase"] == phase)
        elif "phase" in batch:
            groups.update({f"phase_{int(phase)}": batch["phase"] == phase
                           for phase in torch.unique(batch["phase"]).tolist()})
        for name, mask in groups.items():
            if not mask.any():
                continue
            count, summed = totals.get(name, (0, torch.zeros(error.shape[-1], device=error.device)))
            totals[name] = (count + int(mask.sum()), summed + error[mask].sum(0))
    action_dim = dataset["target"].shape[-1]
    action_order = list(ACTION_NAMES[:action_dim])
    action_order.extend(f"action_{index}" for index in range(len(action_order), action_dim))
    return {"weighted_mse": float(weighted_error / dataset["weight"].sum()),
            "action_order": action_order,
            "groups": {name: {"examples": count, "per_action_mse": (summed / count).tolist()}
                       for name, (count, summed) in totals.items()}}


def fit_demonstrations(model, optimizer, dataset, args, updates, dagger_round=None):
    last_loss = None
    for update in range(updates):
        idx = torch.randint(len(dataset["target"]), (min(args.bc_batch_size, len(dataset["target"])),),
                            device=dataset["target"].device)
        batch = {key: value[idx] for key, value in dataset["obs"].items()}
        error = (torch.tanh(model.mean(batch)) - dataset["target"][idx]).square().mean(-1)
        loss = masked_mean(error, dataset["weight"][idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        last_loss = float(loss.detach())
        if update % 100 == 0:
            progress(args, {"phase": "behavior_cloning" if dagger_round is None else "dagger_fitting",
                            "round": dagger_round, "update": update + 1,
                            "total_updates": updates, "weighted_mse": last_loss})
    errors = demonstration_errors(model, dataset, args.bc_batch_size)
    return {"updates": updates, "final_mse": errors["weighted_mse"], "last_minibatch_mse": last_loss,
            "phase_balanced": "phase" in dataset["obs"], "roles_balanced": args.balance_demo_roles,
            "carrier_weighted": not args.balance_demo_roles and "learning_weight" in dataset["obs"],
            "fitting_errors": errors}


def warmstart(model, optimizer, env, args, device, demo_steps=None, bc_updates=None):
    demo_steps = args.demo_steps if demo_steps is None else demo_steps
    bc_updates = args.bc_updates if bc_updates is None else bc_updates
    dataset = collect_demonstrations(model, env, args, device, demo_steps)
    if dataset is not None:
        dataset["report"].update(fit_demonstrations(model, optimizer, dataset, args, bc_updates))
        progress(args, {"phase": "demonstration_summary", **dataset["report"]})
    return dataset


def merge_demonstrations(previous, incoming, balance_roles, limit=MAX_DEMO_SAMPLES):
    if previous is None:
        data, target = incoming["obs"], incoming["target"]
    else:
        data = {key: torch.cat((previous["obs"][key], incoming["obs"][key])) for key in incoming["obs"]}
        target = torch.cat((previous["target"], incoming["target"]))
    total = len(target)
    if total > limit:
        # Even spacing preserves temporal coverage without consuming training RNG.
        indices = torch.arange(limit, device=target.device) * (total - 1) // max(limit - 1, 1)
        data = {key: value[indices] for key, value in data.items()}
        target = target[indices]
    return {"obs": data, "target": target, "weight": demo_weights(data, balance_roles),
            "report": {"samples_before_cap": total, "agent_examples": len(target), "sample_cap": limit}}


def dagger_teacher_probability(round_index, rounds, initial_probability):
    return initial_probability * (rounds - round_index - 1) / (rounds - 1) if rounds > 1 else 0.


def dagger(model, optimizer, env, args, device, dataset):
    reports = []
    for round_index in range(args.dagger_rounds):
        probability = dagger_teacher_probability(round_index, args.dagger_rounds, args.dagger_teacher_prob)
        incoming = collect_demonstrations(model, env, args, device, args.dagger_steps,
                                          probability, round_index + 1)
        dataset = merge_demonstrations(dataset, incoming, args.balance_demo_roles)
        report = {"round": round_index + 1, "collection": incoming["report"],
                  "dataset": dataset["report"],
                  "fit": fit_demonstrations(model, optimizer, dataset, args, args.dagger_updates, round_index + 1)}
        reports.append(report)
        save_json(Path(args.output) / "dagger.json", {"rounds": reports})
        progress(args, {"phase": "dagger_round_complete", **report})
    return dataset, reports


def configure_initial_action_std(model, requested, resumed=False):
    if requested is not None and not resumed:
        with torch.no_grad():
            model.log_std.fill_(float(np.log(requested)))
    return {"requested": requested, "applied": requested is not None and not resumed,
            "source": "checkpoint" if resumed else "override" if requested is not None else "model_default",
            "actual_action_std": model.log_std.detach().clamp(-5, .5).exp().cpu().tolist()}


def ppo_update(model, optimizer, rollout, args, anchor=None):
    actor_before = {name: value.detach().clone() for name, value in model.named_parameters()
                    if not name.startswith("critic_") and name != "log_std"}
    obs = {key: torch.stack([step["obs"][key] for step in rollout]).flatten(0, 1)
           for key in rollout[0]["obs"]}
    stacked = {key: torch.stack([step[key] for step in rollout])
               for key in ("raw", "log_prob", "value", "next_value", "reward", "terminated", "truncated")}
    advantage, returns = compute_gae(stacked["reward"], stacked["value"], stacked["next_value"],
                                     stacked["terminated"], stacked["truncated"], args.gamma, args.gae_lambda)
    advantage, returns = advantage.flatten(0, 1), returns.flatten(0, 1)
    mask = obs["active"]
    avg = masked_mean(advantage, mask)
    std = masked_mean((advantage - avg).square(), mask).sqrt().clamp_min(1e-8)
    advantage = (advantage - avg) / std
    old_log, old_value, raw = [stacked[key].flatten(0, 1) for key in ("log_prob", "value", "raw")]
    stats = defaultdict(list)
    for _ in range(args.ppo_epochs):
        permutation = torch.randperm(len(advantage), device=advantage.device)
        stop = False
        for idx in permutation.split(args.minibatch_size):
            batch = {key: value[idx] for key, value in obs.items()}
            log_prob, entropy, value = model.evaluate(batch, raw[idx])
            log_ratio = log_prob - old_log[idx]
            ratio = log_ratio.exp()
            unclipped = -advantage[idx] * ratio
            clipped = -advantage[idx] * ratio.clamp(1 - args.clip, 1 + args.clip)
            policy_loss = masked_mean(torch.maximum(unclipped, clipped), mask[idx])
            clipped_value = old_value[idx] + (value - old_value[idx]).clamp(-args.value_clip, args.value_clip)
            value_loss = .5 * masked_mean(torch.maximum((value - returns[idx]).square(),
                                                        (clipped_value - returns[idx]).square()), mask[idx])
            entropy_mean = masked_mean(entropy, mask[idx])
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_mean
            anchor_loss = torch.zeros((), device=loss.device)
            anchor_coef = getattr(args, "bc_anchor_coef", 0.)
            if anchor is not None and anchor_coef > 0:
                sampled = torch.randint(len(anchor["target"]), (min(args.bc_anchor_batch_size, len(anchor["target"])),), device=loss.device)
                anchor_obs = {key: value[sampled] for key, value in anchor["obs"].items()}
                anchor_error = (torch.tanh(model.mean(anchor_obs)) - anchor["target"][sampled]).square().mean(-1)
                anchor_loss = masked_mean(anchor_error, anchor["weight"][sampled])
                loss = loss + anchor_coef * anchor_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss; inspect physics and reward magnitudes")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                kl = masked_mean((ratio - 1) - log_ratio, mask[idx])
            for key, metric in {"policy_loss": policy_loss, "value_loss": value_loss,
                                "entropy": entropy_mean, "approx_kl": kl, "grad_norm": grad_norm,
                                "bc_anchor_loss": anchor_loss,
                                "clip_fraction": masked_mean((abs(ratio - 1) > args.clip).float(), mask[idx])}.items():
                stats[key].append(float(metric.detach()))
            if kl > args.target_kl:
                stop = True
                break
        if stop:
            break
    result = {key: float(np.mean(values)) for key, values in stats.items()}
    result["bc_anchor_coef"] = getattr(args, "bc_anchor_coef", 0.)
    with torch.no_grad():
        result["actor_update_l2"] = float(sum((value - actor_before[name]).square().sum()
                                               for name, value in model.named_parameters()
                                               if name in actor_before).sqrt())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/swarm/policy")
    parser.add_argument("--env-factory", default="arena_mujoco.swarm_env:SwarmVectorEnv")
    parser.add_argument("--backend", choices=("warp", "native", "cpu"), default="warp")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--robots", type=int, default=40)
    parser.add_argument("--objects", type=int, default=4)
    parser.add_argument("--episode-seconds", type=float, default=40.)
    parser.add_argument("--native-workers", type=int, default=1, help="Persistent native MuJoCo world workers; 1 keeps serial integration")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--demo-steps", type=int, default=1800)
    parser.add_argument("--stage-demo-steps", type=int, default=1800)
    parser.add_argument("--demo-noise", type=float, default=.025)
    parser.add_argument("--bc-updates", type=int, default=500)
    parser.add_argument("--stage-bc-updates", type=int, default=150)
    parser.add_argument("--bc-batch-size", type=int, default=8192)
    parser.add_argument("--balance-demo-roles", action="store_true", help="Balance carrier phases, then give carriers and formation equal total demonstration weight")
    parser.add_argument("--dagger-rounds", type=int, default=0, help="Physical dataset aggregation and fitting rounds before PPO")
    parser.add_argument("--dagger-steps", type=int, default=1800, help="Control steps per world in each DAgger collection round")
    parser.add_argument("--dagger-updates", type=int, default=1000, help="Supervised updates on the aggregated dataset per DAgger round")
    parser.add_argument("--dagger-teacher-prob", type=float, default=.5, help="Initial per-world teacher probability, linearly reduced to zero in the final round; one round uses zero")
    parser.add_argument("--initial-action-std", type=float, help="Initial raw-action standard deviation for a fresh model; resumed models retain their saved value")
    parser.add_argument("--bc-anchor-coef", type=float, default=0., help="Optional physical demonstration retention loss during PPO; try 0.1")
    parser.add_argument("--bc-anchor-batch-size", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256, help="Environment-time samples, each containing all robots")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--clip", type=float, default=.15)
    parser.add_argument("--value-clip", type=float, default=.2)
    parser.add_argument("--value-coef", type=float, default=.5)
    parser.add_argument("--entropy-coef", type=float, default=.003)
    parser.add_argument("--target-kl", type=float, default=.025)
    parser.add_argument("--max-grad-norm", type=float, default=.5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--disable-reflection", action="store_true", help="Train without left/right reflection averaging for an ablation")
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--stage-eval-episodes", type=int, default=8, help="Evaluation episodes for curriculum stages 0–2")
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-max-steps", type=int, default=0, help="Explicit evaluation control-step cap; 0 derives enough steps for every episode")
    parser.add_argument("--skip-initial-eval", action="store_true", help="Skip the initial warmstart or resume baseline evaluation")
    parser.add_argument("--defer-evaluation", action="store_true",
                        help="Run no policy evaluations and write training.pending.json for a separate CPU audit")
    parser.add_argument("--stage-success", type=float, default=.7)
    parser.add_argument("--min-stage-updates", type=int, default=20)
    parser.add_argument("--start-stage", type=int, choices=range(4), default=0)
    parser.add_argument("--final-success", type=float, default=.8)
    parser.add_argument("--time-budget-seconds", type=float, default=0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.num_envs, args.updates, args.horizon, args.eval_episodes, args.stage_eval_episodes, args.eval_interval,
           args.minibatch_size, args.ppo_epochs) < 1:
        parser.error("Environment, rollout, update and evaluation sizes must be positive")
    if not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1 or args.demo_steps < 0 or args.bc_updates < 0:
        parser.error("Invalid discount, GAE, or demonstration settings")
    if args.bc_anchor_coef < 0 or args.bc_anchor_batch_size < 1:
        parser.error("BC anchor coefficient must be nonnegative and its batch size positive")
    if args.bc_batch_size < 1 or args.dagger_rounds < 0 or args.dagger_steps < 1 or args.dagger_updates < 1:
        parser.error("BC batch size and DAgger steps/updates must be positive; rounds must be nonnegative")
    if not 0 <= args.dagger_teacher_prob <= 1:
        parser.error("DAgger teacher probability must be between zero and one")
    if args.initial_action_std is not None and not np.exp(-5) <= args.initial_action_std <= np.exp(.5):
        parser.error("Initial action std must be within the policy's supported range exp(-5) to exp(0.5)")
    if args.eval_max_steps < 0:
        parser.error("Evaluation control-step cap must be nonnegative")
    if args.native_workers < 1:
        parser.error("Native worker count must be positive")
    if args.bc_anchor_coef > 0 and (args.demo_steps < 1 or args.stage_demo_steps < 1):
        parser.error("BC anchoring requires physical demonstrations at every curriculum stage")
    device = torch.device("cpu" if args.cpu_smoke else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; --cpu-smoke is for implementation checks only")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU smoke check"
    if device.type == "cuda" and not args.allow_other_gpu and not any(name in gpu for name in ("A100", "H100")):
        raise RuntimeError(f"A100 or H100 required, allocated {gpu}")
    if args.cpu_smoke or args.backend == "cpu":
        args.backend = "native"
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.defer_evaluation and (out / "training.json").exists():
        raise RuntimeError("Deferred evaluation needs an output directory without training.json")
    deferred_sources = training_source_provenance() if args.defer_evaluation else None
    start = time.monotonic()
    env = make_env(args, device, args.seed)
    validation_env = (None if args.defer_evaluation else
                      make_env(args, device, args.seed + 100000,
                               min(args.num_envs, args.eval_episodes)))
    stage = args.start_stage
    env.set_curriculum(stage_spec(stage, args))
    obs = tensor_obs(env.reset(), device)
    config = PolicyConfig(local_dim=obs["local"].shape[-1], neighbor_dim=obs["neighbors"].shape[-1],
                          global_dim=obs["global"].shape[-1], hidden=args.hidden,
                          action_dim=getattr(env, "action_dim", 3),
                          mirror=obs["local"].shape[-1] in (32, 40, 42) and not args.disable_reflection)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_config = PolicyConfig(**checkpoint["config"])
        expected = asdict(config)
        expected["mirror"] = saved_config.mirror
        if asdict(saved_config) != expected:
            raise ValueError("Checkpoint and environment observation dimensions or architecture differ")
        if args.disable_reflection and saved_config.mirror:
            raise ValueError("A reflected checkpoint must resume with reflection; use a fresh ablation run")
        config = saved_config
    model = make_actor_critic(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    initial_update = 0
    environment_steps = 0
    agent_steps = 0
    warmstart_report = None
    demo_data = None
    imitation_exports = {}
    warmstart_baseline_kind = (checkpoint.get("warmstart_baseline_kind", "prior_run_pre_ppo")
                              if checkpoint else "behavior_cloning_pre_ppo")
    full_stage_passes = 0
    cumulative_actor_update = 0.
    if args.resume:
        model.load_state_dict(checkpoint["model"])
        action_std_report = configure_initial_action_std(model, args.initial_action_std, resumed=True)
        progress(args, {"phase": "initial_action_std", **action_std_report})
        optimizer.load_state_dict(checkpoint["optimizer"])
        initial_update = checkpoint["update"]
        environment_steps = checkpoint["environment_steps"]
        agent_steps = checkpoint.get("agent_steps", 0)
        cumulative_actor_update = checkpoint.get("cumulative_actor_update_l2", 0.)
        full_stage_passes = checkpoint.get("consecutive_full_stage_passes", 0)
        stage = checkpoint["stage"]
        env.set_curriculum(stage_spec(stage, args))
        obs = tensor_obs(env.reset(), device)
        if args.updates <= initial_update:
            parser.error("--updates must exceed the resumed checkpoint update")
        if args.bc_anchor_coef > 0 or args.dagger_rounds > 0:
            demo_data = warmstart(model, optimizer, env, args, device, args.stage_demo_steps, 0)
            obs = tensor_obs(env.reset(), device)
    else:
        action_std_report = configure_initial_action_std(model, args.initial_action_std)
        progress(args, {"phase": "initial_action_std", **action_std_report})
        demo_data = warmstart(model, optimizer, env, args, device)
        if demo_data:
            warmstart_report = demo_data["report"]
        initial_name = "bc_initial.npz" if args.dagger_rounds else "warmstart.npz"
        export_policy(model, out / initial_name)
        imitation_exports[initial_name] = "initial_behavior_cloning"
        obs = tensor_obs(env.reset(), device)
    demo_data, dagger_reports = dagger(model, optimizer, env, args, device, demo_data)
    if dagger_reports:
        if args.resume and (out / "warmstart.npz").exists():
            stem = f"warmstart_before_dagger_resume_{initial_update}"
            previous = out / f"{stem}.npz"
            suffix = 1
            while previous.exists():
                previous = out / f"{stem}_{suffix}.npz"
                suffix += 1
            previous.write_bytes((out / "warmstart.npz").read_bytes())
            imitation_exports[previous.name] = warmstart_baseline_kind
        export_policy(model, out / "dagger.npz")
        export_policy(model, out / "warmstart.npz")
        warmstart_baseline_kind = "post_dagger_pre_ppo"
        imitation_exports["dagger.npz"] = warmstart_baseline_kind
        obs = tensor_obs(env.reset(), device)
        full_stage_passes = 0
    if (out / "warmstart.npz").exists():
        imitation_exports["warmstart.npz"] = warmstart_baseline_kind
    anchor = demo_data if args.bc_anchor_coef > 0 else None
    del demo_data
    pre_ppo_action_std = model.log_std.detach().clamp(-5, .5).exp().cpu().tolist()
    demo_weighting = "equal_roles_carrier_phases" if args.balance_demo_roles else "legacy_phase_carrier"
    progress(args, {"phase": "imitation_complete", "dagger_rounds": len(dagger_reports),
                    "demo_weighting": demo_weighting, "pre_ppo_action_std": pre_ppo_action_std})
    evaluations = []
    if args.skip_initial_eval or args.defer_evaluation:
        reason = "--defer-evaluation" if args.defer_evaluation else "--skip-initial-eval"
        progress(args, {"phase": "initial_evaluation_skipped", "reason": reason})
    else:
        baseline = evaluate(model, validation_env, stage, args, device)
        baseline["policy"] = "dagger" if dagger_reports else "warmstart" if not args.resume else "resumed"
        print(json.dumps({"phase": "baseline", **baseline}), flush=True)
        evaluations.append(baseline)
    stage_warmstarts = []
    best = (-1, -1.)
    updates_at_stage = checkpoint.get("updates_at_stage", 0) if checkpoint else 0
    history = []
    stopping_reason = "update_budget_exhausted"
    for update in range(initial_update + 1, args.updates + 1):
        tick = time.monotonic()
        rollout = []
        reward_sums = defaultdict(float)
        active_agent_steps = torch.zeros((), device=device)
        episode_count, episode_success = 0, 0.
        model.eval()
        for _ in range(args.horizon):
            with torch.no_grad():
                action, raw, log_prob, value = model.act(obs)
                saved_obs = {key: tensor.clone() for key, tensor in obs.items()}
                active_agent_steps += saved_obs["active"].sum()
                next_obs, reward, terminated, truncated, info = env.step(action)
                terminal_obs = tensor_obs(next_obs, device)
                terminated = torch.as_tensor(terminated, device=device).bool()
                truncated = torch.as_tensor(truncated, device=device).bool()
                next_value = model.value(terminal_obs)
                rollout.append({"obs": saved_obs, "raw": raw.clone(), "log_prob": log_prob.clone(),
                                "value": value.clone(), "next_value": next_value.clone(),
                                "reward": torch.as_tensor(reward, device=device).clone(),
                                "terminated": terminated.clone(), "truncated": truncated.clone()})
                done = terminated | truncated
                if done.any():
                    episode_count += int(done.sum())
                    episode_success += float(torch.as_tensor(info["success"], device=device)[done].sum())
                for key, component in info.get("reward_components", {}).items():
                    component = torch.as_tensor(component, device=device)
                    reward_sums[key] += masked_mean(component, saved_obs["active"]) if component.ndim == 2 else component.mean()
                obs = tensor_obs(env.reset_done(done), device) if done.any() else terminal_obs
        model.train()
        stats = ppo_update(model, optimizer, rollout, args, anchor)
        cumulative_actor_update += stats["actor_update_l2"]
        environment_steps += args.horizon * args.num_envs
        agent_steps += int(active_agent_steps)
        updates_at_stage += 1
        record = {"phase": "ppo", "update": update, "stage": stage, "environment_steps": environment_steps,
                  "agent_steps": agent_steps, "seconds": time.monotonic() - start,
                  "update_seconds": time.monotonic() - tick, "episodes": episode_count,
                  "rollout_success_rate": episode_success / max(episode_count, 1),
                  "reward_components": {key: float(val / args.horizon) for key, val in reward_sums.items()}, **stats}
        model.eval()
        advance = False
        early_stop = False
        if not args.defer_evaluation and (update % args.eval_interval == 0 or update == args.updates):
            result = evaluate(model, validation_env, stage, args, device)
            result["update"] = update
            evaluations.append(result)
            record["evaluation"] = result
            score = (stage, result["success_rate"])
            if result["complete"] and score > best:
                best = score
                export_policy(model, out / "best.npz")
            advance = (stage < 3 and result["complete"] and result["success_rate"] >= args.stage_success
                       and updates_at_stage >= args.min_stage_updates)
            full_pass = full_stage_validation_pass(result, stage, args.final_success)
            full_stage_passes = full_stage_passes + 1 if full_pass else 0
            early_stop = full_stage_passes >= 2
            record["consecutive_full_stage_passes"] = full_stage_passes
            if early_stop:
                record["early_stop_reason"] = "two_full_stage_validation_passes"
        history.append(record)
        with (out / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        save_json(out / "progress.json", record)
        print(json.dumps(record, allow_nan=False), flush=True)
        export_policy(model, out / "weights.npz")
        torch.save({"config": asdict(config), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "update": update, "stage": stage, "environment_steps": environment_steps,
                    "agent_steps": agent_steps, "cumulative_actor_update_l2": cumulative_actor_update,
                    "warmstart_baseline_kind": warmstart_baseline_kind,
                    "updates_at_stage": updates_at_stage, "consecutive_full_stage_passes": full_stage_passes,
                    "args": vars(args)}, out / "checkpoint.pt.tmp")
        (out / "checkpoint.pt.tmp").replace(out / "checkpoint.pt")
        if early_stop:
            stopping_reason = "two_full_stage_validation_passes"
            break
        if args.time_budget_seconds and time.monotonic() - start >= args.time_budget_seconds:
            stopping_reason = "wall_time_budget"
            break
        if advance:
            stage += 1
            updates_at_stage = 0
            env.set_curriculum(stage_spec(stage, args))
            stage_warmstart = warmstart(model, optimizer, env, args, device,
                                        args.stage_demo_steps, args.stage_bc_updates)
            stage_warmstarts.append({"stage": stage, "warmstart": stage_warmstart["report"] if stage_warmstart else None})
            anchor = stage_warmstart if args.bc_anchor_coef > 0 else None
            del stage_warmstart
            obs = tensor_obs(env.reset(), device)
    for instance in (env, validation_env):
        if instance is not None and hasattr(instance, "close"):
            instance.close()
    weight_digest = file_sha256(out / "weights.npz")
    report = {"method": "shared neighbor-attention MAPPO with physical demonstration warmstart",
              "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda, "seed": args.seed,
              "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "policy_device": str(next(model.parameters()).device), "physics_device": str(env.device),
              "native_workers": env.native_workers,
              "cuda_optimization": next(model.parameters()).device.type == "cuda",
              "backend": args.backend, "config": asdict(config), "parameters": sum(p.numel() for p in model.parameters()),
              "warmstart": warmstart_report, "ppo_updates": update, "ppo_updates_this_process": len(history), "environment_steps": environment_steps,
              "dagger": dagger_reports, "initial_action_std": action_std_report,
              "pre_ppo_action_std": pre_ppo_action_std, "demo_weighting": demo_weighting,
              "imitation_exports": imitation_exports, "warmstart_baseline_kind": warmstart_baseline_kind,
              "stage_warmstarts": stage_warmstarts,
              "bc_anchor_coef": args.bc_anchor_coef, "stopping_reason": stopping_reason,
              "agent_steps": agent_steps,
              "cumulative_ppo_actor_update_l2": cumulative_actor_update,
              "training_seconds": time.monotonic() - start, "final_stage": stage, "evaluations": evaluations,
              "weights_sha256": weight_digest,
              "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    if args.defer_evaluation:
        if training_source_provenance() != deferred_sources:
            raise RuntimeError("Training sources changed before the deferred report was sealed")
        artifact_hashes = {name: file_sha256(out / name)
                           for name in ("weights.npz", "checkpoint.pt", "warmstart.npz")}
        report.update({"heldout": None, "zero_baseline": None, "warmstart_baseline": None,
                       "acceptance_passed": False, "evaluation_status": "pending",
                       "source_commit": deferred_sources["commit"],
                       "source_hashes": deferred_sources["files"],
                       "artifact_hashes": artifact_hashes})
        save_json(out / "training.pending.json", report)
        save_json(out / "progress.json", {"phase": "evaluation_deferred", "acceptance_passed": False,
                                          "training_seconds": report["training_seconds"]})
        print("TRAINING_PENDING " + json.dumps(report, allow_nan=False), flush=True)
        return

    # Final audit uses a third, unseen seed range and the full deployment task.
    final_eval, zero_eval, warmstart_eval = run_final_evaluations(model, config, args, device)
    passed = acceptance_passed(args, stage, cumulative_actor_update, final_eval, zero_eval)
    report.update({"training_seconds": time.monotonic() - start,
                   "heldout": final_eval, "zero_baseline": zero_eval,
                   "warmstart_baseline": warmstart_eval, "acceptance_passed": passed,
                   "evaluation_status": "complete"})
    save_json(out / "training.json", report)
    save_json(out / "progress.json", {"phase": "complete", "acceptance_passed": passed,
                                     "heldout": final_eval, "training_seconds": report["training_seconds"]})
    print("TRAINING_REPORT " + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
