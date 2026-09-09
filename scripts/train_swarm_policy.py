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


def progress(args, record):
    save_json(Path(args.output) / "progress.json", record)
    print(json.dumps(record, allow_nan=False), flush=True)


def make_env(args, device, seed, num_envs=None):
    module, factory = args.env_factory.split(":")
    return getattr(importlib.import_module(module), factory)(
        num_envs=num_envs or args.num_envs, num_robots=args.robots,
        num_objects=args.objects, device=str(device), backend=args.backend, seed=seed,
        episode_seconds=args.episode_seconds,
    )


def stage_spec(stage, args):
    spec = dict(STAGES[stage])
    spec["num_active_robots"] = min(args.robots, spec["num_active_robots"])
    spec["num_active_objects"] = min(args.objects, spec["num_active_objects"])
    return spec


@torch.no_grad()
def evaluate(model, env, stage, args, device, policy="learned"):
    env.set_curriculum(stage_spec(stage, args))
    obs = tensor_obs(env.reset(), device)
    completed, successes, deliveries, returns = [], [], [], []
    episode_return = torch.zeros(obs["active"].shape[0], device=device)
    # A bounded evaluation must not hang if an environment fails to terminate.
    limit = args.eval_max_steps
    for step in range(limit):
        if step % 128 == 0:
            progress(args, {"phase": "evaluation", "policy": policy, "stage": stage,
                            "control_step": step, "episodes": len(completed), "required_episodes": args.eval_episodes})
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
        if done.any():
            success = torch.as_tensor(info["success"], device=device)
            delivered = torch.as_tensor(info["delivered"], device=device)
            for idx in torch.where(done)[0].tolist():
                if len(completed) >= args.eval_episodes:
                    break
                completed.append(idx)
                successes.append(float(success[idx]))
                deliveries.append(float(delivered[idx]))
                returns.append(float(episode_return[idx]))
            if len(completed) >= args.eval_episodes:
                break
            episode_return[done] = 0
            obs = tensor_obs(env.reset_done(done), device)
    if not completed:
        raise RuntimeError(f"No evaluation episode terminated within {limit} physics control steps")
    p = float(np.mean(successes))
    n = len(completed)
    # Wilson lower bound prevents a single lucky episode from passing the gate.
    z = 1.96
    lower = (p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
    return {"policy": policy, "stage": stage, "episodes": n, "success_rate": p,
            "success_wilson_lower_95": float(lower), "mean_delivered": float(np.mean(deliveries)),
            "mean_return": float(np.mean(returns)), "complete": n == args.eval_episodes}


def warmstart(model, optimizer, env, args, device, demo_steps=None, bc_updates=None):
    """Collect noisy teacher trajectories through physics, then fit the actor."""
    demo_steps = args.demo_steps if demo_steps is None else demo_steps
    bc_updates = args.bc_updates if bc_updates is None else bc_updates
    obs = tensor_obs(env.reset(), device)
    samples, targets = [], []
    for step in range(demo_steps):
        if step % 128 == 0:
            progress(args, {"phase": "demonstrations", "control_step": step, "total_control_steps": demo_steps})
        target = torch.as_tensor(env.teacher_action(), device=device, dtype=torch.float32)
        active = obs["active"].bool()
        # Store participating agents only. The critic's global state is unused
        # by cloning, and parked robots should not dominate contact examples.
        samples.append({key: value[active].clone() for key, value in obs.items() if key != "global"})
        targets.append(target[active].clone())
        noisy = (target + torch.randn_like(target) * args.demo_noise).clamp(-1, 1)
        next_obs, _, terminated, truncated, _ = env.step(noisy)
        done = torch.as_tensor(terminated, device=device).bool() | torch.as_tensor(truncated, device=device).bool()
        obs = tensor_obs(env.reset_done(done) if done.any() else next_obs, device)
    if not samples:
        return None
    data = {key: torch.cat([sample[key] for sample in samples]) for key in samples[0]}
    target = torch.cat(targets)
    weight = data.get("learning_weight", torch.ones(len(target), device=device)).clone()
    if "phase" in data:
        phases = data["phase"].long()
        counts = torch.bincount(phases, minlength=int(phases.max()) + 1).clamp_min(1)
        weight *= len(phases) / (len(counts) * counts[phases])
    weight /= weight.mean().clamp_min(1e-8)
    last_loss = 0.0
    for update in range(bc_updates):
        idx = torch.randint(len(target), (min(args.bc_batch_size, len(target)),), device=device)
        batch = {key: value[idx] for key, value in data.items()}
        error = (torch.tanh(model.mean(batch)) - target[idx]).square().mean(-1)
        loss = masked_mean(error, weight[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        last_loss = float(loss.detach())
        if update % 100 == 0:
            progress(args, {"phase": "behavior_cloning", "update": update + 1,
                            "total_updates": bc_updates, "weighted_mse": last_loss})
    return {"physics_control_steps": demo_steps * args.num_envs, "agent_examples": len(target),
            "updates": bc_updates, "final_mse": last_loss,
            "phase_balanced": "phase" in data, "carrier_weighted": "learning_weight" in data}


def ppo_update(model, optimizer, rollout, args):
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
                                "clip_fraction": masked_mean((abs(ratio - 1) > args.clip).float(), mask[idx])}.items():
                stats[key].append(float(metric.detach()))
            if kl > args.target_kl:
                stop = True
                break
        if stop:
            break
    result = {key: float(np.mean(values)) for key, values in stats.items()}
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
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--demo-steps", type=int, default=1800)
    parser.add_argument("--stage-demo-steps", type=int, default=1800)
    parser.add_argument("--demo-noise", type=float, default=.025)
    parser.add_argument("--bc-updates", type=int, default=500)
    parser.add_argument("--stage-bc-updates", type=int, default=150)
    parser.add_argument("--bc-batch-size", type=int, default=8192)
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
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-max-steps", type=int, default=10000)
    parser.add_argument("--stage-success", type=float, default=.7)
    parser.add_argument("--min-stage-updates", type=int, default=20)
    parser.add_argument("--start-stage", type=int, choices=range(4), default=0)
    parser.add_argument("--final-success", type=float, default=.8)
    parser.add_argument("--time-budget-seconds", type=float, default=0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.num_envs, args.updates, args.horizon, args.eval_episodes, args.eval_interval,
           args.minibatch_size, args.ppo_epochs) < 1:
        parser.error("Environment, rollout, update and evaluation sizes must be positive")
    if not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1 or args.demo_steps < 0 or args.bc_updates < 0:
        parser.error("Invalid discount, GAE, or demonstration settings")
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
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    env = make_env(args, device, args.seed)
    validation_env = make_env(args, device, args.seed + 100000, min(args.num_envs, args.eval_episodes))
    stage = args.start_stage
    env.set_curriculum(stage_spec(stage, args))
    obs = tensor_obs(env.reset(), device)
    config = PolicyConfig(local_dim=obs["local"].shape[-1], neighbor_dim=obs["neighbors"].shape[-1],
                          global_dim=obs["global"].shape[-1], hidden=args.hidden,
                          mirror=obs["local"].shape[-1] == 32 and not args.disable_reflection)
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
    cumulative_actor_update = 0.
    if args.resume:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        initial_update = checkpoint["update"]
        environment_steps = checkpoint["environment_steps"]
        agent_steps = checkpoint.get("agent_steps", 0)
        cumulative_actor_update = checkpoint.get("cumulative_actor_update_l2", 0.)
        stage = checkpoint["stage"]
        env.set_curriculum(stage_spec(stage, args))
        obs = tensor_obs(env.reset(), device)
        if args.updates <= initial_update:
            parser.error("--updates must exceed the resumed checkpoint update")
    else:
        warmstart_result = warmstart(model, optimizer, env, args, device)
        if warmstart_result:
            warmstart_report = warmstart_result
            del warmstart_result
        export_policy(model, out / "warmstart.npz")
        obs = tensor_obs(env.reset(), device)
    baseline = evaluate(model, validation_env, stage, args, device)
    baseline["policy"] = "warmstart" if not args.resume else "resumed"
    print(json.dumps({"phase": "baseline", **baseline}), flush=True)
    evaluations = [baseline]
    stage_warmstarts = []
    best = (-1, -1.)
    updates_at_stage = 0
    history = []
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
        stats = ppo_update(model, optimizer, rollout, args)
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
        if update % args.eval_interval == 0 or update == args.updates:
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
        history.append(record)
        with (out / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        save_json(out / "progress.json", record)
        print(json.dumps(record, allow_nan=False), flush=True)
        export_policy(model, out / "weights.npz")
        torch.save({"config": asdict(config), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "update": update, "stage": stage, "environment_steps": environment_steps,
                    "agent_steps": agent_steps, "cumulative_actor_update_l2": cumulative_actor_update,
                    "args": vars(args)}, out / "checkpoint.pt.tmp")
        (out / "checkpoint.pt.tmp").replace(out / "checkpoint.pt")
        if args.time_budget_seconds and time.monotonic() - start >= args.time_budget_seconds:
            break
        if advance:
            stage += 1
            updates_at_stage = 0
            env.set_curriculum(stage_spec(stage, args))
            stage_warmstart = warmstart(model, optimizer, env, args, device,
                                        args.stage_demo_steps, args.stage_bc_updates)
            stage_warmstarts.append({"stage": stage, "warmstart": stage_warmstart})
            obs = tensor_obs(env.reset(), device)
    # Final audit uses a third, unseen seed range and the full deployment task.
    for instance in (env, validation_env):
        if hasattr(instance, "close"):
            instance.close()
    audit_env = make_env(args, device, args.seed + 200000, min(args.num_envs, args.eval_episodes))
    final_eval = evaluate(model, audit_env, 3, args, device)
    if hasattr(audit_env, "close"):
        audit_env.close()
    baseline_env = make_env(args, device, args.seed + 200000, min(args.num_envs, args.eval_episodes))
    zero_eval = evaluate(model, baseline_env, 3, args, device, policy="zero")
    if hasattr(baseline_env, "close"):
        baseline_env.close()
    warmstart_eval = None
    if (out / "warmstart.npz").exists():
        warmstart_model = make_actor_critic(config).to(device)
        with np.load(out / "warmstart.npz", allow_pickle=False) as archive:
            warmstart_model.load_state_dict({key: torch.as_tensor(archive[key], device=device)
                                            for key in archive.files if key != "config"}, strict=False)
        warmstart_env = make_env(args, device, args.seed + 200000, min(args.num_envs, args.eval_episodes))
        warmstart_eval = evaluate(warmstart_model, warmstart_env, 3, args, device)
        warmstart_eval["policy"] = "warmstart"
        if hasattr(warmstart_env, "close"):
            warmstart_env.close()
    digest = hashlib.sha256((out / "weights.npz").read_bytes()).hexdigest()
    passed = (not args.cpu_smoke and args.robots == 40 and args.objects >= 2 and stage == 3
              and cumulative_actor_update > 1e-6
              and final_eval["complete"] and final_eval["episodes"] >= 32
              and final_eval["success_rate"] >= args.final_success
              and final_eval["success_wilson_lower_95"] >= .6
              and final_eval["success_rate"] > zero_eval["success_rate"] + .2)
    report = {"method": "shared neighbor-attention MAPPO with physical demonstration warmstart",
              "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda, "seed": args.seed,
              "backend": args.backend, "config": asdict(config), "parameters": sum(p.numel() for p in model.parameters()),
              "warmstart": warmstart_report, "ppo_updates": len(history), "environment_steps": environment_steps,
              "stage_warmstarts": stage_warmstarts,
              "agent_steps": agent_steps,
              "cumulative_ppo_actor_update_l2": cumulative_actor_update,
              "training_seconds": time.monotonic() - start, "final_stage": stage, "evaluations": evaluations,
              "heldout": final_eval, "zero_baseline": zero_eval, "warmstart_baseline": warmstart_eval,
              "acceptance_passed": passed,
              "weights_sha256": digest, "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    save_json(out / "training.json", report)
    save_json(out / "progress.json", {"phase": "complete", "acceptance_passed": passed,
                                     "heldout": final_eval, "training_seconds": report["training_seconds"]})
    print("TRAINING_REPORT " + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
