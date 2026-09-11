"""Native wheel-contact demonstrations and a compact goal-drive policy."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Iterable

import mujoco
import numpy as np
import torch
from torch import nn

from .best_fleet import FleetSimulation, wrap
from .competition_events import initialize_competition_events


DRIVE_SCHEMA = 1


def _process_pool(workers: int) -> ProcessPoolExecutor:
    """Use spawn so CUDA initialization in the trainer cannot poison workers."""
    return ProcessPoolExecutor(max_workers=workers,
                               mp_context=multiprocessing.get_context("spawn"))


@dataclass(frozen=True)
class DriveConfig:
    observation_dim: int = 8
    hidden_dim: int = 128
    max_forward_m_s: float = 0.35
    max_yaw_rad_s: float = 2.5
    goal_scale_m: float = 0.10
    wheel_speed_scale_rad_s: float = 20.0
    body_speed_scale_m_s: float = 0.4
    body_yaw_scale_rad_s: float = 3.0
    goal_tolerance_m: float = 0.002
    heading_tolerance_rad: float = 0.015
    episode_seconds: float = 12.0


class BestDriveMLP(nn.Module):
    """Map goal and measured drive state to body velocity commands."""

    def __init__(self, config: DriveConfig = DriveConfig()):
        super().__init__()
        h = config.hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(config.observation_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, h // 2), nn.Tanh(),
            nn.Linear(h // 2, 2), nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


def drive_observation(goal_xy: Iterable[float], goal_heading: float,
                      pose_xy: Iterable[float], heading: float,
                      wheel_speeds: Iterable[float],
                      body_velocity: Iterable[float] = (0.0, 0.0),
                      config: DriveConfig = DriveConfig()) -> np.ndarray:
    """Build the fixed normalized observation used by training and inference."""
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    pose_xy = np.asarray(pose_xy, dtype=np.float64)
    wheel_speeds = np.asarray(wheel_speeds, dtype=np.float64)
    body_velocity = np.asarray(body_velocity, dtype=np.float64)
    if goal_xy.shape != (2,) or pose_xy.shape != (2,) or wheel_speeds.shape != (2,) or body_velocity.shape != (2,):
        raise ValueError("goal, pose, wheel speeds, and body velocity must each contain two values")
    if not np.isfinite(np.r_[goal_xy, goal_heading, pose_xy, heading,
                             wheel_speeds, body_velocity]).all():
        raise ValueError("Drive observations must be finite")
    difference = goal_xy - pose_xy
    forward_axis = np.array((math.cos(heading), math.sin(heading)))
    right_axis = np.array((math.sin(heading), -math.cos(heading)))
    final_error = wrap(float(goal_heading) - float(heading))
    return np.array((
        np.clip(np.dot(difference, forward_axis) / config.goal_scale_m, -1, 1),
        np.clip(np.dot(difference, right_axis) / config.goal_scale_m, -1, 1),
        math.sin(final_error), math.cos(final_error),
        np.clip(wheel_speeds[0] / config.wheel_speed_scale_rad_s, -1.5, 1.5),
        np.clip(wheel_speeds[1] / config.wheel_speed_scale_rad_s, -1.5, 1.5),
        np.clip(body_velocity[0] / config.body_speed_scale_m_s, -1.5, 1.5),
        np.clip(body_velocity[1] / config.body_yaw_scale_rad_s, -1.5, 1.5),
    ), dtype=np.float32)


def teacher_command(goal_xy: Iterable[float], goal_heading: float,
                    pose_xy: Iterable[float], heading: float,
                    config: DriveConfig = DriveConfig(),
                    primitive: str | None = None) -> np.ndarray:
    """One-step line command in the signed final-heading frame."""
    delta = np.asarray(goal_xy, dtype=np.float64) - np.asarray(pose_xy, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    final_error = wrap(float(goal_heading) - float(heading))
    direction = np.array((math.cos(goal_heading), math.sin(goal_heading)))
    lateral_axis = np.array((-direction[1], direction[0]))
    lateral_error = float(delta @ lateral_axis)
    if distance <= config.goal_tolerance_m:
        current_forward = np.array((math.cos(heading), math.sin(heading)))
        anchor_error = float(delta @ current_forward)
        forward = config.max_forward_m_s * math.tanh(anchor_error / 0.055)
        yaw = config.max_yaw_rad_s * math.tanh(final_error / 0.35)
        if (distance <= config.goal_tolerance_m and
                abs(final_error) <= config.heading_tolerance_rad):
            forward = 0.0
            yaw = 0.0
    else:
        base_heading_error = wrap(float(goal_heading) - float(heading))
        if distance > 0.02 and abs(base_heading_error) > 0.05:
            forward = 0.0
            yaw = config.max_yaw_rad_s * math.tanh(base_heading_error / 0.35)
            return np.array((forward, yaw / config.max_yaw_rad_s), dtype=np.float32)
        forward_error = float(delta @ direction)
        sign = 1.0 if forward_error >= 0 else -1.0
        correction = np.clip(
            math.atan2(lateral_error, max(abs(forward_error), 0.025)) * sign,
            -0.35, 0.35)
        heading_error = wrap(float(goal_heading) + correction - float(heading))
        yaw = config.max_yaw_rad_s * math.tanh(heading_error / 0.35)
        forward_request = forward_error * max(0.0, math.cos(heading_error))
        forward = config.max_forward_m_s * math.tanh(forward_request / 0.055)
    return np.array((forward / config.max_forward_m_s,
                     yaw / config.max_yaw_rad_s), dtype=np.float32)


class _GeometricDriveTeacher:
    """Stateful Driver.line equivalent, including lateral-error retries."""

    def __init__(self, goal_xy: Iterable[float], goal_heading: float,
                 primitive: str, config: DriveConfig):
        self.goal = np.asarray(goal_xy, dtype=np.float64)
        self.heading = float(goal_heading)
        self.primitive = primitive
        self.config = config
        self.phase = "final_turn" if primitive == "turn" else "initial_turn"
        self.phase_steps = 0
        self.retries = 0
        self.approach_sign = -1.0 if primitive == "reverse" else 1.0
        self.turn_anchor = None

    def _turn(self, pose_xy: np.ndarray, heading: float) -> np.ndarray:
        anchor = pose_xy if self.turn_anchor is None else self.turn_anchor
        delta = anchor - pose_xy
        axis = np.array((math.cos(heading), math.sin(heading)))
        forward = math.tanh(float(delta @ axis) / 0.055)
        yaw = math.tanh(wrap(self.heading - heading) / 0.35)
        return np.array((forward, yaw), dtype=np.float32)

    def command(self, pose_xy: Iterable[float], heading: float) -> np.ndarray:
        pose_xy = np.asarray(pose_xy, dtype=np.float64)
        if self.turn_anchor is None:
            self.turn_anchor = pose_xy.copy()
        self.phase_steps += 1
        final_error = wrap(self.heading - float(heading))
        direction = np.array((math.cos(self.heading), math.sin(self.heading)))
        lateral_axis = np.array((-direction[1], direction[0]))
        delta = self.goal - pose_xy
        forward_error = float(delta @ direction)
        lateral_error = float(delta @ lateral_axis)
        distance = float(np.linalg.norm(delta))

        if self.phase == "initial_turn":
            action = self._turn(pose_xy, heading)
            if abs(final_error) <= 0.012:
                self.phase, self.phase_steps = "line", 0
            return action

        if self.phase == "recovery":
            recovery_goal = self.goal - self.approach_sign * 0.025 * direction
            recovery_delta = recovery_goal - pose_xy
            recovery_forward = float(recovery_delta @ direction)
            recovery_lateral = float(recovery_delta @ lateral_axis)
            if abs(recovery_forward) <= 0.002:
                self.phase, self.phase_steps = "line", 0
            else:
                sign = 1.0 if recovery_forward >= 0 else -1.0
                aim = self.heading + np.clip(
                    math.atan2(recovery_lateral, max(abs(recovery_forward), 0.025)) * sign,
                    -0.35, 0.35)
                angle = wrap(aim - float(heading))
                return np.array((math.tanh(recovery_forward * max(0.0, math.cos(angle)) / 0.055),
                                 math.tanh(angle / 0.35)), dtype=np.float32)

        if self.phase == "final_turn":
            action = self._turn(pose_xy, heading)
            if (abs(final_error) <= self.config.heading_tolerance_rad and
                    distance > self.config.goal_tolerance_m and self.retries < 5):
                self.phase, self.phase_steps = "recovery", 0
                self.retries += 1
            return action

        if (abs(forward_error) <= self.config.goal_tolerance_m and
                abs(lateral_error) <= max(self.config.goal_tolerance_m, 0.0012)):
            self.phase, self.phase_steps = "final_turn", 0
            self.turn_anchor = pose_xy.copy()
            return self._turn(pose_xy, heading)
        if (self.phase_steps * 0.02 > 2.5 and
                abs(forward_error) <= self.config.goal_tolerance_m and
                abs(lateral_error) > max(self.config.goal_tolerance_m, 0.0012) and
                self.retries < 5):
            self.phase, self.phase_steps = "recovery", 0
            self.retries += 1
        return teacher_command(self.goal, self.heading, pose_xy, heading,
                               self.config, self.primitive)


def drive_status(goal_xy: Iterable[float], goal_heading: float,
                 pose_xy: Iterable[float], heading: float,
                 config: DriveConfig = DriveConfig()) -> dict:
    distance = float(np.linalg.norm(np.asarray(goal_xy) - np.asarray(pose_xy)))
    heading_error = abs(wrap(float(goal_heading) - float(heading)))
    return {"distance_m": distance, "heading_error_rad": heading_error,
            "success": distance <= config.goal_tolerance_m and
                       heading_error <= config.heading_tolerance_rad}


def _wheel_speeds(simulation: FleetSimulation, key: str = "lab") -> np.ndarray:
    robot = simulation.robots[key]
    return np.array([simulation.data.qvel[simulation.model.joint(name).dofadr[0]]
                     for name in robot["wheel_joints"]], dtype=np.float64)


def reset_drive_episode(simulation: FleetSimulation, start_xy: Iterable[float],
                        start_heading: float, key: str = "lab") -> None:
    """Reset one coupon boundary. No pose writes occur after this returns."""
    mujoco.mj_resetData(simulation.model, simulation.data)
    robot = simulation.robots[key]
    freejoint = simulation.model.joint(robot["freejoint"])
    address = int(freejoint.qposadr[0])
    simulation.data.qpos[address:address + 3] = (*start_xy, robot["wheel_radius_m"])
    model_yaw = float(start_heading) - math.pi / 2
    simulation.data.qpos[address + 3:address + 7] = (
        math.cos(model_yaw / 2), 0.0, 0.0, math.sin(model_yaw / 2))
    for joint in robot["jaw_joints"]:
        simulation.data.qpos[simulation.model.joint(joint).qposadr[0]] = 0.025
    simulation.targets[key] = np.array((0.0, 0.0, 0.0, 0.025))
    simulation.integrals[key][:] = 0
    simulation.extension_targets[key] = 0.0
    simulation.max_torque = 0.0
    simulation.max_tilt = 0.0
    simulation.data.ctrl[:] = 0
    mujoco.mj_forward(simulation.model, simulation.data)
    simulation.setup = initialize_competition_events(
        simulation.model, simulation.data, simulation.metadata)


def sample_drive_task(rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray, float, str]:
    """Sample precision line and turn primitives in the empty center corridor."""
    for _ in range(100):
        start = rng.uniform((0.16, 0.46), (0.82, 0.72))
        curriculum = rng.random()
        start_heading = rng.uniform(-math.pi, math.pi)
        if curriculum < 0.20:
            return start, start_heading, start.copy(), rng.uniform(-math.pi, math.pi), "turn"
        if curriculum < 0.50:
            distance = rng.uniform(0.004, 0.06)
        elif curriculum < 0.72:
            distance = rng.uniform(0.06, 0.18)
        else:
            distance = rng.uniform(0.18, 0.62)
        direction = rng.uniform(-math.pi, math.pi)
        goal = start + distance * np.array((math.cos(direction), math.sin(direction)))
        if (0.16 <= goal[0] <= 0.82 and 0.46 <= goal[1] <= 0.72):
            reverse = rng.random() < 0.30
            goal_heading = wrap(direction + math.pi) if reverse else direction
            return start, start_heading, goal, goal_heading, "reverse" if reverse else "forward"
    raise RuntimeError("Could not sample a bounded center-corridor drive task")


def drive_domain_nominal(simulation: FleetSimulation, key: str = "lab") -> dict:
    """Capture immutable actuator and contact values for repeated randomization."""
    robot = simulation.robots[key]
    motor_ids = [simulation.model.actuator(name).id for name in robot["motor_names"]]
    wheel_geom_ids = []
    for geom_id in range(simulation.model.ngeom):
        name = mujoco.mj_id2name(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("fleet_lab_") and "wheel_" in name:
            wheel_geom_ids.append(geom_id)
    if len(wheel_geom_ids) != 2:
        raise ValueError(f"Expected two lab wheel geoms, found {len(wheel_geom_ids)}")
    return {"motor_ids": motor_ids, "wheel_geom_ids": wheel_geom_ids,
            "friction": simulation.model.geom_friction[wheel_geom_ids].copy(),
            "motor_strength": float(simulation.motor_strength[key]),
            "no_load_speed": float(robot["motor_no_load_speed_rad_s"])}


def apply_drive_domain(simulation: FleetSimulation, nominal: dict,
                       rng: np.random.Generator, key: str = "lab") -> dict:
    """Apply real actuator and wheel-contact variation once per episode."""
    strength = float(rng.uniform(0.80, 1.00))
    no_load = float(rng.uniform(0.90, 1.10))
    friction = float(rng.uniform(0.75, 1.15))
    simulation.motor_strength[key] = nominal["motor_strength"] * strength
    varied_friction = nominal["friction"].copy()
    varied_friction[:, 0] *= friction
    simulation.model.geom_friction[nominal["wheel_geom_ids"]] = varied_friction
    simulation.wheel_friction_scale = friction
    simulation.robots[key]["motor_no_load_speed_rad_s"] = nominal["no_load_speed"] * no_load
    simulation.metadata["drive_domain"] = {
        "motor_strength": dict(simulation.motor_strength),
        "wheel_friction_scale": friction,
        "motor_no_load_factor": no_load,
    }
    return {"motor_strength_factor": strength, "motor_no_load_factor": no_load,
            "wheel_friction_factor": friction}


def drive_failure(simulation: FleetSimulation, key: str = "lab") -> str | None:
    body = simulation.data.body(simulation.robots[key]["name"])
    position = body.xpos
    if not np.isfinite(position).all() or not np.isfinite(body.xmat).all():
        return "nonfinite_body_state"
    tilt = math.acos(np.clip(body.xmat[8], -1, 1))
    if tilt > 0.45:
        return "tipped"
    if not (0.0 <= position[0] <= 1.143 and 0.0 <= position[1] <= 1.181):
        return "off_board"
    return None


def rollout_episode(simulation: FleetSimulation, seed: int, config: DriveConfig,
                    disturbance_fraction: float = 0.18,
                    policy=None, nominal: dict | None = None) -> dict:
    """Collect or score one episode using only native motor/contact steps."""
    rng = np.random.default_rng(seed)
    start_xy, start_heading, goal_xy, goal_heading, primitive = sample_drive_task(rng)
    reset_drive_episode(simulation, start_xy, start_heading)
    nominal = nominal or drive_domain_nominal(simulation)
    domain = apply_drive_domain(simulation, nominal, rng)
    observations, targets = [], []
    previous_xy, previous_heading = simulation.pose("lab")
    teacher = _GeometricDriveTeacher(goal_xy, goal_heading, primitive, config)
    peak_tilt = 0.0
    peak_actual_torque = 0.0
    failure_reason = None
    last_good_pose = (previous_xy.copy(), previous_heading)
    steps = round(config.episode_seconds / simulation.control_dt)
    for step in range(steps):
        pose_xy, heading = simulation.pose("lab")
        body_forward = np.array((math.cos(heading), math.sin(heading)))
        world_velocity = (pose_xy - previous_xy) / simulation.control_dt if step else np.zeros(2)
        measured_velocity = (float(np.dot(world_velocity, body_forward)),
                             wrap(heading - previous_heading) / simulation.control_dt if step else 0.0)
        wheels = _wheel_speeds(simulation)
        observation = drive_observation(goal_xy, goal_heading, pose_xy, heading,
                                        wheels, measured_velocity, config)
        target = teacher.command(pose_xy, heading)
        observations.append(observation)
        targets.append(target)
        if drive_status(goal_xy, goal_heading, pose_xy, heading, config)["success"]:
            break
        if policy is None:
            action = target.copy()
            if rng.random() < disturbance_fraction:
                action += rng.normal((0.0, 0.0), (0.12, 0.14)).astype(np.float32)
        elif policy == "teacher":
            action = target
        elif policy == "zero":
            action = np.zeros(2, dtype=np.float32)
        else:
            action = np.asarray(policy(observation), dtype=np.float32)
        action = np.clip(action, -1, 1)
        simulation.command("lab", forward=float(action[0] * config.max_forward_m_s),
                           yaw=float(action[1] * config.max_yaw_rad_s), lift=0.0, jaw=0.025)
        previous_xy, previous_heading = pose_xy, heading
        try:
            simulation.step()
        except (FloatingPointError, ValueError) as error:
            failure_reason = f"simulation_error:{type(error).__name__}"
            break
        last_good_pose = simulation.pose("lab")
        motor_ids = nominal["motor_ids"]
        peak_actual_torque = max(peak_actual_torque,
                                 float(np.abs(simulation.data.actuator_force[motor_ids]).max()))
        body = simulation.data.body(simulation.robots["lab"]["name"])
        peak_tilt = max(peak_tilt, math.acos(np.clip(body.xmat[8], -1, 1)))
        failure_reason = drive_failure(simulation)
        if failure_reason is not None:
            break
    pose_xy, heading = last_good_pose if failure_reason is not None else simulation.pose("lab")
    status = drive_status(goal_xy, goal_heading, pose_xy, heading, config)
    if failure_reason is not None:
        status["success"] = False
    status.update({"steps": len(observations), "seconds": len(observations) * simulation.control_dt,
                   "peak_tilt_rad": peak_tilt, "maximum_commanded_torque_nm": simulation.max_torque,
                   "maximum_actual_torque_nm": peak_actual_torque, "domain": domain,
                   "seed": seed, "primitive": primitive, "failure_reason": failure_reason,
                   "start_xy": start_xy.tolist(), "goal_xy": goal_xy.tolist()})
    return {"observations": np.asarray(observations, dtype=np.float32),
            "targets": np.asarray(targets, dtype=np.float32), "status": status}


def rollout_ppo_episode(simulation: FleetSimulation, seed: int, config: DriveConfig,
                        policy, action_std: float, time_penalty_per_second: float,
                        nominal: dict | None = None) -> dict:
    """Execute one stochastic actor rollout and record its actual rewards."""
    rng = np.random.default_rng(seed)
    start_xy, start_heading, goal_xy, goal_heading, primitive = sample_drive_task(rng)
    reset_drive_episode(simulation, start_xy, start_heading)
    nominal = nominal or drive_domain_nominal(simulation)
    domain = apply_drive_domain(simulation, nominal, rng)
    previous_xy, previous_heading = simulation.pose("lab")
    observations, actions, rewards, dones, timestamps = [], [], [], [], []
    components = {"goal_progress": 0.0, "time": 0.0, "success": 0.0,
                  "timeout": 0.0, "failure": 0.0}
    steps = round(config.episode_seconds / simulation.control_dt)
    peak_tilt = 0.0
    peak_actual_torque = 0.0
    failure_reason = None
    last_good_pose = (previous_xy.copy(), previous_heading)
    for step in range(steps):
        pose_xy, heading = simulation.pose("lab")
        body_forward = np.array((math.cos(heading), math.sin(heading)))
        world_velocity = (pose_xy - previous_xy) / simulation.control_dt if step else np.zeros(2)
        measured_velocity = (float(np.dot(world_velocity, body_forward)),
                             wrap(heading - previous_heading) / simulation.control_dt if step else 0.0)
        observation = drive_observation(goal_xy, goal_heading, pose_xy, heading,
                                        _wheel_speeds(simulation), measured_velocity, config)
        mean = np.asarray(policy(observation), dtype=np.float32)
        raw_action = (mean + rng.normal(0, action_std, 2)).astype(np.float32)
        action = np.clip(raw_action, -1, 1)
        before = drive_status(goal_xy, goal_heading, pose_xy, heading, config)
        simulation.command("lab", forward=float(action[0] * config.max_forward_m_s),
                           yaw=float(action[1] * config.max_yaw_rad_s), lift=0.0, jaw=0.025)
        previous_xy, previous_heading = pose_xy, heading
        failure_reason = None
        try:
            simulation.step()
            after_xy, after_heading = simulation.pose("lab")
            after = drive_status(goal_xy, goal_heading, after_xy, after_heading, config)
            failure_reason = drive_failure(simulation)
            if failure_reason is None:
                last_good_pose = (after_xy.copy(), after_heading)
        except (FloatingPointError, ValueError) as error:
            after = before
            failure_reason = f"simulation_error:{type(error).__name__}"
        progress = (20.0 * (before["distance_m"] - after["distance_m"]) +
                    0.35 * (before["heading_error_rad"] - after["heading_error_rad"]))
        time_reward = -float(time_penalty_per_second) * simulation.control_dt
        success_reward = 10.0 if after["success"] else 0.0
        failure_reward = -25.0 if failure_reason is not None else 0.0
        timeout_reward = (-20.0 if step == steps - 1 and
                          not after["success"] and failure_reason is None else 0.0)
        reward = progress + time_reward + success_reward + timeout_reward + failure_reward
        observations.append(observation)
        actions.append(raw_action)
        rewards.append(reward)
        dones.append(after["success"] or failure_reason is not None or step == steps - 1)
        timestamps.append(float(simulation.data.time))
        components["goal_progress"] += progress
        components["time"] += time_reward
        components["success"] += success_reward
        components["timeout"] += timeout_reward
        components["failure"] += failure_reward
        if not failure_reason:
            motor_ids = nominal["motor_ids"]
            peak_actual_torque = max(peak_actual_torque,
                                     float(np.abs(simulation.data.actuator_force[motor_ids]).max()))
            body = simulation.data.body(simulation.robots["lab"]["name"])
            peak_tilt = max(peak_tilt, math.acos(np.clip(body.xmat[8], -1, 1)))
        if dones[-1]:
            break
    final_xy, final_heading = (last_good_pose if failure_reason is not None
                               else simulation.pose("lab"))
    status = drive_status(goal_xy, goal_heading, final_xy, final_heading, config)
    if failure_reason is not None:
        status["success"] = False
    status.update({"steps": len(rewards), "seconds": len(rewards) * simulation.control_dt,
                   "start_time_s": 0.0, "end_time_s": float(simulation.data.time),
                   "reward_components": components, "return": float(sum(rewards)),
                   "peak_tilt_rad": peak_tilt, "maximum_actual_torque_nm": peak_actual_torque,
                   "domain": domain, "seed": seed, "primitive": primitive,
                   "failure_reason": failure_reason})
    return {"observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "timestamps": np.asarray(timestamps, dtype=np.float32),
            "status": status}


def _collect_chunk(payload: tuple[list[int], int, dict]) -> dict:
    seeds, domain_seed, config_values = payload
    config = DriveConfig(**config_values)
    simulation = FleetSimulation(seed=domain_seed, randomize=False, only=["lab"])
    nominal = drive_domain_nominal(simulation)
    episodes = [rollout_episode(simulation, seed, config, nominal=nominal) for seed in seeds]
    return {
        "observations": np.concatenate([episode["observations"] for episode in episodes]),
        "targets": np.concatenate([episode["targets"] for episode in episodes]),
        "status": [episode["status"] for episode in episodes],
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def collect_drive_dataset(root: str | Path, config: DriveConfig,
                          split_episodes: dict[str, int], split_seeds: dict[str, int],
                          workers: int, source_hashes: dict, force: bool = False) -> dict:
    """Freeze actual MuJoCo rollout states and geometric teacher commands."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    expected = {"schema_version": DRIVE_SCHEMA, "config": asdict(config),
                "split_episodes": split_episodes, "split_seeds": split_seeds,
                "source_sha256": source_hashes}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text())
        if all(manifest.get(key) == value for key, value in expected.items()):
            return manifest
        raise ValueError("Drive dataset configuration changed; use --force-generate")
    if len(set(split_seeds.values())) != len(split_seeds):
        raise ValueError("Drive split seeds must be distinct")
    workers = max(1, workers)
    split_records = {}
    overall_started = time.perf_counter()
    for split, episode_count in split_episodes.items():
        started = time.perf_counter()
        seed_rng = np.random.default_rng(split_seeds[split])
        episode_seeds = seed_rng.integers(0, np.iinfo(np.int32).max, episode_count).tolist()
        chunks = [episode_seeds[index::min(workers, episode_count)]
                  for index in range(min(workers, episode_count))]
        payloads = [(chunk, split_seeds[split] + 10_000 + index, asdict(config))
                    for index, chunk in enumerate(chunks)]
        if len(payloads) == 1:
            results = [_collect_chunk(payloads[0])]
        else:
            with _process_pool(len(payloads)) as executor:
                results = list(executor.map(_collect_chunk, payloads))
        observations = np.concatenate([result["observations"] for result in results])
        targets = np.concatenate([result["targets"] for result in results])
        statuses = [status for result in results for status in result["status"]]
        temporary = root / f".{split}.tmp.npz"
        final = root / f"{split}.npz"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, observations=observations, targets=targets)
        os.replace(temporary, final)
        successes = [float(status["success"]) for status in statuses]
        primitive_names = sorted({status["primitive"] for status in statuses})
        split_records[split] = {
            "file": final.name, "sha256": sha256_file(final), "bytes": final.stat().st_size,
            "seed": split_seeds[split], "episode_seeds": episode_seeds,
            "episodes": episode_count, "transitions": len(observations),
            "teacher_reach_rate": float(np.mean(successes)),
            "terminal_failures": int(sum(status["failure_reason"] is not None for status in statuses)),
            "primitive_episode_counts": {
                name: int(sum(status["primitive"] == name for status in statuses))
                for name in primitive_names},
            "primitive_reach_rate": {
                name: float(np.mean([status["success"] for status in statuses
                                     if status["primitive"] == name]))
                for name in primitive_names},
            "mean_episode_seconds": float(np.mean([status["seconds"] for status in statuses])),
            "maximum_commanded_torque_nm": float(max(status["maximum_commanded_torque_nm"] for status in statuses)),
            "maximum_actual_torque_nm": float(max(status["maximum_actual_torque_nm"] for status in statuses)),
            "maximum_tilt_rad": float(max(status["peak_tilt_rad"] for status in statuses)),
            "generation_seconds": time.perf_counter() - started,
        }
        print(json.dumps({"phase": "drive_collection", "split": split,
                          **{key: split_records[split][key] for key in
                             ("episodes", "transitions", "teacher_reach_rate", "generation_seconds")}}),
              flush=True)
    manifest = {
        **expected,
        "method": "precision-curriculum disturbed teacher rollouts through FleetSimulation(only=['lab']) native wheel contacts",
        "episode_reset": "free-joint pose is written only at each episode boundary",
        "corridor_xy_m": [0.16, 0.46, 0.82, 0.72],
        "mujoco_version": mujoco.__version__,
        "workers": workers,
        "splits": split_records,
        "generation_seconds": time.perf_counter() - overall_started,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def load_drive_split(root: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if split not in manifest["splits"]:
        raise ValueError(f"Drive dataset has no {split!r} split")
    with np.load(root / manifest["splits"][split]["file"], allow_pickle=False) as archive:
        return archive["observations"].copy(), archive["targets"].copy()


def _collect_policy_chunk(payload: tuple[str, list[int], int, dict]) -> dict:
    weights, seeds, domain_seed, config_values = payload
    config = DriveConfig(**config_values)
    policy = BestDrivePolicy(weights)
    simulation = FleetSimulation(seed=domain_seed, randomize=False, only=["lab"])
    nominal = drive_domain_nominal(simulation)
    episodes = [rollout_episode(simulation, seed, config, disturbance_fraction=0,
                                policy=policy.normalized, nominal=nominal) for seed in seeds]
    return {
        "observations": np.concatenate([episode["observations"] for episode in episodes]),
        "targets": np.concatenate([episode["targets"] for episode in episodes]),
        "status": [episode["status"] for episode in episodes],
    }


def collect_dagger_round(weights: str | Path, seeds: list[int], config: DriveConfig,
                         workers: int, domain_seed: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Label learner-visited native physics states with the geometric teacher."""
    if not seeds:
        raise ValueError("DAgger collection needs at least one episode")
    worker_count = min(max(1, workers), len(seeds))
    chunks = [seeds[index::worker_count] for index in range(worker_count)]
    payloads = [(str(weights), chunk, domain_seed + index, asdict(config))
                for index, chunk in enumerate(chunks)]
    if len(payloads) == 1:
        results = [_collect_policy_chunk(payloads[0])]
    else:
        with _process_pool(len(payloads)) as executor:
            results = list(executor.map(_collect_policy_chunk, payloads))
    return (np.concatenate([result["observations"] for result in results]),
            np.concatenate([result["targets"] for result in results]),
            [status for result in results for status in result["status"]])


def export_drive_policy(model: BestDriveMLP, config: DriveConfig, path: str | Path) -> str:
    arrays = {"config": np.array(json.dumps(asdict(config), sort_keys=True))}
    linear_index = 0
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            arrays[f"weight_{linear_index}"] = layer.weight.detach().cpu().numpy()
            arrays[f"bias_{linear_index}"] = layer.bias.detach().cpu().numpy()
            linear_index += 1
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    return sha256_file(path)


class BestDrivePolicy:
    """Dependency-light NumPy inference for the exported drive controller."""

    def __init__(self, weights: str | Path):
        with np.load(weights, allow_pickle=False) as archive:
            self.config = DriveConfig(**json.loads(str(archive["config"])))
            indices = sorted(int(name.split("_")[1]) for name in archive.files if name.startswith("weight_"))
            self.weights = [(archive[f"weight_{index}"].copy(), archive[f"bias_{index}"].copy())
                            for index in indices]

    def normalized(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape[-1] != self.config.observation_dim:
            raise ValueError(f"Drive policy expects {self.config.observation_dim} observation values")
        for weight, bias in self.weights:
            value = np.tanh(value @ weight.T + bias)
        return np.clip(value, -1, 1)

    __call__ = normalized

    def command(self, goal_xy: Iterable[float], goal_heading: float,
                pose_xy: Iterable[float], heading: float,
                wheel_speeds: Iterable[float],
                body_velocity: Iterable[float] = (0.0, 0.0)) -> tuple[float, float]:
        observation = drive_observation(goal_xy, goal_heading, pose_xy, heading,
                                        wheel_speeds, body_velocity, self.config)
        action = self.normalized(observation)
        return (float(action[0] * self.config.max_forward_m_s),
                float(action[1] * self.config.max_yaw_rad_s))


class BestDriveController:
    """Stateful adapter from a FleetSimulation pose to the exported policy."""

    def __init__(self, policy: BestDrivePolicy, key: str = "lab"):
        self.policy = policy
        self.key = key
        self.previous_pose = None
        self.previous_time = None

    def reset(self, simulation: FleetSimulation) -> None:
        self.previous_pose = simulation.pose(self.key)
        self.previous_time = float(simulation.data.time)

    def command(self, simulation: FleetSimulation, goal_xy: Iterable[float],
                goal_heading: float) -> tuple[float, float]:
        pose_xy, heading = simulation.pose(self.key)
        current_time = float(simulation.data.time)
        elapsed = current_time - self.previous_time if self.previous_time is not None else 0.0
        if self.previous_pose is None or elapsed <= 0:
            body_velocity = (0.0, 0.0)
        else:
            previous_xy, previous_heading = self.previous_pose
            world_velocity = (pose_xy - previous_xy) / elapsed
            forward_axis = np.array((math.cos(heading), math.sin(heading)))
            body_velocity = (float(np.dot(world_velocity, forward_axis)),
                             wrap(heading - previous_heading) / elapsed)
        self.previous_pose = (pose_xy.copy(), heading)
        self.previous_time = current_time
        return self.policy.command(goal_xy, goal_heading, pose_xy, heading,
                                   _wheel_speeds(simulation, self.key), body_velocity)


def _collect_ppo_chunk(payload: tuple[str, list[int], int, dict, float, float]) -> dict:
    weights, seeds, domain_seed, config_values, action_std, time_penalty = payload
    config = DriveConfig(**config_values)
    policy = BestDrivePolicy(weights)
    simulation = FleetSimulation(seed=domain_seed, randomize=False, only=["lab"])
    nominal = drive_domain_nominal(simulation)
    episodes = [rollout_ppo_episode(
        simulation, seed, config, policy.normalized, action_std,
        time_penalty, nominal) for seed in seeds]
    return {
        "observations": np.concatenate([episode["observations"] for episode in episodes]),
        "actions": np.concatenate([episode["actions"] for episode in episodes]),
        "rewards": np.concatenate([episode["rewards"] for episode in episodes]),
        "dones": np.concatenate([episode["dones"] for episode in episodes]),
        "timestamps": np.concatenate([episode["timestamps"] for episode in episodes]),
        "status": [episode["status"] for episode in episodes],
    }


def collect_ppo_rollouts(weights: str | Path, seeds: list[int], config: DriveConfig,
                         workers: int, domain_seed: int, action_std: float,
                         time_penalty_per_second: float) -> dict:
    """Collect stochastic policy transitions from parallel native worlds."""
    if not seeds:
        raise ValueError("PPO collection needs at least one episode")
    worker_count = min(max(1, workers), len(seeds))
    chunks = [seeds[index::worker_count] for index in range(worker_count)]
    payloads = [(str(weights), chunk, domain_seed + index, asdict(config), action_std,
                 time_penalty_per_second) for index, chunk in enumerate(chunks)]
    if len(payloads) == 1:
        results = [_collect_ppo_chunk(payloads[0])]
    else:
        with _process_pool(len(payloads)) as executor:
            results = list(executor.map(_collect_ppo_chunk, payloads))
    return {
        key: np.concatenate([result[key] for result in results])
        for key in ("observations", "actions", "rewards", "dones", "timestamps")
    } | {"status": [status for result in results for status in result["status"]]}


def _evaluate_policy_chunk(payload: tuple[str, list[int], int, dict, str]) -> list[dict]:
    weights, seeds, domain_seed, config_values, policy_kind = payload
    config = DriveConfig(**config_values)
    policy = BestDrivePolicy(weights) if policy_kind == "learned" else None
    simulation = FleetSimulation(seed=domain_seed, randomize=False, only=["lab"])
    nominal = drive_domain_nominal(simulation)
    selected_policy = policy.normalized if policy is not None else policy_kind
    return [rollout_episode(simulation, seed, config, disturbance_fraction=0,
                            policy=selected_policy, nominal=nominal)["status"] for seed in seeds]


def evaluate_drive_policy(weights: str | Path, seeds: list[int], config: DriveConfig,
                          workers: int, domain_seed: int,
                          policy_kind: str = "learned") -> list[dict]:
    """Score a saved policy on bounded, held-out native physics episodes."""
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    worker_count = min(max(1, workers), len(seeds))
    chunks = [seeds[index::worker_count] for index in range(worker_count)]
    if policy_kind not in {"learned", "teacher", "zero"}:
        raise ValueError("policy_kind must be learned, teacher, or zero")
    payloads = [(str(weights), chunk, domain_seed + index, asdict(config), policy_kind)
                for index, chunk in enumerate(chunks)]
    if len(payloads) == 1:
        results = [_evaluate_policy_chunk(payloads[0])]
    else:
        with _process_pool(len(payloads)) as executor:
            results = list(executor.map(_evaluate_policy_chunk, payloads))
    return [status for result in results for status in result]
