"""Gymnasium interface for a reference cylinder-pushing task."""
from __future__ import annotations

import copy
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .runtime import ArenaSimulation


class ArenaEnv(gym.Env):
    """Push one patient cylinder toward the healthcare area's center.

    Actions are normalized left/right forward wheel-speed commands. The
    observation contains world-frame robot qpos/qvel, wheel encoders/speeds,
    each movable object's qpos/qvel, goal XY, target index, applied action,
    and elapsed episode fraction. Object state is privileged simulator data.
    This reference reward is independent of the competition scoring rules.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, config="senior_preliminary", tape_mode="flex", *,
                 randomize=False, seed=None, profile=None, frame_skip=50,
                 max_episode_seconds=120.0, action_delay_s=0.02,
                 motor_lag_s=0.04, action_noise_std=0.0,
                 goal_xy=(0.09, 0.6095), success_radius_m=0.04,
                 max_tape_damage_fraction=0.20, render_mode=None,
                 render_width=640, render_height=640):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if int(frame_skip) != frame_skip or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        values = (max_episode_seconds, action_delay_s, motor_lag_s, action_noise_std,
                  success_radius_m, max_tape_damage_fraction)
        if not np.isfinite(values).all() or min(values) < 0 or max_episode_seconds == 0 or success_radius_m == 0:
            raise ValueError("Episode and controller parameters must be finite and nonnegative")
        if max_tape_damage_fraction > 1:
            raise ValueError("max_tape_damage_fraction must be in [0, 1]")
        self.sim = ArenaSimulation(config=config, tape_mode=tape_mode, robot=True,
                                   randomize=randomize, seed=seed, profile=profile)
        self.frame_skip = int(frame_skip)
        self.max_episode_seconds = float(max_episode_seconds)
        self.action_delay_s = float(action_delay_s)
        self.motor_lag_s = float(motor_lag_s)
        self.action_noise_std = float(action_noise_std)
        self.goal_xy = np.asarray(goal_xy, dtype=float)
        if self.goal_xy.shape != (2,) or not np.isfinite(self.goal_xy).all():
            raise ValueError("goal_xy must contain two finite coordinates")
        self.success_radius_m = float(success_radius_m)
        self.max_tape_damage_fraction = float(max_tape_damage_fraction)
        self.render_mode = render_mode
        self.render_width, self.render_height = int(render_width), int(render_height)
        self._renderer = None
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._bind_model()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(23 + 13 * len(self._objects),), dtype=np.float32)
        self.reset(seed=seed)

    @property
    def model(self):
        return self.sim.model

    @property
    def data(self):
        return self.sim.data

    def _joint_addresses(self, name):
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"Missing joint {name!r}")
        return int(self.model.jnt_qposadr[joint]), int(self.model.jnt_dofadr[joint])

    def _bind_model(self):
        self._robot_qpos, self._robot_dof = self._joint_addresses("robot_free")
        self._wheel_addresses = [self._joint_addresses(name) for name in ("wheel_left", "wheel_right")]
        self._robot_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "reference_robot")
        self._objects = []
        for entry in self.sim.metadata["objects"]:
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, entry["name"])
            if body < 0 or self.model.body_jntnum[body] == 0:
                raise ValueError(f"Movable object has no body/joint: {entry['name']}")
            joint = int(self.model.body_jntadr[body])
            if self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"Movable object must have a free joint: {entry['name']}")
            self._objects.append((entry, int(self.model.jnt_qposadr[joint]), int(self.model.jnt_dofadr[joint])))
        candidates = [i for i, (entry, _, _) in enumerate(self._objects)
                      if entry.get("category") == "patient" or "Cylinder_Red" in entry["name"]]
        if not candidates:
            raise ValueError("Reference pushing task requires a patient cylinder")
        self.target_index = candidates[0]
        self.dt = float(self.model.opt.timestep) * self.frame_skip
        self.metadata = {**type(self).metadata, "render_fps": round(1 / self.dt)}
        self._delay_steps = int(np.ceil(self.action_delay_s / self.dt))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        # Gym's RNG supplies a reproducible stream of physics randomizations.
        sim_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.sim.reset(seed=sim_seed)
        self._bind_model()
        if options and "target_index" in options:
            index = int(options["target_index"])
            if not 0 <= index < len(self._objects):
                raise ValueError("target_index is outside the movable object list")
            self.target_index = index
        self._queue = deque(np.zeros(2) for _ in range(self._delay_steps))
        self._filtered_action = np.zeros(2)
        self._steps = 0
        self._done = False
        self._potential = self._task_potential()
        return self._observation(), self._info()

    def _target_state(self):
        _, qpos, dof = self._objects[self.target_index]
        return self.data.qpos[qpos:qpos + 7], self.data.qvel[dof:dof + 6]

    def _task_potential(self):
        target, _ = self._target_state()
        direction = self.goal_xy - target[:2]
        distance = float(np.linalg.norm(direction))
        behind = target[:2] - 0.11 * direction / max(distance, 1e-12)
        robot = self.data.qpos[self._robot_qpos:self._robot_qpos + 2]
        return -distance - 0.2 * float(np.linalg.norm(robot - behind))

    def _observation(self):
        values = [self.data.qpos[self._robot_qpos:self._robot_qpos + 7],
                  self.data.qvel[self._robot_dof:self._robot_dof + 6],
                  [self.data.qpos[q] for q, _ in self._wheel_addresses],
                  [self.data.qvel[v] for _, v in self._wheel_addresses]]
        for _, q, v in self._objects:
            values.extend((self.data.qpos[q:q + 7], self.data.qvel[v:v + 6]))
        values.extend((self.goal_xy, [self.target_index], self._filtered_action,
                       [min(self._steps * self.dt / self.max_episode_seconds, 1.0)]))
        return np.concatenate(values).astype(np.float32)

    def _info(self):
        target, velocity = self._target_state()
        info = {"target_name": self._objects[self.target_index][0]["name"],
                "target_distance_m": float(np.linalg.norm(target[:2] - self.goal_xy)),
                "target_speed_m_s": float(np.linalg.norm(velocity[:3])),
                "elapsed_seconds": self._steps * self.dt,
                "is_success": False, "task": "reference_push_to_healthcare"}
        if self.sim.tape is not None:
            info.update(self.sim.tape.metrics())
        return info

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode is complete; call reset before step")
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must contain two finite wheel-speed commands")
        command = np.clip(action, -1.0, 1.0)
        if self.action_noise_std:
            command = np.clip(command + self.np_random.normal(0, self.action_noise_std, 2), -1, 1)
        self._queue.append(command.copy())
        delayed = self._queue.popleft()
        # Apply the command lag at the physics rate, independently of frame_skip.
        physics_dt = float(self.model.opt.timestep)
        alpha = -np.expm1(-physics_dt / self.motor_lag_s) if self.motor_lag_s else 1.0
        for _ in range(self.frame_skip):
            self._filtered_action += alpha * (delayed - self._filtered_action)
            self.sim.step(self._filtered_action, nstep=1)
        self._steps += 1
        potential = self._task_potential()
        reward = 10 * (potential - self._potential) - 0.001 * float(np.dot(command, command)) - 0.001
        self._potential = potential
        info = self._info()
        robot_pos = self.data.qpos[self._robot_qpos:self._robot_qpos + 3]
        target, _ = self._target_state()
        upright = self.data.xmat[self._robot_body].reshape(3, 3)[2, 2] > np.cos(np.deg2rad(55))
        bounds = lambda p: bool(-0.08 <= p[0] <= 1.223 and -0.08 <= p[1] <= 1.261 and p[2] > -0.05)
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        reasons = []
        if not finite:
            reasons.append("nonfinite_physics")
        if not bounds(robot_pos) or not bounds(target[:3]):
            reasons.append("outside_arena")
        if not upright:
            reasons.append("robot_tipped")
        if info.get("tape_damage_fraction", 0.0) > self.max_tape_damage_fraction:
            reasons.append("tape_damage_limit")
        if info.get("tape_material_limit_exceeded", False):
            reasons.append("tape_material_limit")
        if self._steps * self.dt >= self.max_episode_seconds:
            reasons.append("time_limit")
        success = (finite and not reasons and info["target_distance_m"] < self.success_radius_m
                   and info["target_speed_m_s"] < 0.05)
        info["is_success"] = success
        info["truncation_reasons"] = reasons
        if success:
            reward += 10.0
        self._done = bool(success or reasons)
        return self._observation(), float(reward), bool(success), bool(reasons), info

    def get_state(self):
        """Capture physics, random stream, action history, and reward state."""
        return copy.deepcopy({"version": 1, "simulation": self.sim.get_state(),
                              "rng": self.np_random.bit_generator.state,
                              "queue": [v.tolist() for v in self._queue],
                              "filtered_action": self._filtered_action.tolist(),
                              "steps": self._steps, "done": self._done,
                              "potential": self._potential, "target_index": self.target_index,
                              "goal_xy": self.goal_xy.tolist(),
                              "controller": [self.frame_skip, self.action_delay_s, self.motor_lag_s,
                                             self.action_noise_std, self.max_episode_seconds,
                                             self.success_radius_m, self.max_tape_damage_fraction]})

    def set_state(self, state):
        """Restore a checkpoint into an environment with matching settings."""
        current = self.get_state()
        if state.get("version") != 1 or state.get("controller") != current["controller"]:
            raise ValueError("Environment state version or controller settings do not match")
        if len(state["queue"]) != self._delay_steps:
            raise ValueError("State action-delay queue has the wrong length")
        if not 0 <= state["target_index"] < len(self._objects):
            raise ValueError("State target index is outside the movable object list")
        vectors = [np.asarray(value, dtype=float) for value in
                   [*state["queue"], state["filtered_action"], state["goal_xy"]]]
        if any(v.shape != (2,) or not np.isfinite(v).all() for v in vectors):
            raise ValueError("State contains an invalid action or goal vector")
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.sim.set_state(copy.deepcopy(state["simulation"]))
        self._bind_model()
        self.np_random.bit_generator.state = copy.deepcopy(state["rng"])
        self._queue = deque(v.copy() for v in vectors[:-2])
        self._filtered_action = vectors[-2].copy()
        self.goal_xy = vectors[-1].copy()
        self._steps, self._done = int(state["steps"]), bool(state["done"])
        self._potential, self.target_index = float(state["potential"]), int(state["target_index"])

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.5715, 0.5905, 0]
        camera.distance, camera.azimuth, camera.elevation = 1.65, 90, -90
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


ArenaPushEnv = ArenaEnv
