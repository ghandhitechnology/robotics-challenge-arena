#!/usr/bin/env python3
"""Replay and audit a forty-module magnetic-body recording from native reset.

Teacher recordings can pass recording-integrity checks with --allow-teacher.
Only a successful physical task, accepted neural checkpoint, and complete native
replay can produce final_neural_proof=true. Report claims never enable a gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_mujoco.swarm_body_env import FEATURES, SwarmBodyEnv
from arena_mujoco.swarm_flow import connected_components
from arena_mujoco.swarm_policy import NumpySwarmPolicy
from scripts.verify_swarm_proof import (
    audit_neural_actions, audit_scene, audit_transport, digest, object_bottom, require,
)

ARTIFACTS = {"scene.xml", "metadata.json", "trajectory.npz", "policy_trace.npz"}
SOURCES = {
    "arena_mujoco/swarm_env.py", "arena_mujoco/swarm_body_env.py",
    "arena_mujoco/swarm_flow.py", "arena_mujoco/swarm_magnets.py",
    "arena_mujoco/swarm_robot.py", "arena_mujoco/swarm_policy.py",
    "arena_mujoco/builder.py", "arena_mujoco/materials.py", "arena_spec.json",
    "scripts/run_swarm_body.py", "scripts/verify_swarm_body.py",
    "scripts/verify_swarm_proof.py", "scripts/train_swarm_policy.py",
}
OBSERVATIONS = ("local", "neighbors", "neighbor_mask", "active", "global")
TASK = "magnetic_swarm_body_transport"
ENV_FACTORY = "arena_mujoco.swarm_body_env:SwarmBodyEnv"
FORCE_THRESHOLD_N = .002
START_BOUNDS = np.array([.863, .701, 1.143, 1.181])
CONTRACT = {"local_dim": 40, "neighbor_dim": 12, "global_dim": 88, "action_dim": 4}


def source_hashes():
    return {name: digest(ROOT / name) for name in sorted(SOURCES)}


def make_environment(report):
    """Repeat the recorder's constructor, curriculum assignment, and final reset."""
    env = SwarmBodyEnv(num_envs=1, num_robots=int(report["robot_count"]),
                       num_objects=int(report["object_count"]), backend="native", device="cpu",
                       native_workers=1, seed=int(report["seed"]),
                       timestep=float(report["physics_timestep_s"]),
                       episode_seconds=float(report["episode_seconds"]))
    env.set_curriculum({"num_active_robots": int(report["robot_count"]),
                        "num_active_objects": int(report["object_count"]),
                        "difficulty": float(report["difficulty"])})
    observation = env.reset()
    attach_substep_audit(env)
    return env, observation


def attach_substep_audit(env):
    """Observe each native force sample without changing integration or forces."""
    magnets = env.magnets[0]
    original_apply = magnets.apply
    module_mask = np.zeros(env.model.nbody, dtype=bool)
    module_mask[magnets.robot_bodies] = True
    audit = {"count": 0, "min_largest_component": 1, "nonmodule_wrench_max": 0.,
             "generalized_force_max": 0.}

    def observe(data, enabled):
        if audit["count"] % env.substeps == 0:
            audit["min_largest_component"] = env.num_robots
            audit["nonmodule_wrench_max"] = audit["generalized_force_max"] = 0.
        result = original_apply(data, enabled)
        largest = connected_components(magnets.load_bearing_graph(FORCE_THRESHOLD_N))[1][0]
        audit["min_largest_component"] = min(audit["min_largest_component"], largest)
        audit["nonmodule_wrench_max"] = max(audit["nonmodule_wrench_max"],
                                            float(np.max(np.abs(data.xfrc_applied[~module_mask]))))
        audit["generalized_force_max"] = max(audit["generalized_force_max"],
                                            float(np.max(np.abs(data.qfrc_applied))))
        audit["count"] += 1
        return result

    magnets.apply = observe
    env._proof_substep_audit = audit


def capture_frame(env):
    """Read current state; never refresh forces, install poses, or change links.

    Magnetic arrays are the actual forces applied in the last native substep.
    Their timestamp precedes the post-integration pose by one physics timestep.
    Reset starts with zero applied field and no sampled interactions.
    """
    data, magnets = env.native_data[0], env.magnets[0]
    payload_contacts = np.zeros(env.num_objects, dtype=np.int32)
    for contact in data.contact[:data.ncon]:
        if contact.dist > .0003:
            continue
        a, b = int(contact.geom1), int(contact.geom2)
        for first, second in ((a, b), (b, a)):
            obj = env.geom_object[first]
            if obj >= 0 and env.geom_robot[second] >= 0:
                payload_contacts[obj] += 1
    return {
        "times": np.array(float(data.time)), "qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(), "xfrc_applied": data.xfrc_applied.copy(),
        "qfrc_applied": data.qfrc_applied.copy(),
        "object_phases": env.phase[0].numpy().copy(),
        "pad_contacts": env.pad_contact[0].numpy().astype(np.uint8).copy(),
        "object_floor_contacts": env.object_floor_contact[0].numpy().astype(np.uint8).copy(),
        "payload_robot_contacts": payload_contacts, "body_phase": env.body_phase[0].numpy().copy(),
        "magnet_commands": env.magnet_command[0].numpy().copy(),
        "magnetic_enabled": ((env.magnet_command[0].numpy() > 0) & env.active[0].numpy().astype(bool)),
        "magnetic_graph": magnets.load_bearing_graph(FORCE_THRESHOLD_N),
        "magnetic_interaction_graph": magnets.interaction_graph(),
        "magnetic_close_graph": magnets.contact_graph(),
        "magnetic_pair_forces_n": magnets.pair_force_magnitudes_n.copy(),
        "magnetic_pair_force_vectors_n": magnets.pair_force_vectors_n.copy(),
        "magnetic_force_time_s": np.array(max(0., float(data.time)-env.timestep)),
        "substep_min_largest_component": np.array(env._proof_substep_audit["min_largest_component"]),
        "substep_nonmodule_wrench_max": np.array(env._proof_substep_audit["nonmodule_wrench_max"]),
        "substep_generalized_force_max": np.array(env._proof_substep_audit["generalized_force_max"]),
    }


def initial_footprints(model, data, robots):
    """Exact world-XY bounds of initial physical colliders, including child parts."""
    bounds = []
    for robot in robots:
        root_body = model.body(robot["name"]).id
        selected = np.flatnonzero((model.body_rootid[model.geom_bodyid] == root_body) &
                                  ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        minima, maxima = [], []
        for geom in selected:
            matrix = data.geom_xmat[geom].reshape(3, 3)
            position = data.geom_xpos[geom]
            size, kind = model.geom_size[geom], model.geom_type[geom]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[geom]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start:start+count] @ matrix.T + position
                minima.append(vertices.min(0)[:2]); maxima.append(vertices.max(0)[:2])
                continue
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                extent = np.abs(matrix) @ size
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = np.full(3, size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                extent = size[0]*np.linalg.norm(matrix[:, :2], axis=1) + size[1]*np.abs(matrix[:, 2])
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent = size[0] + size[1]*np.abs(matrix[:, 2])
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                extent = np.linalg.norm(matrix*size, axis=1)
            else:
                raise ValueError(f"Unsupported robot collider type {kind}")
            minima.append((position-extent)[:2]); maxima.append((position+extent)[:2])
        if not minima:
            raise ValueError(f"Robot has no physical footprint: {robot['name']}")
        bounds.append(np.r_[np.min(minima, axis=0), np.max(maxima, axis=0)])
    return np.asarray(bounds)


def audit_training(training, weight_hash, config, errors):
    """Recompute checkpoint acceptance from declared training measurements."""
    require(training.get("weights_sha256") == weight_hash, "Training weights SHA-256 mismatch", errors)
    require(training.get("acceptance_passed") is True, "Training acceptance gate did not pass", errors)
    require(any(name in str(training.get("gpu", "")) for name in ("A100", "H100"))
            and isinstance(training.get("cuda"), str) and bool(training.get("cuda"))
            and training.get("cuda_optimization") is True
            and str(training.get("policy_device", "")).startswith("cuda")
            and training.get("backend") == "native",
            "Training lacks A100/H100 CUDA optimization with native magnetic physics", errors)
    require(training.get("ppo_updates", 0) > 0 and training.get("cumulative_ppo_actor_update_l2", 0) > 1e-6
            and training.get("final_stage") == 3,
            "Training lacks completed full-stage PPO actor updates", errors)
    held = training.get("heldout", {})
    episodes, rate = held.get("episodes", 0), held.get("success_rate", -1)
    valid_rate = (isinstance(episodes, int) and not isinstance(episodes, bool) and episodes >= 32
                  and isinstance(rate, (int, float)) and math.isfinite(rate) and 0 <= rate <= 1)
    lower = -1.
    if valid_rate:
        z = 1.96
        lower = (rate+z*z/(2*episodes)-z*math.sqrt(rate*(1-rate)/episodes+z*z/(4*episodes**2)))/(1+z*z/episodes)
    require(valid_rate and held.get("complete") is True and rate >= .8 and lower >= .6,
            "Held-out results fail the 32-episode, 80% success, Wilson lower-bound gates", errors)
    require(isinstance(held.get("success_wilson_lower_95"), (int, float))
            and abs(held.get("success_wilson_lower_95", -2)-lower) < 1e-8,
            "Reported held-out Wilson bound disagrees with episode count and success rate", errors)
    require(held.get("stage") == 3 and held.get("policy") == "learned",
            "Held-out evaluation must use the learned actor at full difficulty", errors)
    baseline = training.get("zero_baseline", {})
    baseline_rate = baseline.get("success_rate", -1)
    require(baseline.get("complete") is True and baseline.get("episodes", 0) >= 32
            and baseline.get("policy") == "zero" and baseline.get("stage") == 3
            and isinstance(baseline_rate, (int, float)) and math.isfinite(baseline_rate)
            and 0 <= baseline_rate <= 1 and rate > baseline_rate+.2,
            "Training lacks the required held-out improvement over zero actions", errors)
    args = training.get("args", {})
    require(args.get("env_factory") == ENV_FACTORY and args.get("robots") == 40
            and args.get("objects") == 2 and args.get("cpu_smoke") is False,
            "Training configuration is not the complete forty-module magnetic-body task", errors)
    require(training.get("config") == config and all(config.get(k) == v for k, v in CONTRACT.items()),
            "Training/export feature contract must be 40 local, 12 neighbor, 88 global, 4 actions", errors)
    return {"training_heldout_wilson_lower_95_recomputed": lower,
            "training_hardware_provenance": "reported A100/H100 CUDA optimization"}


def _longest_interval(mask, times):
    longest = current = 0.
    for index in range(1, len(times)):
        current = current+float(times[index]-times[index-1]) if mask[index] and mask[index-1] else 0.
        longest = max(longest, current)
    return longest


def audit_body_task(trajectory, trace, model, metadata, report, errors):
    """Derive body motion, release/rejoin, connectivity, transport and hold."""
    times, poses = trajectory["times"], trajectory["qpos"]
    addresses = np.array([model.joint(r["name"]+"_free").qposadr[0] for r in metadata["robots"]])
    robot_positions = poses[:, addresses[:, None]+np.arange(2)]
    net = np.linalg.norm(robot_positions[-1]-robot_positions[0], axis=-1)
    path = np.linalg.norm(np.diff(robot_positions, axis=0), axis=-1).sum(0)
    graph = trajectory["magnetic_graph"].astype(bool)
    largest = np.array([connected_components(frame)[1][0] for frame in graph])
    connected = (largest == 40) & (trajectory["substep_min_largest_component"] == 40)
    release_commands = np.any(trace["actions"][..., 3] <= 0, axis=1)
    release_indices = np.flatnonzero(release_commands)
    release = int(release_indices[0]) if len(release_indices) else None
    before = len(times) if release is None else release+1
    connected_before = _longest_interval(connected[:before], times[:before])
    require(connected_before >= 1.-1e-7,
            "All forty modules must remain force-connected for one second before the first release command", errors)
    require(release is not None, "Recording contains no commanded magnetic release", errors)
    previous, following = graph[:-1], graph[1:]
    formed = np.triu(following & ~previous, 1)
    broken = np.triu(previous & ~following, 1)
    released_robots = trace["actions"][..., 3] <= 0
    requested_edges = released_robots[:, :, None] | released_robots[:, None, :]
    commanded_breaks = broken & requested_edges
    require(commanded_breaks.any(), "No force-bearing magnetic edge actually breaks on a release command", errors)
    reformed = 0 if release is None else int(formed[release+1:].sum())
    require(reformed > 0, "No magnetic links form after deliberate release", errors)
    require(np.all(net >= .20), "At least one of forty robots moved less than 0.20 m net", errors)
    centered = robot_positions-robot_positions.mean(axis=1, keepdims=True)
    covariance = np.einsum('fni,fnj->fij', centered, centered)/40
    eigen = np.maximum(np.linalg.eigvalsh(covariance), 0)
    aspect = np.sqrt((eigen[:, 1]+1e-12)/(eigen[:, 0]+1e-12))
    completion = report.get("completion_time_s")
    finish_index = len(times)-1
    if isinstance(completion, (int, float)) and math.isfinite(completion):
        matches = np.flatnonzero(np.abs(times-completion) < 1e-7)
        if len(matches) == 1:
            finish_index = int(matches[0])
            hold = np.arange(len(times)) >= finish_index
            require(np.all(connected[hold]), "All forty robots must stay force-connected throughout the final hold", errors)
            require(np.all(trajectory["payload_robot_contacts"][hold] == 0),
                    "Payload still contacts a robot during the final floor-release hold", errors)
            if hold.sum() > 1:
                speed = np.linalg.norm(np.diff(robot_positions[hold], axis=0), axis=-1)/np.diff(times[hold])[:, None]
                require(np.all(speed < .015), "A module keeps moving during the final hold", errors)
    interval_components = np.minimum(largest, trajectory["substep_min_largest_component"])
    middle = interval_components[max(1, release or 1):max(2, finish_index+1)]
    checks = {
        "robot_net_displacement_m": net.tolist(), "robot_path_distance_m": path.tolist(),
        "all_forty_connected_before_release_s": connected_before,
        "first_magnet_release_command_s": None if release is None else float(times[release]),
        "commanded_magnetic_edge_breaks": int(commanded_breaks.sum()),
        "magnetic_edges_formed_after_release": reformed,
        "new_neighbor_pairs_after_release": (0 if release is None else int(np.any(formed[release+1:], axis=0)[
            ~np.any(graph[:release+1], axis=0)].sum())),
        "magnetic_edge_formations": int(formed.sum()), "magnetic_edge_breaks": int(broken.sum()),
        "middle_at_least_32_connected_fraction": float(np.mean(middle >= 32)) if len(middle) else 0.,
        "middle_connectivity_threshold_enforced": False,
        "initial_shape_aspect_ratio": float(aspect[0]), "final_shape_aspect_ratio": float(aspect[-1]),
        "shape_aspect_ratio_range": [float(aspect.min()), float(aspect.max())],
        "final_largest_force_connected_component": int(largest[-1]),
    }
    positions, bottoms, heights = object_bottom(poses, model, metadata["objects"])
    goals = np.asarray(report["goals_m"], dtype=float)
    checks.update(audit_transport(times, positions, bottoms, trajectory["pad_contacts"],
                                  trajectory["object_floor_contacts"], trajectory["object_phases"], goals,
                                  completion, .15, errors, heights))
    if "final_hold_s" in checks:
        require(checks["final_hold_s"] >= report["requested_hold_seconds"]-1e-7,
                "Physical final hold is shorter than the requested duration", errors)
    if report.get("failure") is not None:
        errors.append("Recording ended with a simulator/task failure: "+str(report["failure"]))
    return checks


def _array_shapes(model, frames, robots, objects):
    return {
        "times": (frames,), "qpos": (frames, model.nq), "qvel": (frames, model.nv),
        "ctrl": (frames, model.nu), "xfrc_applied": (frames, model.nbody, 6),
        "qfrc_applied": (frames, model.nv), "object_phases": (frames, objects),
        "pad_contacts": (frames, robots), "object_floor_contacts": (frames, objects),
        "payload_robot_contacts": (frames, objects), "body_phase": (frames,),
        "magnet_commands": (frames, robots), "magnetic_enabled": (frames, robots),
        "magnetic_graph": (frames, robots, robots),
        "magnetic_interaction_graph": (frames, robots, robots),
        "magnetic_close_graph": (frames, robots, robots),
        "magnetic_pair_forces_n": (frames, robots, robots),
        "magnetic_pair_force_vectors_n": (frames, robots, robots, 3),
        "magnetic_force_time_s": (frames,), "substep_min_largest_component": (frames,),
        "substep_nonmodule_wrench_max": (frames,), "substep_generalized_force_max": (frames,),
    }


def audit_recorded_arrays(trajectory, trace, model, metadata, report, errors):
    times = trajectory["times"]
    if not require(times.ndim == 1 and len(times) >= 2, "Trajectory must contain reset and at least one complete step", errors):
        return False
    frames, robots, objects = len(times), len(metadata["robots"]), len(metadata["objects"])
    valid = True
    for key, shape in _array_shapes(model, frames, robots, objects).items():
        value = trajectory[key]
        valid &= require(value.shape == shape and np.isfinite(value).all(), f"Invalid trajectory array: {key}", errors)
    steps = frames-1
    trace_shapes = {"times": (steps,), "actions": (steps, robots, 4),
                    "local": (steps, robots, 40), "neighbors": (steps, robots, 6, 12),
                    "neighbor_mask": (steps, robots, 6), "active": (steps, robots), "global": (steps, 88)}
    for key, shape in trace_shapes.items():
        valid &= require(trace[key].shape == shape and np.isfinite(trace[key]).all(),
                         f"Incomplete recording or invalid policy trace: {key}", errors)
    if not valid:
        return False
    require(abs(times[0]) <= 1e-9 and np.allclose(np.diff(times), .02, atol=1e-8, rtol=0)
            and np.allclose(trace["times"], times[:-1], atol=1e-9, rtol=0),
            "Recording must include every aligned control timestamp from reset", errors)
    require(np.max(np.abs(trace["actions"])) <= 1.+1e-6, "Action exceeds normalized bounds", errors)
    require(np.isin(trajectory["substep_min_largest_component"], np.arange(1, robots+1)).all(),
            "Invalid substep component size", errors)
    require(not np.any(trajectory["substep_nonmodule_wrench_max"]),
            "A physics substep applies an external payload or non-module wrench", errors)
    require(not np.any(trajectory["substep_generalized_force_max"]),
            "A physics substep applies extra generalized forces", errors)
    require(np.all(trace["active"] == 1), "Full recording must activate all forty robots", errors)
    require(np.all(trajectory["magnet_commands"][0] == 1)
            and np.allclose(trajectory["magnet_commands"][1:], trace["actions"][..., 3], atol=1e-7, rtol=0),
            "Saved magnetic commands do not match the fourth policy action", errors)
    for key in ("pad_contacts", "object_floor_contacts", "magnetic_enabled", "magnetic_graph",
                "magnetic_interaction_graph", "magnetic_close_graph"):
        require(np.isin(trajectory[key], [0, 1]).all(), f"Nonbinary trajectory field: {key}", errors)
    require(np.array_equal(trajectory["magnetic_enabled"], trajectory["magnet_commands"] > 0),
            "Magnetic enable state disagrees with commands", errors)
    require(np.isin(trajectory["object_phases"], np.arange(6)).all(), "Invalid payload phase", errors)
    require(np.isin(trajectory["body_phase"], np.arange(4)).all(), "Invalid body phase", errors)
    require(np.allclose(trajectory["magnetic_force_time_s"], np.maximum(0, times-report["physics_timestep_s"]), atol=1e-9, rtol=0),
            "Magnetic force samples have incorrect substep timestamps", errors)
    forces, vectors = trajectory["magnetic_pair_forces_n"], trajectory["magnetic_pair_force_vectors_n"]
    require(np.all(forces >= 0) and np.all(forces <= .6+1e-12), "Magnetic pair force exceeds four finite docks", errors)
    require(np.allclose(vectors, -vectors.transpose(0, 2, 1, 3), atol=1e-12, rtol=0)
            and np.allclose(np.linalg.norm(vectors, axis=-1), forces, atol=1e-12, rtol=0),
            "Magnetic force vectors are not equal/opposite or disagree with their magnitudes", errors)
    interaction = trajectory["magnetic_interaction_graph"].astype(bool)
    bearing = trajectory["magnetic_graph"].astype(bool)
    require(np.array_equal(bearing, interaction & (forces >= FORCE_THRESHOLD_N)),
            "Magnetic body graph is not the measured 2 mN force-bearing graph", errors)
    for key in ("magnetic_graph", "magnetic_interaction_graph", "magnetic_close_graph"):
        graph = trajectory[key]
        require(np.array_equal(graph, graph.transpose(0, 2, 1)) and not np.diagonal(graph, axis1=1, axis2=2).any(),
                f"Magnetic graph is directed or has self-links: {key}", errors)
    body_ids = np.array([model.body(r["name"]).id for r in metadata["robots"]])
    excluded = np.ones(model.nbody, dtype=bool); excluded[body_ids] = False
    require(not np.any(trajectory["xfrc_applied"][:, excluded]), "Applied external wrench exists on a payload or non-module body", errors)
    require(not np.any(trajectory["qfrc_applied"]), "Recording contains extra generalized applied forces", errors)
    require(np.allclose(vectors.sum(2), trajectory["xfrc_applied"][:, body_ids, :3], atol=1e-12, rtol=0),
            "Actual module forces differ from the measured internal magnetic pair forces", errors)
    for joint in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        adr = model.jnt_qposadr[joint]
        require(np.all(np.abs(np.linalg.norm(trajectory["qpos"][:, adr+3:adr+7], axis=-1)-1) < 1e-5),
                "Invalid free-body orientation quaternion", errors)
    return True


def audit_full_replay(env, observation, trajectory, trace, report, policy, errors):
    """Integrate all four commands from a fresh reset; never install saved poses."""
    beginning = len(errors)
    maxima = {key: 0. for key in trajectory}
    observation_maxima = {key: 0. for key in OBSERVATIONS}
    action_error = 0.
    completed = 0
    failures = []
    successes = []
    delivered = 0
    discrete = {"object_phases", "pad_contacts", "object_floor_contacts", "payload_robot_contacts", "body_phase",
                "magnetic_enabled", "magnetic_graph", "magnetic_interaction_graph", "magnetic_close_graph",
                "substep_min_largest_component"}

    def compare_frame(index):
        actual = capture_frame(env)
        for key, value in actual.items():
            maxima[key] = max(maxima[key], float(np.max(np.abs(np.asarray(value, float)-trajectory[key][index]))))

    compare_frame(0)
    try:
        for index, recorded_action in enumerate(trace["actions"]):
            numpy_obs = {key: observation[key].detach().cpu().numpy() for key in OBSERVATIONS}
            for key in OBSERVATIONS:
                observation_maxima[key] = max(observation_maxima[key], float(np.max(np.abs(
                    numpy_obs[key][0].astype(float)-trace[key][index].astype(float)))))
            predicted = env.teacher_action().numpy() if report["policy"] == "teacher" else policy(numpy_obs)
            action_error = max(action_error, float(np.max(np.abs(predicted[0]-recorded_action))))
            observation, _, _, _, info = env.step(torch.as_tensor(recorded_action[None], dtype=torch.float32))
            compare_frame(index+1)
            completed += 1
            failures.append(bool(info["failure"][0]))
            successes.append(bool(info["success"][0]))
            delivered = int(info["delivered"][0])
    except (RuntimeError, ValueError, FloatingPointError) as error:
        errors.append(f"Native replay failed after {completed} actions: {type(error).__name__}: {error}")
    for key, maximum in maxima.items():
        tolerance = 0. if key in discrete else (1e-9 if key in {"times", "magnetic_force_time_s"} else 1e-7)
        require(maximum <= tolerance, f"Native replay mismatch for {key}: {maximum:.9g}", errors)
    for key, maximum in observation_maxima.items():
        tolerance = 0. if key in {"active", "neighbor_mask"} else 2e-5
        require(maximum <= tolerance, f"Native replay observation mismatch for {key}: {maximum:.9g}", errors)
    require(action_error <= 2e-6, f"Recomputed {report['policy']} action mismatch: {action_error:.9g}", errors)
    require(completed == len(trace["actions"]), "Native replay did not integrate every saved action", errors)
    return {"physics_replay_passed": len(errors) == beginning, "replay_actions_completed": completed,
            "replay_max_errors": maxima, "replay_observation_max_errors": observation_maxima,
            "recomputed_action_max_error": action_error, "replay_failure_control_steps": sum(failures),
            "replay_success_control_steps": sum(successes), "replay_completed_objects": delivered}


def _manifest(directory, report, errors):
    artifacts = report.get("artifacts", {})
    require(ARTIFACTS.issubset(artifacts), "Artifact hash manifest is incomplete", errors)
    for name, expected in artifacts.items():
        path = (directory/name).resolve()
        require(path.is_relative_to(directory.resolve()) and path.is_file() and digest(path) == expected,
                f"Artifact hash mismatch: {name}", errors)
    hashes = report.get("source_hashes", {})
    require(SOURCES.issubset(hashes), "Source hash manifest is incomplete", errors)
    for name, expected in hashes.items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and digest(path) == expected,
                f"Source hash mismatch: {name}", errors)
    require(report.get("sources_stable_during_recording") is True,
            "Source files changed while the recording was running", errors)


def verify_body(directory, *, weights=None, training_report=None, allow_teacher=False):
    directory = Path(directory)
    errors, task_errors, checks = [], [], {}
    result = {"valid": False, "recording_valid": False, "task_passed": False, "final_neural_proof": False,
              "physics_replayed": False, "proof_directory": str(directory.resolve()),
              "errors": errors, "task_errors": task_errors, "checks": checks}
    env = None
    try:
        report = json.loads((directory/"report.json").read_text())
        metadata = json.loads((directory/"metadata.json").read_text())
        checks["report_sha256"] = digest(directory/"report.json")
        _manifest(directory, report, errors)
        if errors:
            return result
        require(report.get("task") == TASK and metadata.get("task") == TASK
                and metadata.get("rule_score") is False and report.get("official_competition_score", "missing") is None,
                "Recording is not the magnetic-body benchmark", errors)
        require(report.get("env_factory") == ENV_FACTORY and report.get("simulator") == "native_MuJoCo"
                and report.get("mujoco") == mujoco.__version__, "Native body environment/simulator identity mismatch", errors)
        require(report.get("robot_count") == 40 and report.get("object_count") == 2
                and len(metadata.get("objects", [])) == 2, "Full body proof requires forty robots and two payloads", errors)
        require(report.get("physics_timestep_s") in (.001, .002) and report.get("control_timestep_s") == .02,
                "Unvalidated physics or control timestep", errors)
        require(report.get("magnetic_force_threshold_n") == FORCE_THRESHOLD_N,
                "Body connectivity must use the measured 2 mN force threshold", errors)
        require(report.get("local_feature_names") == FEATURES and report.get("policy_contract") == CONTRACT,
                "Recorded policy feature contract differs from the magnetic-body environment", errors)
        require(isinstance(report.get("episode_seconds"), (float, int)) and 0 < report["episode_seconds"] <= 3600
                and isinstance(report.get("requested_hold_seconds"), (float, int)) and 5 <= report["requested_hold_seconds"] <= 60,
                "Invalid episode duration or final hold shorter than five seconds", errors)
        require(report.get("difficulty") == 1., "Full body recording requires full curriculum difficulty", errors)
        policy_type = report.get("policy")
        require(policy_type == "neural" or (policy_type == "teacher" and allow_teacher),
                "Final proof requires neural actions; teacher diagnostics need --allow-teacher", errors)
        if errors:
            return result
        model = mujoco.MjModel.from_xml_path(str(directory/"scene.xml"))
        audit_scene(model, metadata, errors)
        with np.load(directory/"trajectory.npz", allow_pickle=False) as archive:
            trajectory = {key: archive[key] for key in _array_shapes(model, 0, 40, 2)}
        with np.load(directory/"policy_trace.npz", allow_pickle=False) as archive:
            trace = {key: archive[key] for key in ("times", "actions", *OBSERVATIONS)}
        if not audit_recorded_arrays(trajectory, trace, model, metadata, report, errors):
            return result
        policy = None
        if policy_type == "neural":
            weights = Path(weights) if weights else directory/"weights.npz"
            training_report = Path(training_report) if training_report else directory/"training.json"
            require(weights.is_file() and training_report.is_file(), "Missing neural weights or training report", errors)
            if not weights.is_file() or not training_report.is_file():
                return result
            weight_hash = digest(weights)
            require(report.get("weights_sha256") == weight_hash, "Proof weights SHA-256 mismatch", errors)
            require(report.get("training_report_sha256") == digest(training_report), "Proof training-report SHA-256 mismatch", errors)
            policy = NumpySwarmPolicy(weights)
            training = json.loads(training_report.read_text())
            require({"weights.npz", "training.json"}.issubset(report["artifacts"]),
                    "Neural artifact manifest must bind the checkpoint and training report", errors)
            checks.update(audit_training(training, weight_hash, vars(policy.config), errors))
            require(training.get("args", {}).get("episode_seconds") == report["episode_seconds"],
                    "Deployment episode duration differs from training time-feature scaling", errors)
            require(report.get("neural_calls") == len(trace["actions"]), "Neural call count disagrees with action trace", errors)
            checks.update(audit_neural_actions(policy, trace, errors))
        else:
            require(report.get("neural_calls") == 0 and report.get("weights_sha256") is None
                    and report.get("training_report_sha256") is None,
                    "Teacher diagnostic falsely claims neural provenance", errors)
        env, observation = make_environment(report)
        require(env.xml == (directory/"scene.xml").read_text(), "Saved scene differs from the hashed body scene builder", errors)
        require(env.metadata == metadata, "Saved metadata differs from the hashed body scene builder", errors)
        require(np.allclose(env.goals[0].numpy(), report["goals_m"], atol=1e-8, rtol=0),
                "Saved delivery goals differ from the reset recipe", errors)
        bounds = initial_footprints(env.model, env.native_data[0], env.metadata["robots"])
        require(np.all(bounds[:, :2] >= START_BOUNDS[:2]-1e-8) and np.all(bounds[:, 2:] <= START_BOUNDS[2:]+1e-8),
                "An initial physical robot footprint lies outside the start zone", errors)
        checks["initial_robot_footprints_xy_m"] = bounds.tolist()
        checks.update(audit_full_replay(env, observation, trajectory, trace, report, policy, errors))
        result["physics_replayed"] = checks["physics_replay_passed"]
        if checks["replay_failure_control_steps"]:
            task_errors.append("Native replay encountered a failed physical control step")
        checks.update(audit_body_task(trajectory, trace, model, metadata, report, task_errors))
        result["task_passed"] = not task_errors
        require(report.get("completed_objects") == checks["replay_completed_objects"],
                "Reported delivered count disagrees with native replay", errors)
        require(report.get("recorded_control_steps") == len(trace["actions"])
                and report.get("recorded_frames") == len(trajectory["times"]),
                "Reported frame/action counts disagree with the complete recording", errors)
        if "final_hold_s" in checks:
            require(abs(float(report.get("final_hold_s", -1))-checks["final_hold_s"]) < 1e-7,
                    "Reported hold duration disagrees with recorded timestamps", errors)
        require(report.get("success") is result["task_passed"], "Reported task success disagrees with independent physical checks", errors)
        if report.get("final_neural_proof") is True:
            require(policy_type == "neural" and not errors and not task_errors,
                    "Report falsely claims a final neural proof", errors)
        result["recording_valid"] = not errors
        result["final_neural_proof"] = not errors and not task_errors and policy_type == "neural"
        result["valid"] = result["final_neural_proof"] or (not errors and policy_type == "teacher" and allow_teacher)
        result["evidence_kind"] = "teacher_diagnostic" if policy_type == "teacher" else "neural_magnetic_body"
        checks.update(trajectory_frames=len(trajectory["times"]), traced_actions=len(trace["actions"]),
                      robot_count=40, payload_count=2, independent_free_bodies=42, physical_motor_actuators=120,
                      source_files_checked=len(report["source_hashes"]))
    except (KeyError, ValueError, TypeError, IndexError, OSError, RuntimeError) as error:
        errors.append(f"Malformed or unreplayable body recording: {type(error).__name__}: {error}")
    finally:
        if env is not None:
            env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT/"output/swarm/body_proof")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = verify_body(args.directory, weights=args.weights, training_report=args.training_report,
                         allow_teacher=args.allow_teacher)
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
