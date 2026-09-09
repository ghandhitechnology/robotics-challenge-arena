#!/usr/bin/env python3
"""Verify saved preliminary proof artifacts without rerunning the mission."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_mujoco.competition_events import validate_initial_setup
from arena_mujoco.competition_policy import ACTION_NAMES, ERROR_SCALES, OBSERVATION_NAMES, VELOCITY_LIMITS, NeuralPolicy
from arena_mujoco.competition_scoring import score_competition


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_proof(directory, *, weights=None):
    directory = Path(directory)
    weights = Path(weights) if weights else ROOT / "output/competition/policy/weights.npz"
    errors, checks = [], {}

    def require(condition, message):
        if not condition:
            errors.append(message)
        return bool(condition)

    def finish():
        return {"valid": not errors, "proof_directory": str(directory.resolve()),
                "physics_replayed": False, "checks": checks, "errors": errors}

    required = ["report.json", "metadata.json", "scene.xml", "trajectory.npz",
                "policy_trace.npz", "declaration_state.npz", "final_state.npz"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        errors.append("Missing proof artifacts: " + ", ".join(missing))
        return finish()
    report = json.loads((directory / "report.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    require(report.get("success") is True, "Report does not claim a successful full proof")
    require(report.get("failure") is None, "Report contains a mission failure")
    require(report.get("policy") == "neural", "Full neural-policy proof requires policy=neural")
    require(report.get("completed_deliveries") == report.get("planned_deliveries") == 16,
            "Full proof must complete all 16 planned deliveries")
    for filename, key in (("scene.xml", "scene_sha256"), ("trajectory.npz", "trajectory_sha256")):
        require(report.get(key) == digest(directory / filename), f"Hash mismatch: {filename}")
    if not weights.is_file():
        errors.append(f"Missing exported policy: {weights}")
        return finish()
    require(report.get("weights_sha256") == digest(weights), "Exported policy hash mismatch")
    sources = report.get("source_files_sha256", {})
    expected_sources = {str(path.relative_to(ROOT)) for path in (ROOT / "arena_mujoco").glob("*.py")}
    expected_sources.update(("arena_spec.json", "competition_rules.json", "robot_design.json"))
    require(expected_sources.issubset(sources), "Source manifest omits required Python or specification files")
    for relative, sha in sources.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append(f"Missing or invalid source manifest path: {relative}")
        else:
            require(digest(path) == sha, f"Source hash mismatch: {relative}")
    checks["source_files_checked"] = len(sources)

    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        times, poses, initial = archive["times"], archive["qpos"], archive["initial_qpos"]
    if not require(times.ndim == 1 and poses.shape == (len(times), model.nq) and initial.shape == (model.nq,),
                   "Trajectory dimensions disagree with the scene"):
        return finish()
    if not require(len(times) > 1 and np.isfinite(times).all() and np.isfinite(poses).all()
                   and np.isfinite(initial).all() and np.all(np.diff(times) > 0),
                   "Trajectory must contain finite poses and strictly increasing timestamps"):
        return finish()
    data.qpos[:] = initial
    initial_metadata = copy.deepcopy(metadata)
    setup = validate_initial_setup(model, data, initial_metadata)
    require(setup["valid"], "Initial physical setup is invalid: " + "; ".join(setup["errors"]))
    require(metadata.get("competition_events", {}).get("initial_setup_valid") is True,
            "Saved event history lacks verified initial setup")
    checks["initial_setup_valid"] = setup["valid"]

    def load_state(filename):
        with np.load(directory / filename, allow_pickle=False) as archive:
            state = {key: archive[key].copy() for key in ("qpos", "qvel", "ctrl", "time")}
        if not all(state[key].shape == shape for key, shape in
                   (("qpos", (model.nq,)), ("qvel", (model.nv,)), ("ctrl", (model.nu,)))):
            raise ValueError(f"Wrong physics dimensions in {filename}")
        if state["time"].size != 1 or not all(np.isfinite(value).all() for value in state.values()):
            raise ValueError(f"Nonfinite or invalid physics state in {filename}")
        return state

    def restore(state):
        data.qpos[:] = state["qpos"]
        data.qvel[:] = state["qvel"]
        data.ctrl[:] = state["ctrl"]
        data.time = float(state["time"].item())

    declaration, final = load_state("declaration_state.npz"), load_state("final_state.npz")
    declaration_time, final_time = float(declaration["time"].item()), float(final["time"].item())
    events = metadata.get("competition_events", {})
    event_log = events.get("events", [])
    require(abs(float(events.get("last_observed_time_s", -1)) - final_time) < 1e-8,
            "Event history does not reach the saved final time")
    require(events.get("robot_outside_count") == sum(item.get("event") == "robot_fully_outside" for item in event_log),
            "Outside-event counter disagrees with event log")
    require(events.get("human_intervention") is any(item.get("event") == "human_intervention" for item in event_log),
            "Intervention flag disagrees with event log")
    restore(declaration)
    declared_score = score_competition(model, data, metadata)
    require(declared_score["success"], "Saved declaration geometry does not earn a valid 160-point feasibility result")
    for key in ("task_score", "score", "penalty_points", "success", "mode", "feasibility_success",
                "official_success", "official_score", "within_official_time", "inventory_valid",
                "robot_outside_count", "physics_warning_count", "sample_slot_assignments", "per_task"):
        require(report.get("score", {}).get(key) == declared_score[key], f"Declaration score mismatch: {key}")
    require(report.get("completion_seconds") is not None and
            abs(float(report["completion_seconds"]) - declaration_time) < 1e-8,
            "Completion time disagrees with saved declaration state")
    require(abs(float(report.get("score", {}).get("elapsed_seconds", -1)) - declaration_time) < 1e-8,
            "Reported score time disagrees with declaration state")
    restore(final)
    final_score = score_competition(model, data, metadata)
    require(final_score["success"], "Saved final geometry does not retain the full score")
    require(final_time - declaration_time >= 5 - 1e-8, "Final view is shorter than five seconds")
    require(abs(float(report.get("simulation_seconds", -1)) - final_time) < 1e-8,
            "Simulation time disagrees with final state")
    checks.update(declaration_task_score=declared_score["task_score"], final_task_score=final_score["task_score"],
                  declaration_time_s=declaration_time, final_time_s=final_time)

    view_checks = report.get("final_view_checks", [])
    require(report.get("final_view_valid") is True and len(view_checks) >= 10,
            "Missing successful final-view checks")
    view_times = np.asarray([item["time_s"] for item in view_checks], dtype=float)
    if len(view_times):
        require(np.isfinite(view_times).all() and np.all(np.diff(view_times) > 0)
                and view_times[0] > declaration_time and view_times[0] <= declaration_time + .50000001
                and np.max(np.diff(view_times), initial=0) <= .50000001
                and view_times[-1] >= declaration_time + 5 - 1e-8 and view_times[-1] <= final_time + 1e-8,
                "Final-view checks do not cover five seconds at half-second intervals")
    require(all(item.get("task_score") == 160 and item.get("success") is True for item in view_checks),
            "A final-view check failed")

    require(times[0] >= 0 and times[-1] <= final_time + 1e-8 and final_time - times[-1] <= .10000001,
            "Trajectory does not end near the saved final state")
    if abs(times[-1] - final_time) < 1e-8:
        require(np.allclose(poses[-1], final["qpos"], rtol=0, atol=1e-10),
                "Final trajectory pose differs from final state at the same timestamp")
    selected = (times > declaration_time + 1e-8) & (times < final_time - 1e-8)
    view_pose_times = np.concatenate(([declaration_time], times[selected], [final_time]))
    view_poses = np.concatenate((declaration["qpos"][None], poses[selected], final["qpos"][None]))
    require(np.max(np.diff(view_pose_times), initial=0) <= .10000001,
            "Trajectory has a gap over 100 ms during the final view")
    sampled_valid = True
    for i in range(1, len(view_poses)):
        data.qpos[:] = view_poses[i]
        mujoco.mj_differentiatePos(model, data.qvel, float(view_pose_times[i] - view_pose_times[i - 1]),
                                  view_poses[i - 1], view_poses[i])
        data.time = float(view_pose_times[i])
        sampled_valid &= bool(score_competition(model, data, metadata)["success"])
    require(sampled_valid, "Sampled final-view trajectory loses placement, release, or finite-difference stability")
    checks["final_view_trajectory_samples"] = len(view_poses)

    with np.load(directory / "policy_trace.npz", allow_pickle=False) as archive:
        observation, action, action_times = archive["observation"], archive["action"], archive["time_s"]
        limits, scales = archive["velocity_limits"], archive["error_scales"]
        obs_names, action_names = archive["observation_names"], archive["action_names"]
    if not require(observation.ndim == 2 and observation.shape[1] == 4 and action.shape == observation.shape
                   and action_times.shape == (len(action),), "Policy trace dimensions are invalid"):
        return finish()
    require(np.array_equal(limits, VELOCITY_LIMITS) and np.array_equal(scales, ERROR_SCALES),
            "Saved policy scales disagree with the verified controller")
    require(np.array_equal(obs_names, OBSERVATION_NAMES) and np.array_equal(action_names, ACTION_NAMES),
            "Policy trace channel names disagree with the verified controller")
    if not require(len(action) > 0 and np.isfinite(observation).all() and np.isfinite(action).all()
                   and np.isfinite(action_times).all(), "Policy trace contains no calls or nonfinite values"):
        return finish()
    require(np.all(np.abs(observation) <= 8 + 1e-12) and np.all(np.abs(action) <= 1 + 1e-12),
            "Policy trace exceeds normalized observation/action ranges")
    require(len(action) == report.get("policy_calls"), "Policy call count disagrees with trace length")
    require(np.allclose(np.diff(action_times), .02, rtol=0, atol=1e-8)
            and action_times[0] >= -1e-8 and action_times[0] <= .02000001
            and abs(action_times[-1] - final_time) <= .02000001,
            "Policy timestamps do not cover the run at 50 Hz")
    policy = NeuralPolicy(weights)
    if not require(all(np.isfinite(weight).all() for weight in policy.weights), "Exported policy contains nonfinite weights"):
        return finish()
    maximum_error = 0.
    for start in range(0, len(action), 4096):
        predicted = policy(observation[start:start + 4096])
        maximum_error = max(maximum_error, float(np.max(np.abs(predicted - action[start:start + 4096]))))
    require(maximum_error <= 1e-10, "Saved actions disagree with the exported neural network")
    checks.update(policy_calls_checked=len(action), maximum_network_action_error=maximum_error,
                  verification_scope="Artifact hashes, initial setup, neural actions, declaration score and sampled final-view stability")
    return finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    try:
        result = verify_proof(args.directory, weights=args.weights)
    except (OSError, ValueError, KeyError, TypeError, FloatingPointError) as error:
        result = {"valid": False, "physics_replayed": False, "errors": [str(error)]}
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
