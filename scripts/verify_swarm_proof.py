#!/usr/bin/env python3
"""Audit recorded swarm geometry, provenance, and every deployed neural action."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_mujoco.swarm_env import FEATURES, SwarmVectorEnv
from arena_mujoco.swarm_policy import NumpySwarmPolicy

ARTIFACTS = {"scene.xml", "metadata.json", "trajectory.npz", "policy_trace.npz"}
SOURCES = {"arena_mujoco/swarm_env.py", "arena_mujoco/swarm_robot.py", "arena_mujoco/swarm_policy.py",
           "arena_mujoco/builder.py", "arena_mujoco/materials.py", "arena_spec.json", "scripts/run_swarm.py"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message, errors):
    if not bool(condition):
        errors.append(message)
    return bool(condition)


def audit_timing(times, poses, action_times, actions, nq, robots, control_dt, errors):
    valid = require(times.ndim == 1 and poses.shape == (len(times), nq),
                    "Trajectory qpos dimensions disagree with the scene", errors)
    valid &= require(action_times.ndim == 1 and actions.shape == (len(action_times), robots, 3),
                     "Policy action dimensions disagree with the scene", errors)
    valid &= require(len(times) >= 2 and len(times) == len(action_times) + 1,
                     "Incomplete recording: every action must have its following physics frame", errors)
    valid &= require(all(np.isfinite(value).all() for value in (times, poses, action_times, actions)),
                     "Trajectory or policy trace contains nonfinite values", errors)
    if not valid:
        return False
    require(np.isfinite(control_dt) and control_dt > 0, "Invalid control timestep", errors)
    require(abs(times[0]) < 1e-8 and np.allclose(np.diff(times), control_dt, atol=1e-7, rtol=0),
            "Trajectory timestamps must start at zero and contain every control step", errors)
    require(np.allclose(action_times, times[:-1], atol=1e-8, rtol=0),
            "Action timestamps are not aligned with their input physics frames", errors)
    require(np.max(np.abs(actions)) <= 1 + 1e-6, "Normalized action exceeds actuator command bounds", errors)
    return True


def audit_neural_actions(policy, trace, errors, chunk_size=128):
    """Recompute all commands, including late transport and final-hold actions."""
    maximum = 0.
    count = len(trace["actions"])
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        obs = {key: trace[key][start:stop] for key in ("local", "neighbors", "neighbor_mask", "active")}
        actual = policy(obs)
        maximum = max(maximum, float(np.max(np.abs(actual - trace["actions"][start:stop]))))
    require(maximum <= 2e-6, f"Neural actions disagree with exported weights: max error {maximum:.9g}", errors)
    return {"neural_action_rows_recomputed": count, "neural_action_max_error": maximum}


def object_bottom(poses, model, objects):
    """Lowest collider point, accounting for recorded payload tilt."""
    positions, bottoms, half_heights = [], [], []
    for obj in objects:
        name = obj["name"]
        adr = model.joint(name + "_free").qposadr[0]
        pose = poses[:, adr:adr + 7]
        geom = model.geom(name + "_geom")
        if not np.allclose(geom.pos, 0) or not np.allclose(geom.quat, [1, 0, 0, 0]):
            raise ValueError(f"Unexpected offset collider for {name}")
        w, x, y, z = pose[:, 3:7].T
        row = np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], axis=-1)
        kind = int(geom.type[0])
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(row) @ geom.size
            half_height = geom.size[2]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = geom.size[0] * np.sqrt(np.maximum(0, 1-row[:, 2]**2)) + geom.size[1] * np.abs(row[:, 2])
            half_height = geom.size[1]
        else:
            raise ValueError(f"Unsupported payload collider: {name}")
        positions.append(pose[:, :3])
        bottoms.append(pose[:, 2] - extent)
        half_heights.append(float(half_height))
    return np.stack(positions, axis=1), np.stack(bottoms, axis=1), np.asarray(half_heights)


def audit_transport(times, positions, bottoms, pad_contacts, floor_contacts, phases, goals,
                    completion_time, minimum_distance, errors, half_heights=None):
    """Derive pickup, supported transport, release, and rest from saved states."""
    objects = positions.shape[1]
    both = pad_contacts[:, :objects*2].reshape(len(times), objects, 2).min(-1).astype(bool)
    half_heights = positions[0, :, 2] if half_heights is None else half_heights
    center_clearance = positions[..., 2] - half_heights
    supported = both & (center_clearance > .004) & (bottoms >= -1e-6) & ~floor_contacts.astype(bool)
    delta = np.diff(positions[..., :2], axis=0)
    step_distance = np.linalg.norm(delta, axis=-1)
    carried = (step_distance * supported[1:]).sum(0)
    axis = goals - positions[0, :, :2]
    axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    supported_progress = ((delta * axis).sum(-1) * supported[1:] * supported[:-1]).sum(0)
    displacement = np.linalg.norm(positions[-1, :, :2] - positions[0, :, :2], axis=-1)
    goal_error = np.linalg.norm(positions[..., :2] - goals, axis=-1)
    require(np.all(displacement >= minimum_distance), "A payload did not travel the required net distance", errors)
    require(np.all(supported_progress >= minimum_distance),
            "A payload lacks the required forward transport while physically lifted and gripped", errors)
    dt = float(np.median(np.diff(times)))
    support_steps = supported.sum(0)
    require(np.all(support_steps * dt >= .1), "A payload has no sustained two-pad airborne pickup", errors)
    require(np.all(np.abs(bottoms[0]) < .002), "Payloads must start resting on the field", errors)
    phase_delta = np.diff(phases, axis=0)
    retry = np.isin(phases[:-1], [1, 2]) & (phases[1:] == 0) & ~supported[1:]
    require(np.all(phases[0] == 0) and np.all((phase_delta == 0) | (phase_delta == 1) | retry),
            "Payload phase sequence contains an invalid transition", errors)
    require(np.all(phases[-1] == 5), "Recording ends before every payload is released", errors)
    require(np.all(goal_error[-1] < .012), "Final payload position is outside its delivery goal", errors)
    result = {"object_net_displacement_m": displacement.tolist(), "object_carried_distance_m": carried.tolist(),
              "object_supported_progress_m": supported_progress.tolist(),
              "object_supported_control_steps": support_steps.tolist(),
              "object_final_goal_error_m": goal_error[-1].tolist(),
              "object_min_airborne_bottom_clearance_m": [float(bottoms[supported[:, o], o].min())
                                                          if supported[:, o].any() else None for o in range(objects)]}
    if not require(isinstance(completion_time, (float, int)) and np.isfinite(completion_time),
                   "Recording has no finite completion timestamp", errors):
        return result
    matching = np.flatnonzero(np.abs(times - completion_time) < 1e-7)
    if not require(len(matching) == 1, "Completion timestamp is absent from the trajectory", errors):
        return result
    begin = int(matching[0])
    hold_seconds = float(times[-1] - times[begin])
    require(hold_seconds >= 5 - 1e-7, "Final hold is shorter than five seconds", errors)
    require(np.all(supported[:begin+1].any(axis=0)), "Completion precedes a physical pickup", errors)
    require(np.all(phases[begin:] == 5), "Final hold begins before every release phase completed", errors)
    require(np.all(goal_error[begin:] < .012), "Payload leaves its goal during the final hold", errors)
    require(np.all(np.abs(bottoms[begin:]) < .002), "Payload is airborne or penetrating the field during final hold", errors)
    require(not np.any(pad_contacts[begin:]), "Robot pad contact remains during final hold", errors)
    require(np.all(floor_contacts[begin:] == 1), "Payload lacks field support during final hold", errors)
    drift = np.linalg.norm(positions[begin:, :, :2] - positions[begin, :, :2], axis=-1).max(0)
    require(np.all(drift <= .002), "Payload drifts more than 2 mm during the final hold", errors)
    if begin < len(times) - 1:
        speed = np.linalg.norm(np.diff(positions[begin:, :, :2], axis=0), axis=-1) / np.diff(times[begin:])[:, None]
        require(np.all(speed < .012), "Payload remains in motion during final hold", errors)
    result.update(final_hold_s=hold_seconds, final_hold_max_drift_m=drift.tolist())
    return result


def audit_scene(model, metadata, errors):
    robots, objects = metadata.get("robots", []), metadata.get("objects", [])
    require(len(robots) == 40, "Scene must contain exactly 40 modules", errors)
    require(model.nu == 120, "Scene must contain exactly 120 robot actuators", errors)
    require(model.neq == 0, "Scene contains an equality constraint or object weld", errors)
    require(model.nmocap == 0, "Scene contains mocap-driven bodies", errors)
    require(int(np.sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)) == 40 + len(objects),
            "Every robot and payload must have its own free joint", errors)
    expected_actuators = set()
    for index, robot in enumerate(robots):
        name = f"swarm_{index:02d}"
        require(robot.get("name") == name, "Robot metadata order or names are invalid", errors)
        body, joint = model.body(name), model.joint(name + "_free")
        require(int(body.parentid[0]) == 0 and int(joint.type[0]) == mujoco.mjtJoint.mjJNT_FREE
                and int(joint.bodyid[0]) == body.id, f"{name} is not an independent free module", errors)
        require(abs(model.body_subtreemass[body.id] - .040) < 1e-7, f"{name} mass exceeds the declared 40 g model", errors)
        for side in ("left", "right"):
            motor = model.actuator(name + "_motor_" + side)
            wheel = model.joint(name + "_wheel_" + side)
            expected_actuators.add(motor.id)
            require(int(wheel.type[0]) == mujoco.mjtJoint.mjJNT_HINGE
                    and motor.trnid[0] == wheel.id and motor.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT,
                    f"{name} wheel motor does not drive its wheel hinge", errors)
            require(bool(motor.ctrllimited[0]) and bool(motor.forcelimited[0])
                    and np.allclose(motor.ctrlrange, [-.002, .002], atol=1e-12, rtol=0)
                    and np.allclose(motor.forcerange, [-.002, .002], atol=1e-12, rtol=0)
                    and np.allclose(motor.gear, [1, 0, 0, 0, 0, 0], atol=1e-12, rtol=0),
                    f"{name} wheel torque limit differs from 0.002 N m", errors)
        lift, motor = model.joint(name + "_lift"), model.actuator(name + "_lift_motor")
        expected_actuators.add(motor.id)
        require(int(lift.type[0]) == mujoco.mjtJoint.mjJNT_SLIDE and bool(lift.limited[0])
                and np.allclose(lift.range, [0, .008], atol=1e-12, rtol=0), f"{name} lift travel differs from 8 mm", errors)
        require(motor.trnid[0] == lift.id and motor.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT
                and bool(motor.ctrllimited[0]) and bool(motor.forcelimited[0])
                and np.allclose(motor.ctrlrange, [0, .008], atol=1e-12, rtol=0)
                and np.allclose(motor.forcerange, [-.5, .5], atol=1e-12, rtol=0)
                and np.allclose(motor.gear, [1, 0, 0, 0, 0, 0], atol=1e-12, rtol=0),
                f"{name} lift actuator violates its position or 0.5 N force limit", errors)
    require(expected_actuators == set(range(model.nu)), "Scene contains an actuator outside the robot motors", errors)
    for obj in objects:
        body, joint = model.body(obj["name"]), model.joint(obj["name"] + "_free")
        require(body.parentid[0] == 0 and joint.type[0] == mujoco.mjtJoint.mjJNT_FREE
                and joint.bodyid[0] == body.id, f"Payload {obj['name']} is constrained or parented to a robot", errors)


def audit_contacts(model, poses, metadata, pad_contacts, floor_contacts, errors):
    """Reconstruct contact geometry at every recorded pose, without integration."""
    robots, objects = metadata["robots"], metadata["objects"]
    robot_geom = np.full(model.ngeom, -1, dtype=int)
    pad_geom = np.full(model.ngeom, -1, dtype=int)
    object_geom = np.full(model.ngeom, -1, dtype=int)
    for index, robot in enumerate(robots):
        body_id = model.body(robot["name"]).id
        robot_geom[model.body_rootid[model.geom_bodyid] == body_id] = index
        for name in robot["pad_geoms"]:
            pad_geom[model.geom(name).id] = index
    for index, obj in enumerate(objects):
        object_geom[model.geom(obj["name"] + "_geom").id] = index
    data = mujoco.MjData(model)
    mismatches = 0
    collision_steps = 0
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        pads = np.zeros(len(robots), dtype=np.uint8)
        floor = np.zeros(len(objects), dtype=np.uint8)
        robot_collision = False
        for contact in data.contact:
            if contact.dist > .0003:
                continue
            a, b = contact.geom
            robot_collision |= robot_geom[a] >= 0 and robot_geom[b] >= 0 and robot_geom[a] != robot_geom[b]
            for first, second in ((a, b), (b, a)):
                pad, obj = pad_geom[first], object_geom[second]
                if pad >= 0 and obj == min(pad // 2, len(objects) - 1):
                    pads[pad] = 1
                obj = object_geom[first]
                if obj >= 0 and robot_geom[second] < 0 and object_geom[second] < 0:
                    floor[obj] = 1
        mismatches += int(not np.array_equal(pads, pad_contacts[frame]) or not np.array_equal(floor, floor_contacts[frame]))
        collision_steps += int(frame > 0 and robot_collision)
    require(mismatches == 0, f"Saved pad/floor contacts disagree with geometry in {mismatches} frames", errors)
    return {"contact_frames_reconstructed": len(poses), "contact_mismatch_frames": mismatches,
            "robot_collision_control_steps": collision_steps}


def audit_formation(times, poses, model, metadata, setup, report, errors):
    carrier_count = 2 * len(metadata["objects"])
    addresses = np.asarray([model.joint(robot["name"] + "_free").qposadr[0] for robot in metadata["robots"]])
    positions = poses[:, addresses[:, None] + np.arange(2)]
    travel = np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum(0)
    require(np.allclose(travel, report.get("robot_travel_m", []), atol=.002, rtol=0),
            "Reported robot travel disagrees with recorded poses", errors)
    checks = {"carrier_robots": carrier_count, "formation_robots": len(addresses)-carrier_count,
              "robot_travel_m": travel.tolist()}
    completion = report.get("completion_time_s")
    if carrier_count == len(addresses) or not isinstance(completion, (float, int)):
        return checks
    hold = times >= completion - 1e-7
    if not hold.any():
        return checks
    queue = positions[hold, carrier_count:]
    distance = np.linalg.norm(queue - setup["queue_targets"][carrier_count:], axis=-1)
    require(np.all(distance < .012), "Formation robot is outside its seeded queue target during final hold", errors)
    if len(queue) > 1:
        speed = np.linalg.norm(np.diff(queue, axis=0), axis=-1) / np.diff(times[hold])[:, None]
        require(np.all(speed < .012), "Formation robot remains in motion during final hold", errors)
    checks["formation_max_goal_error_m"] = float(distance.max())
    return checks


def audit_observation_geometry(trace, poses, phases, contacts, model, metadata, setup, errors, chunk_size=128):
    """Bind recorded actor inputs to poses, goals, contacts, and prior commands.

    Instantaneous velocities are not present in the trajectory file and cannot
    be reconstructed exactly from its position samples.
    """
    robots, objects = metadata["robots"], metadata["objects"]
    n, count = len(robots), len(trace["actions"])
    assignment = np.minimum(np.arange(n) // 2, len(objects) - 1)
    partner = np.arange(n) ^ 1
    carrier = np.arange(n) < len(objects)*2
    sign = np.where(np.arange(n) % 2 == 0, 1., -1.)
    rad = np.asarray([obj["radius"] for obj in objects])[assignment]
    half = np.asarray([obj["half_height"] for obj in objects])[assignment]
    axis = setup["axis"][assignment]
    goals = setup["goals"][assignment]
    radr = np.asarray([model.joint(robot["name"] + "_free").qposadr[0] for robot in robots])
    oadr = np.asarray([model.joint(obj["name"] + "_free").qposadr[0] for obj in objects])
    ladr = np.asarray([model.joint(robot["name"] + "_lift").qposadr[0] for robot in robots])
    maximum = 0.
    neighbor_error = 0.
    invalid_neighbor_sets = 0
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        qpos = poses[start:end]
        qr = qpos[:, radr[:, None] + np.arange(7)]
        qo = qpos[:, oadr[:, None] + np.arange(7)]
        w, x, y, z = np.moveaxis(qr[..., 3:7], -1, 0)
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        forward = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
        right = np.stack([np.cos(yaw), np.sin(yaw)], -1)
        phase = phases[start:end, assignment]
        obj = qo[:, assignment, :2]
        offset = rad + np.where(phase == 0, .0285, .0276)
        target = obj - sign[None, :, None] * axis * offset[..., None]
        retreat = obj - sign[None, :, None] * axis * (rad + .060)[None, :, None]
        target = np.where((phase >= 4)[..., None], retreat, target)
        target = np.where(carrier[None, :, None], target, setup["queue_targets"])
        wanted = setup["wanted_yaw"]
        def local(vector):
            return np.stack([(vector*right).sum(-1), (vector*forward).sum(-1)], -1)
        expected = np.zeros_like(trace["local"][start:end])
        expected[..., :2] = local(target - qr[..., :2]) / .15
        expected[..., 2:4] = np.stack([np.sin(wanted - yaw), np.cos(wanted-yaw)], -1)
        expected[..., 4:6] = local(obj - qr[..., :2]) / .1
        expected[..., 6:8] = local(goals - obj) / .4
        expected[..., 11] = qpos[:, ladr] / .008
        expected[..., 12] = contacts[start:end]
        expected[..., 13] = contacts[start:end, partner]
        expected[..., 14] = (qo[:, assignment, 2] - half) / .008
        expected[..., 15] = carrier
        expected[..., 16:22] = np.eye(6)[phase]
        previous = np.concatenate([np.tile([0., 0., -1.], (1, n, 1)), trace["actions"][:-1]])[start:end]
        expected[..., 22:25] = previous
        expected[..., 25:27] = local(axis)
        expected[..., 27] = rad / .03
        fields = list(range(8)) + list(range(11, 28))
        expected = np.clip(expected, -10, 10)
        maximum = max(maximum, float(np.max(np.abs(expected[..., fields] - trace["local"][start:end][..., fields]))))
        neighbors = trace["neighbors"][start:end]
        inferred_position = qr[:, :, None, :2] + .15 * (neighbors[..., :1] * right[:, :, None] + neighbors[..., 1:2] * forward[:, :, None])
        difference = inferred_position[..., None, :] - qr[:, None, None, :, :2]
        nearest = np.argmin(np.sum(difference*difference, axis=-1), axis=-1)
        frame_ids = np.arange(end-start)[:, None, None]
        matched = qr[frame_ids, nearest, :2]
        position_error = np.linalg.norm(inferred_position - matched, axis=-1)
        neighbor_error = max(neighbor_error, float(position_error.max()))
        same = (assignment[nearest] == assignment[None, :, None]) & carrier[None, :, None] & carrier[nearest]
        expected_direction = np.stack([(forward[frame_ids, nearest] * right[:, :, None]).sum(-1),
                                       (forward[frame_ids, nearest] * forward[:, :, None]).sum(-1)], -1)
        neighbor_error = max(neighbor_error, float(np.max(abs(neighbors[..., 4:6] - expected_direction))),
                             float(np.max(abs(neighbors[..., 6] - same))),
                             float(np.max(abs(neighbors[..., 7] - (nearest == partner[None, :, None])))))
        distance = np.linalg.norm(matched - qr[:, :, None, :2], axis=-1)
        expected_mask = (distance < .35) | same
        # Environment membership uses float32 squared distance; geometric
        # reconstruction uses float64 poses. Ties at the radius are ambiguous.
        boundary = (np.abs(distance - .35) < 1e-5) & ~same
        invalid_neighbor_sets += int(np.sum((trace["neighbor_mask"][start:end].astype(bool) != expected_mask) & ~boundary))
        invalid_neighbor_sets += int(np.sum(~np.any(nearest == partner[None, :, None], axis=-1)))
        invalid_neighbor_sets += int(np.sum(np.diff(np.sort(nearest, axis=-1), axis=-1) == 0))
        invalid_neighbor_sets += int(np.sum(nearest == np.arange(n)[None, :, None]))
    require(maximum < 2e-5, f"Recorded local observations disagree with physical poses/goals/actions: {maximum:.9g}", errors)
    require(neighbor_error < 2e-5 and invalid_neighbor_sets == 0,
            f"Recorded neighbor observations disagree with geometry: error {neighbor_error:.9g}, invalid entries {invalid_neighbor_sets}", errors)
    return {"observation_rows_geometry_checked": count, "local_geometry_max_error": maximum,
            "neighbor_geometry_max_error": neighbor_error,
            "observation_fields_without_exact_pose_reconstruction": [8, 9, 10, 28, 29, 30, 31]}


def verify_proof(directory, *, weights=None, training_report=None, allow_teacher=False, chunk_size=128):
    directory = Path(directory)
    errors, checks = [], {}
    result = {"valid": False, "proof_directory": str(directory.resolve()), "physics_replayed": False,
              "geometry_reconstructed": False, "final_neural_proof": False, "checks": checks, "errors": errors}
    required = ARTIFACTS | {"report.json"}
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        errors.append("Missing proof artifacts: " + ", ".join(missing))
        return result
    try:
        report = json.loads((directory / "report.json").read_text())
        metadata = json.loads((directory / "metadata.json").read_text())
        checks["report_sha256"] = digest(directory / "report.json")
        require(report.get("success") is True and report.get("failure") is None, "Report records an unsuccessful rollout", errors)
        require(report.get("task") == "cooperative_transport_benchmark" and report.get("official_competition_score", "missing") is None,
                "Proof must be identified as the transport benchmark with no official score", errors)
        policy_type = report.get("policy")
        require(policy_type == "neural" or (allow_teacher and policy_type == "teacher"),
                "Final proof requires a neural policy; teacher artifacts need --allow-teacher for pipeline tests", errors)
        result["evidence_kind"] = "teacher_pipeline_test" if policy_type == "teacher" else "neural_transport"
        require(report.get("simulator") == "native_MuJoCo", "Proof must be recorded in native MuJoCo", errors)
        robots, objects = report.get("robot_count"), report.get("object_count")
        if not require(robots == 40 and isinstance(objects, int) and 2 <= objects <= 20,
                       "Proof requires 40 robots and at least two simultaneous payloads", errors):
            return result
        require(report.get("completed_objects") == objects, "Report does not complete every payload", errors)
        require(len(metadata.get("objects", [])) == objects, "Object count disagrees with metadata", errors)
        require(metadata.get("task") == report.get("task") and metadata.get("rule_score") is False,
                "Metadata misidentifies the benchmark", errors)
        require({"cylinder", "kit", "sample"}.issubset({obj.get("kind") for obj in metadata["objects"]}),
                "Full proof must cover cylinder, medical kit, and sample geometries", errors)
        artifacts = report.get("artifacts", {})
        require(ARTIFACTS.issubset(artifacts), "Artifact hash manifest is incomplete", errors)
        for name, expected in artifacts.items():
            path = (directory / name).resolve()
            require(path.is_relative_to(directory.resolve()) and path.is_file() and digest(path) == expected,
                    f"Artifact hash mismatch: {name}", errors)
        source_hashes = report.get("source_hashes", {})
        require(SOURCES.issubset(source_hashes), "Source hash manifest is incomplete", errors)
        for name, expected in source_hashes.items():
            path = (ROOT / name).resolve()
            require(path.is_relative_to(ROOT) and path.is_file() and digest(path) == expected,
                    f"Source hash mismatch: {name}", errors)
        checks["source_files_checked"] = len(source_hashes)
        # Stop before loading arrays from artifacts whose identity is already wrong.
        if any("hash mismatch" in error or "manifest is incomplete" in error for error in errors):
            return result
        model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
        audit_scene(model, metadata, errors)
        require(abs(model.opt.timestep - report["physics_timestep_s"]) < 1e-12,
                "Reported physics timestep disagrees with scene", errors)
        require(report["physics_timestep_s"] in (.001, .002), "Unvalidated physics timestep", errors)
        require(report.get("control_timestep_s") == .02, "Unvalidated control timestep", errors)
        require(isinstance(report.get("episode_seconds"), (int, float)) and report["episode_seconds"] > 0,
                "Missing or invalid episode time budget", errors)
        require(isinstance(report.get("requested_hold_seconds"), (int, float)) and report["requested_hold_seconds"] >= 5,
                "Requested final hold is shorter than five seconds", errors)
        with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
            trajectory = {key: archive[key] for key in ("times", "qpos", "object_phases", "pad_contacts", "object_floor_contacts")}
        with np.load(directory / "policy_trace.npz", allow_pickle=False) as archive:
            trace = {key: archive[key] for key in ("times", "actions", "local", "neighbors", "neighbor_mask", "active")}
        times, poses = trajectory["times"], trajectory["qpos"]
        if not audit_timing(times, poses, trace["times"], trace["actions"], model.nq, robots,
                            report["control_timestep_s"], errors):
            return result
        frames, steps = len(times), len(trace["actions"])
        for key, shape in (("object_phases", (frames, objects)), ("pad_contacts", (frames, robots)),
                           ("object_floor_contacts", (frames, objects))):
            if not require(trajectory[key].shape == shape, f"Invalid trajectory shape: {key}", errors):
                return result
        k = min(6, robots - 1)
        for key, shape in (("local", (steps, robots, 32)), ("neighbors", (steps, robots, k, 8)),
                           ("neighbor_mask", (steps, robots, k)), ("active", (steps, robots))):
            if not require(trace[key].shape == shape and np.isfinite(trace[key]).all(), f"Invalid policy trace: {key}", errors):
                return result
        require(np.all(trace["active"] == 1), "Full proof must activate all 40 modules", errors)
        for key in ("pad_contacts", "object_floor_contacts"):
            require(np.isin(trajectory[key], [0, 1]).all(), f"Contact field is not binary: {key}", errors)
        phases = trajectory["object_phases"]
        if not require(np.isin(phases, np.arange(6)).all(), "Invalid payload phase values", errors):
            return result
        phases = phases.astype(int)
        require(report.get("local_feature_names") == FEATURES, "Policy feature names differ from the recorded source contract", errors)
        for joint_id in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            adr = model.jnt_qposadr[joint_id]
            require(np.all(abs(np.linalg.norm(poses[:, adr+3:adr+7], axis=-1) - 1) < 1e-5),
                    f"Invalid free-body quaternion at joint {joint_id}", errors)
        difficulty = float(report["difficulty"])
        require(0 <= difficulty <= 1, "Invalid curriculum difficulty", errors)
        if policy_type == "neural":
            require(difficulty == 1., "Final neural proof requires full curriculum difficulty", errors)
        # Recreate the two resets performed by run_swarm.py to bind goals and
        # initial positions to the saved seed and the hashed environment source.
        setup_env = SwarmVectorEnv(num_envs=1, num_robots=robots, num_objects=objects, seed=int(report["seed"]),
                                  timestep=float(report["physics_timestep_s"]),
                                  episode_seconds=float(report.get("episode_seconds", 60.)))
        setup_env.set_curriculum({"num_active_robots": robots, "num_active_objects": objects, "difficulty": difficulty})
        setup_env.reset()
        setup = {"goals": setup_env.goals[0].numpy().copy(), "axis": setup_env.axis[0].numpy().copy(),
                 "queue_targets": setup_env.queue_targets[0].numpy().copy(),
                 "wanted_yaw": setup_env._targets()[1][0].numpy().copy()}
        require(setup_env.xml == (directory / "scene.xml").read_text(), "Saved scene differs from the hashed scene builder", errors)
        require(np.allclose(setup_env.native_snapshot().qpos, poses[0], atol=1e-8, rtol=0),
                "Initial poses do not match the reported seed and curriculum", errors)
        require(setup_env.metadata["objects"] == metadata["objects"], "Payload metadata differs from the scene builder", errors)
        goals = np.asarray(report["goals_m"], dtype=float)
        if not require(goals.shape == (objects, 2) and np.isfinite(goals).all(), "Invalid delivery goals", errors):
            return result
        require(np.allclose(goals, setup["goals"], atol=1e-8, rtol=0), "Reported goals do not match the seed recipe", errors)
        setup_env.close()
        positions, bottoms, half_heights = object_bottom(poses, model, metadata["objects"])
        checks.update(audit_transport(times, positions, bottoms, trajectory["pad_contacts"], trajectory["object_floor_contacts"],
                                      phases, goals, report.get("completion_time_s"), .15 if difficulty == 1 else max(.02, .05+.15*difficulty-.025),
                                      errors, half_heights))
        require(report.get("final_hold_valid") is True, "Report contains a failed final hold", errors)
        if "final_hold_s" in checks:
            require(abs(float(report["final_hold_s"]) - checks["final_hold_s"]) < 1e-7, "Reported hold duration disagrees with timestamps", errors)
        require(np.allclose(report["object_carried_distance_m"], checks["object_carried_distance_m"], atol=.003, rtol=0),
                "Reported carried distance disagrees with geometry", errors)
        require(np.allclose(report["object_supported_control_steps"], checks["object_supported_control_steps"], atol=0, rtol=0),
                "Reported airborne support counts disagree with geometry", errors)
        checks.update(audit_contacts(model, poses, metadata, trajectory["pad_contacts"], trajectory["object_floor_contacts"], errors))
        require(report.get("collision_control_steps") == checks["robot_collision_control_steps"],
                "Reported collision count disagrees with reconstructed contacts", errors)
        checks.update(audit_formation(times, poses, model, metadata, setup, report, errors))
        result["geometry_reconstructed"] = True
        checks.update(audit_observation_geometry(trace, poses, phases, trajectory["pad_contacts"], model, metadata, setup, errors, chunk_size))
        if policy_type == "neural":
            weights = Path(weights) if weights else ROOT / "output/swarm/policy/weights.npz"
            training_report = Path(training_report) if training_report else weights.parent / "training.json"
            if not require(weights.is_file() and training_report.is_file(), "Missing neural weights or training report", errors):
                return result
            weight_hash = digest(weights)
            require(report.get("weights_sha256") == weight_hash, "Proof weights SHA-256 mismatch", errors)
            training = json.loads(training_report.read_text())
            checks["training_report_sha256"] = digest(training_report)
            require(training.get("weights_sha256") == weight_hash, "Training report weights SHA-256 mismatch", errors)
            require(training.get("acceptance_passed") is True, "Training acceptance gate did not pass", errors)
            require(any(name in str(training.get("gpu", "")) for name in ("A100", "H100"))
                    and training.get("cuda") is not None and training.get("backend") in ("warp", "native")
                    and training.get("cuda_optimization") is True and str(training.get("policy_device", "")).startswith("cuda"),
                    "Training report lacks A100/H100 CUDA optimization with a declared physics backend", errors)
            require(training.get("ppo_updates", 0) > 0 and training.get("cumulative_ppo_actor_update_l2", 0) > 1e-6
                    and training.get("final_stage") == 3, "Training report lacks completed full-stage PPO actor updates", errors)
            heldout = training.get("heldout", {})
            require(heldout.get("complete") is True and heldout.get("episodes", 0) >= 32
                    and heldout.get("success_rate", 0) >= .8 and heldout.get("success_wilson_lower_95", 0) >= .6,
                    "Training held-out results do not meet the acceptance thresholds", errors)
            require(heldout.get("success_rate", 0) > training.get("zero_baseline", {}).get("success_rate", 1) + .2,
                    "Training report lacks the required improvement over zero actions", errors)
            require(training.get("args", {}).get("robots") == 40 and training.get("args", {}).get("objects", 0) >= 2,
                    "Training configuration does not cover the full swarm task", errors)
            require(report.get("neural_calls") == steps, "Neural call count disagrees with the complete action trace", errors)
            policy = NumpySwarmPolicy(weights)
            require(training.get("config") == vars(policy.config), "Training and exported actor architectures disagree", errors)
            checks.update(audit_neural_actions(policy, trace, errors, chunk_size))
        else:
            require(report.get("neural_calls") == 0 and report.get("weights_sha256") is None,
                    "Teacher pipeline test falsely claims neural actions", errors)
        checks.update(trajectory_frames=frames, traced_actions=steps, robot_count=robots, object_count=objects,
                      free_bodies=40+objects, actuators=model.nu)
        result["valid"] = not errors
        result["final_neural_proof"] = not errors and policy_type == "neural"
    except (KeyError, ValueError, TypeError, IndexError, OSError, json.JSONDecodeError) as error:
        errors.append(f"Malformed proof artifact: {type(error).__name__}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "output/swarm/proof")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--allow-teacher", action="store_true", help="Accept teacher pipeline fixtures, never a final neural proof")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    result = verify_proof(args.directory, weights=args.weights, training_report=args.training_report,
                          allow_teacher=args.allow_teacher, chunk_size=args.chunk_size)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
