"""Geometric end-state scoring for the Korean senior preliminary round."""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path

import mujoco
import numpy as np


@lru_cache(maxsize=1)
def _default_rules():
    return json.loads((Path(__file__).resolve().parents[1] / "competition_rules.json").read_text())


def _ellipse_radius(offset, first, second):
    """Maximum distance of an offset ellipse from the origin, via its quartic."""
    a = 2 * np.dot(offset, first)
    b = 2 * np.dot(offset, second)
    c = (np.dot(first, first) - np.dot(second, second)) / 2
    d = np.dot(first, second)
    coefficients = np.array([-b + 2*d, -2*a + 8*c, -12*d, -2*a - 8*c, b + 2*d])
    scale = np.max(np.abs(coefficients))
    angles = [0.0, math.pi]
    if scale > 1e-25:
        coefficients /= scale
        coefficients = np.trim_zeros(coefficients, "f")
        for root in np.roots(coefficients):
            if abs(root.imag) < 1e-7:
                angles.append(2 * math.atan(float(root.real)))
    points = [offset + first * math.cos(t) + second * math.sin(t) for t in angles]
    return max(float(np.linalg.norm(point)) for point in points)


def _geom_geometry(model, data, gid):
    """Return exact world AABB and a projected-circle support evaluator."""
    kind = int(model.geom_type[gid])
    position = data.geom_xpos[gid].copy()
    rotation = data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        vertices = (corners * size) @ rotation.T + position
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[gid])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start:start + count] @ rotation.T + position
    else:
        vertices = None
    if vertices is not None:
        return vertices.min(axis=0), vertices.max(axis=0), lambda center: float(
            np.linalg.norm(vertices[:, :2] - center, axis=1).max())
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = np.full(3, size[0])
        radius = lambda center: float(np.linalg.norm(position[:2] - center) + size[0])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        axis = rotation[:, 2]
        extent = size[0] * np.sqrt(np.maximum(0, 1 - axis**2)) + size[1] * np.abs(axis)

        def radius(center):
            return max(_ellipse_radius(position[:2] - center + sign * size[1] * axis[:2],
                                       size[0] * rotation[:2, 0], size[0] * rotation[:2, 1])
                       for sign in (-1, 1))
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = rotation[:, 2]
        extent = size[0] + size[1] * np.abs(axis)
        radius = lambda center: max(float(np.linalg.norm(
            position[:2] - center + sign * size[1] * axis[:2]) + size[0]) for sign in (-1, 1))
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        transform = rotation * size
        extent = np.linalg.norm(transform, axis=1)
        directions, lengths, _ = np.linalg.svd(transform[:2], full_matrices=False)
        radius = lambda center: _ellipse_radius(position[:2] - center,
                                                directions[:, 0] * lengths[0],
                                                directions[:, 1] * lengths[1])
    else:
        raise ValueError(f"Unsupported scoring geom type {kind} on {model.geom(gid).name}")
    return position - extent, position + extent, radius


def _descendants(model, roots):
    result = set(roots)
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in result:
            result.add(body)
    return result


def _kind(entry):
    category, name = entry.get("category", ""), entry["name"].lower()
    if category == "sample" or name.startswith("biological_sample"):
        return "sample"
    if category == "medical_kit" or name.startswith("medical_kit"):
        return "medical_kit"
    color = entry.get("color") or {"patient": "red", "observation": "yellow", "low_risk": "green"}.get(category)
    for candidate in ("red", "yellow", "green"):
        if color == candidate or name.startswith(f"cylinder_{candidate}"):
            return f"{candidate}_cylinder"
    return "other"


def score_competition(model, data, metadata, time_limit_s=None, *, rules=None):
    """Score current geometry; call at deliveries/end, not every physics step.

    None selects unlimited feasibility; 120 selects official-time evaluation.
    The caller records historical exits, intervention, and initial setup validity
    in metadata['competition_events']. Missing history prevents a success claim.
    mj_forward refreshes poses and contacts without advancing simulation time.
    Stability/seating tolerances below are explicit simulation criteria, not
    tolerances published by the competition organizer.
    """
    rules = _default_rules() if rules is None else rules
    if isinstance(rules, (str, Path)):
        rules = json.loads(Path(rules).read_text())
    profile = rules["profiles"]["national_senior_preliminary"]
    geometry = rules["geometry"]
    if time_limit_s is not None and (not math.isfinite(time_limit_s) or time_limit_s <= 0):
        raise ValueError("time_limit_s must be positive and finite, or None")
    official_limit = float(profile["official_time_limit_s"])
    mode = ("unlimited_feasibility" if time_limit_s is None else
            "official_120s" if time_limit_s == official_limit else "limited_feasibility")
    tolerances = {"boundary_epsilon_m": 1e-9, "release_contact_distance_m": 1e-6,
                  "linear_speed_m_s": 0.005, "angular_speed_rad_s": 0.1,
                  "sample_seating_height_m": 0.0005}
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise FloatingPointError("Cannot score nonfinite physics state")
    mujoco.mj_forward(model, data)
    robot_metadata = list(metadata.get("robots", []))
    if metadata.get("robot"):
        robot_metadata.append(metadata["robot"])
    robot_roots = set()
    for robot in robot_metadata:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot["name"])
        if body < 0:
            raise ValueError(f"Robot scoring body is missing: {robot['name']}")
        robot_roots.add(body)
    robot_bodies = _descendants(model, robot_roots)
    robot_geoms = {gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in robot_bodies
                   and (model.geom_contype[gid] or model.geom_conaffinity[gid])}
    touched = set()
    for contact in data.contact:
        first, second = map(int, contact.geom)
        if contact.dist <= tolerances["release_contact_distance_m"]:
            if first in robot_geoms and second >= 0:
                touched.add(int(model.geom_bodyid[second]))
            if second in robot_geoms and first >= 0:
                touched.add(int(model.geom_bodyid[first]))
    attached = set()
    for equality in range(model.neq):
        if data.eq_active[equality] and model.eq_type[equality] in (
                mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
            first, second = int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])
            if first in robot_bodies:
                attached.add(second)
            if second in robot_bodies:
                attached.add(first)
    zones = geometry["zones"]
    destination_ids = list(profile["wrong_color_rule"]["zone_accepted_colors"])
    objects, inventory = {}, {}
    for entry in metadata["objects"]:
        name, kind = entry["name"], _kind(entry)
        if name in objects:
            raise ValueError(f"Duplicate scoring object {name}")
        inventory[kind] = inventory.get(kind, 0) + 1
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            raise ValueError(f"Scoring object body is missing: {name}")
        bodies = _descendants(model, {body})
        gids = [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in bodies
                and (model.geom_contype[gid] or model.geom_conaffinity[gid])]
        if not gids:
            raise ValueError(f"Scoring object has no collision geometry: {name}")
        shapes = [_geom_geometry(model, data, gid) for gid in gids]
        minimum = np.min([shape[0] for shape in shapes], axis=0)
        maximum = np.max([shape[1] for shape in shapes], axis=0)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        linear, angular = float(np.linalg.norm(velocity[3:])), float(np.linalg.norm(velocity[:3]))
        released = not bool(bodies & (touched | attached | robot_bodies))
        stable = linear <= tolerances["linear_speed_m_s"] and angular <= tolerances["angular_speed_rad_s"]
        clearances = {}
        for zid in destination_ids:
            x0, y0, x1, y1 = zones[zid]["bounds_xy"]
            clearances[zid] = float(min(minimum[0] - x0, minimum[1] - y0,
                                       x1 - maximum[0], y1 - maximum[1]))
        contained = [zid for zid, clearance in clearances.items() if clearance > tolerances["boundary_epsilon_m"]]
        evidence = {"kind": kind, "released": released, "stable": stable,
                    "linear_speed_m_s": linear, "angular_speed_rad_s": angular,
                    "bounds_xyz": [minimum.tolist(), maximum.tolist()],
                    "contained_zones": contained, "zone_clearance_m": clearances,
                    "eligible_zones": contained if released and stable else []}
        if kind == "sample":
            lab = geometry["laboratory"]
            slot_clearances = {slot["id"]: float(lab["slot_radius_m"] - max(
                shape[2](np.asarray(slot["center_xy"])) for shape in shapes)) for slot in lab["slots"]}
            height = tolerances["sample_seating_height_m"]
            seated = bool(-height <= minimum[2] <= height
                          and maximum[2] <= lab["sample_thickness_m"] + height)
            evidence.update(slot_clearance_m=slot_clearances, seated=seated,
                            eligible_slots=[sid for sid, clearance in slot_clearances.items()
                                            if clearance > tolerances["boundary_epsilon_m"]
                                            and released and stable and seated])
        objects[name] = evidence
    # Maximum one-to-one assignment also works if a custom slot layout overlaps.
    assigned = {}

    def assign(name, visited):
        for slot in objects[name].get("eligible_slots", []):
            if slot in visited:
                continue
            visited.add(slot)
            if slot not in assigned or assign(assigned[slot], visited):
                assigned[slot] = name
                return True
        return False

    for name, item in objects.items():
        if item["kind"] == "sample":
            assign(name, set())
    contaminated = {}
    for zid in destination_ids:
        accepted = profile["wrong_color_rule"]["zone_accepted_colors"][zid]
        wrong = [name for name, item in objects.items()
                 if item["kind"].endswith("_cylinder") and item["kind"].removesuffix("_cylinder") not in accepted
                 and zid in item["contained_zones"]]
        if wrong:
            contaminated[zid] = wrong

    def eligible(kind, zid):
        return [name for name, item in objects.items()
                if item["kind"] == kind and zid in item["eligible_zones"]]

    tasks = {task["id"]: task for task in profile["tasks"]}
    kit_counts = {zid: len(eligible("medical_kit", zid)) for zid in tasks["kits"]["full_score_distribution"]}
    kit_scored = sum(min(count, kit_counts[zid]) for zid, count in tasks["kits"]["full_score_distribution"].items())
    red = 0 if "hospital" in contaminated else len(eligible("red_cylinder", "hospital"))
    green = 0 if "recovery" in contaminated else len(eligible("green_cylinder", "recovery"))
    yellow_raw = {zid: len(eligible("yellow_cylinder", zid)) for zid in ("pcc_lower", "pcc_upper")}
    yellow_counts = {zid: 0 if zid in contaminated else count for zid, count in yellow_raw.items()}
    yellow_distribution = all(count >= tasks["yellow_patients"]["minimum_per_pcc"] for count in yellow_raw.values())
    yellow = sum(yellow_counts.values()) if yellow_distribution else 0
    counts = {"samples": len(assigned), "kits": kit_scored, "red_patients": red,
              "yellow_patients": yellow, "green_patients": green}
    per_task = {tid: {"scored_count": min(counts[tid], task["count"]),
                      "required_count": task["count"],
                      "points": min(counts[tid], task["count"]) * task["points_each"],
                      "maximum_points": task["maximum_points"]} for tid, task in tasks.items()}
    task_score = sum(task["points"] for task in per_task.values())
    field = geometry["field_bounds_xy"]
    currently_outside = False
    for root in robot_roots:
        bodies = _descendants(model, {root})
        shapes = [_geom_geometry(model, data, gid) for gid in robot_geoms if int(model.geom_bodyid[gid]) in bodies]
        if shapes:
            low, high = np.min([s[0] for s in shapes], axis=0), np.max([s[1] for s in shapes], axis=0)
            currently_outside |= bool(high[0] < field[0] or high[1] < field[1]
                                      or low[0] > field[2] or low[1] > field[3])
    events = metadata.get("competition_events", {})
    history_complete = all(key in events for key in ("robot_outside_count", "human_intervention", "initial_setup_valid"))
    outside_count = events.get("robot_outside_count", int(currently_outside))
    if isinstance(outside_count, bool) or int(outside_count) != outside_count or outside_count < 0:
        raise ValueError("competition_events.robot_outside_count must be a nonnegative integer")
    # Do not silently accept a stale event counter while a robot is outside now.
    outside_count = max(int(outside_count), int(currently_outside))
    penalties = outside_count * profile["penalties"]["robot_completely_outside_field_points_per_occurrence"]
    inventory_valid = all(inventory.get(kind, 0) == count for kind, count in profile["initial_inventory"].items())
    inventory_valid &= not bool(inventory.get("other", 0))
    setup_valid = events.get("initial_setup_valid") is True
    intervention = bool(events.get("human_intervention", False))
    elapsed = float(data.time)
    within_limit = time_limit_s is None or elapsed <= time_limit_s + 1e-9
    within_official = elapsed <= official_limit + 1e-9
    all_tasks = task_score == profile["maximum_task_score"]
    physics_warnings = int(data.warning.number.sum())
    valid_run = history_complete and setup_valid and inventory_valid and not intervention and physics_warnings == 0
    full_placement = all_tasks and not contaminated
    feasibility_success = bool(full_placement and valid_run and penalties == 0 and within_limit)
    official_success = bool(feasibility_success and within_official and mode == "official_120s")
    return {"profile": "national_senior_preliminary", "mode": mode, "elapsed_seconds": elapsed,
            "requested_time_limit_s": time_limit_s, "official_time_limit_s": official_limit,
            "within_requested_time_limit": bool(within_limit), "within_official_time": bool(within_official),
            "task_score": task_score, "penalty_points": penalties, "score": task_score + penalties,
            "maximum_score": profile["maximum_task_score"], "per_task": per_task,
            "all_tasks_complete": bool(all_tasks), "feasibility_success": feasibility_success,
            "official_success": official_success,
            "success": official_success if mode == "official_120s" else feasibility_success,
            "official_score": task_score + penalties if mode == "official_120s" and within_official and valid_run else None,
            "sample_slot_assignments": assigned, "kit_counts_by_zone": kit_counts,
            "yellow_counts_by_pcc": yellow_raw, "yellow_distribution_satisfied": yellow_distribution,
            "contaminated_zones": contaminated, "objects": objects,
            "inventory": inventory, "inventory_valid": bool(inventory_valid),
            "event_history_complete": history_complete, "initial_setup_valid": setup_valid,
            "human_intervention": intervention, "robot_currently_outside": currently_outside,
            "physics_warning_count": physics_warnings,
            "robot_outside_count": outside_count, "simulation_tolerances": tolerances,
            "scoring_interpretations": [
                "Samples use distinct slots and must be seated near the hole floor.",
                "Wrong-color contamination is evaluated from full containment, even if the wrong item is moving or held.",
                "Yellow distribution uses eligible placements before zone contamination; only uncontaminated PCC patient points remain.",
                "Stability is instantaneous; a held final view requires a separate time-history check."]}
