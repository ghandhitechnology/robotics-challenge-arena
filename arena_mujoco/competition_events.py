"""Initial-layout evidence and boundary-event history for preliminary runs."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import mujoco
import numpy as np

from .competition_scoring import _default_rules, _descendants, _geom_geometry, _kind


def _body_geoms(model, name):
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if root < 0:
        raise ValueError(f"Missing competition body: {name}")
    bodies = _descendants(model, {root})
    gids = [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in bodies
            and (model.geom_contype[gid] or model.geom_conaffinity[gid])]
    if not gids:
        raise ValueError(f"No collision geometry for competition body: {name}")
    return gids


def _robot_groups(model, metadata):
    entries = list(metadata.get("robots", []))
    if metadata.get("robot"):
        entries.append(metadata["robot"])
    if not entries:
        raise ValueError("Competition robot metadata is required")
    return {entry["name"]: _body_geoms(model, entry["name"]) for entry in entries}


def _bounds(model, data, gids):
    shapes = [_geom_geometry(model, data, gid) for gid in gids]
    return np.min([shape[0] for shape in shapes], axis=0), np.max([shape[1] for shape in shapes], axis=0)


def _inside(low, high, rectangle, tolerance=1e-9):
    return bool(np.all(low[:2] >= np.asarray(rectangle[:2]) - tolerance)
                and np.all(high[:2] <= np.asarray(rectangle[2:]) + tolerance))


def _robots_fully_outside(model, data, metadata):
    """Return each robot's complete collision AABB exit status.

    This conservative test includes wheels, raised links, and extended jaws.
    A diagonal shape whose AABB still touches the field is counted as inside.
    """
    mujoco.mj_forward(model, data)
    x0, y0, x1, y1 = _default_rules()["geometry"]["field_bounds_xy"]
    outside = {}
    for name, gids in _robot_groups(model, metadata).items():
        low, high = _bounds(model, data, gids)
        outside[name] = bool(high[0] < x0 or high[1] < y0 or low[0] > x1 or low[1] > y1)
    return outside


def robot_fully_outside(model, data, metadata):
    """Return whether any robot's complete collision AABB misses the field."""
    return any(_robots_fully_outside(model, data, metadata).values())


@lru_cache(maxsize=1)
def _expected_objects():
    spec = json.loads((Path(__file__).resolve().parents[1] / "arena_spec.json").read_text())
    placement = spec["placements"]
    result = {}
    for row_index, row in enumerate(placement["cylinder_rows"]):
        for column, x in enumerate(placement["cylinder_x"]):
            name = f"Cylinder_{row['color'].title()}_{row_index * 2 + column + 1:02d}"
            result[name] = (f"{row['color']}_cylinder", [x * .001, row["y"] * .001])
    settings = spec["configurations"]["senior_preliminary"]
    slots = [[x * .001, placement["kit_y"][row - 1] * .001]
             for row in settings["kit_rows"] for x in placement["kit_x"]]
    for i, xy in enumerate(slots[:settings["kits"]], 1):
        result[f"Medical_Kit_{i:02d}"] = ("medical_kit", xy)
    for i, xy in enumerate(placement["samples"][:settings["samples"]], 1):
        result[f"Biological_Sample_{i:02d}"] = ("sample", np.asarray(xy) * .001)
    return result


def validate_initial_setup(model, data, metadata):
    """Check the time-zero physical layout and return JSON-safe evidence.

    Unloaded pieces retain the reconstructed diagram positions. A moved kit
    counts as a preload only when supported by actual upward robot contact,
    above the floor, and wholly inside start. Tolerances are simulation checks.
    """
    tolerances = {"position_m": 1e-6, "floor_height_m": .0001,
                  "penetration_m": 1e-5, "support_gap_m": .0001,
                  "initial_speed": 1e-8, "upright_cosine": .999999}
    errors, evidence, preloads = [], {}, []
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        return {"valid": False, "errors": ["Nonfinite initial physics state"], "objects": {}}
    if abs(float(data.time)) > 1e-9:
        errors.append("Initial setup must be checked at simulation time zero")
    if np.max(np.abs(data.qvel), initial=0) > tolerances["initial_speed"]:
        errors.append("Initial generalized velocities must be zero")
    mujoco.mj_forward(model, data)
    start = _default_rules()["geometry"]["zones"]["start"]["bounds_xy"]
    robot_groups = _robot_groups(model, metadata)
    robot_gids = {gid for gids in robot_groups.values() for gid in gids}
    robot_bounds = {}
    for name, gids in robot_groups.items():
        low, high = _bounds(model, data, gids)
        robot_bounds[name] = [low.tolist(), high.tolist()]
        if not _inside(low, high, start):
            errors.append(f"Robot collision envelope exceeds start: {name}")
        if low[2] < -tolerances["penetration_m"]:
            errors.append(f"Robot penetrates the floor: {name}")
    expected = _expected_objects()
    names = [entry["name"] for entry in metadata["objects"]]
    if len(names) != len(set(names)) or set(names) != set(expected):
        errors.append("Object inventory differs from the senior preliminary layout")
    object_gids = {}
    for entry in metadata["objects"]:
        name, kind = entry["name"], _kind(entry)
        gids = _body_geoms(model, name)
        object_gids[name] = gids
        low, high = _bounds(model, data, gids)
        body = model.body(name).id
        nominal = expected.get(name)
        on_mark = bool(nominal and kind == nominal[0] and
                       np.linalg.norm(data.xpos[body, :2] - nominal[1]) <= tolerances["position_m"])
        upright = data.xmat[body].reshape(3, 3)[2, 2] >= tolerances["upright_cosine"]
        on_floor = abs(low[2]) <= tolerances["floor_height_m"]
        supported = False
        # A real contact under the kit is needed; an arbitrary elevated qpos
        # or a weld equality alone cannot manufacture preload evidence.
        for contact in data.contact:
            first, second = map(int, contact.geom)
            if contact.dist > tolerances["support_gap_m"]:
                continue
            if first in robot_gids and second in gids and contact.frame[2] > .5:
                supported = True
            if second in robot_gids and first in gids and contact.frame[2] < -.5:
                supported = True
        preload = bool(kind == "medical_kit" and supported and low[2] > tolerances["floor_height_m"]
                       and _inside(low, high, start))
        if preload:
            preloads.append(name)
        elif not (on_mark and upright and on_floor):
            errors.append(f"Object is neither at its diagram position nor a supported kit preload: {name}")
        evidence[name] = {"kind": kind, "on_diagram_mark": on_mark, "upright": bool(upright),
                          "on_floor": bool(on_floor), "preloaded": preload,
                          "bounds_xyz": [low.tolist(), high.tolist()]}
    # Signed distances also inspect disabled pairs. Native contacts cover
    # coincident boxes for which mj_geomDistance can return zero.
    contact_distances = {}
    for contact in data.contact:
        pair = tuple(sorted(map(int, contact.geom)))
        contact_distances[pair] = min(contact_distances.get(pair, 0.0), float(contact.dist))
    overlaps = []
    groups = list(object_gids.items())
    for i, (name, gids) in enumerate(groups):
        others = [("robot", robot_gids)] + groups[i + 1:]
        for other_name, other_gids in others:
            minimum = min(min(mujoco.mj_geomDistance(model, data, first, second, .001, None),
                              contact_distances.get(tuple(sorted((first, second))), .001))
                          for first in gids for second in other_gids)
            if minimum < -tolerances["penetration_m"]:
                overlaps.append({"first": name, "second": other_name, "penetration_m": -float(minimum)})
    if overlaps:
        errors.append("Initial objects overlap another object or the robot")
    return {"valid": not errors, "errors": errors, "robot_bounds_xyz": robot_bounds,
            "start_bounds_xy": start, "objects": evidence, "preloaded_kits": preloads,
            "overlaps": overlaps, "simulation_tolerances": tolerances,
            "layout_source": "arena_spec.json senior_preliminary reconstructed diagram positions"}


def initialize_competition_events(model, data, metadata):
    """Record verified initial setup before starting or resetting a match."""
    if abs(float(data.time)) > 1e-9:
        raise ValueError("Competition event history can only start at time zero")
    setup = validate_initial_setup(model, data, metadata)
    metadata["competition_initial_setup"] = setup
    outside = _robots_fully_outside(model, data, metadata)
    metadata["competition_events"] = {"robot_outside_count": 0, "human_intervention": False,
                                      "initial_setup_valid": setup["valid"],
                                      "robot_was_fully_outside": any(outside.values()),
                                      "robots_were_fully_outside": outside,
                                      "last_observed_time_s": 0.0, "events": []}
    return setup


def update_competition_events(model, data, metadata, *, human_intervention=False):
    """Call after each control step; count each robot's inside-to-outside transitions."""
    events = metadata["competition_events"]
    if "robots_were_fully_outside" not in events:
        raise ValueError("Initialize competition event history before stepping")
    now = float(data.time)
    if now < events["last_observed_time_s"]:
        raise ValueError("Competition time moved backwards; initialize a new run")
    outside = _robots_fully_outside(model, data, metadata)
    for name, is_outside in outside.items():
        if is_outside and not events["robots_were_fully_outside"][name]:
            events["robot_outside_count"] += 1
            events["events"].append({"event": "robot_fully_outside", "robot": name, "time_s": now})
    if human_intervention and not events["human_intervention"]:
        events["human_intervention"] = True
        events["events"].append({"event": "human_intervention", "time_s": now})
    events["robot_was_fully_outside"] = any(outside.values())
    events["robots_were_fully_outside"] = outside
    events["last_observed_time_s"] = now
    return events
