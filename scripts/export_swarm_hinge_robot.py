#!/usr/bin/env python3
"""Export the hinge module's native geometry and articulated design contract."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_mujoco.swarm_hinge_robot import HINGE_DESIGN, add_hinge_robot


DEFAULT_OUT = ROOT / "output/swarm/hinge_robot"
JOINT_SAMPLES = 41
SPHERE_LONGITUDES = 48
SPHERE_LATITUDES = 24


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene(magnetic=False):
    root = ET.Element("mujoco", model="swarm_hinge_magnetic_module" if magnetic else "swarm_hinge_module")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=".001", integrator="implicitfast", cone="elliptic")
    robot = add_hinge_robot(root, 0, (0., 0., 0.))
    if magnetic:
        from arena_mujoco.swarm_hinge_scene import add_hinge_magnetic_docks
        add_hinge_magnetic_docks(root, [robot])
    return root, robot


def _box_mesh(size):
    sx, sy, sz = size
    vertices = np.array(list(itertools.product((-sx, sx), (-sy, sy), (-sz, sz))), dtype=float)
    index = {tuple(vertex): i for i, vertex in enumerate(vertices)}
    faces = []
    for axis in range(3):
        other = [value for value in range(3) if value != axis]
        for sign in (-1, 1):
            corners = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                point = [0., 0., 0.]
                point[axis] = sign * size[axis]
                point[other[0]] = a * size[other[0]]
                point[other[1]] = b * size[other[1]]
                corners.append(index[tuple(point)])
            if sign < 0:
                corners.reverse()
            faces.extend(((corners[0], corners[1], corners[2]),
                          (corners[0], corners[2], corners[3])))
    return vertices, np.asarray(faces, dtype=np.int32)


def _sphere_mesh(radius):
    vertices = [[0., 0., radius]]
    for latitude in range(1, SPHERE_LATITUDES):
        phi = math.pi * latitude / SPHERE_LATITUDES
        for longitude in range(SPHERE_LONGITUDES):
            theta = 2 * math.pi * longitude / SPHERE_LONGITUDES
            vertices.append([radius * math.sin(phi) * math.cos(theta),
                             radius * math.sin(phi) * math.sin(theta),
                             radius * math.cos(phi)])
    bottom = len(vertices)
    vertices.append([0., 0., -radius])
    faces = []
    first_ring = 1
    for longitude in range(SPHERE_LONGITUDES):
        following = (longitude + 1) % SPHERE_LONGITUDES
        faces.append((0, first_ring + longitude, first_ring + following))
    for latitude in range(SPHERE_LATITUDES - 2):
        first = 1 + latitude * SPHERE_LONGITUDES
        second = first + SPHERE_LONGITUDES
        for longitude in range(SPHERE_LONGITUDES):
            following = (longitude + 1) % SPHERE_LONGITUDES
            faces.extend(((first + longitude, second + longitude, second + following),
                          (first + longitude, second + following, first + following)))
    last_ring = 1 + (SPHERE_LATITUDES - 2) * SPHERE_LONGITUDES
    for longitude in range(SPHERE_LONGITUDES):
        following = (longitude + 1) % SPHERE_LONGITUDES
        faces.append((bottom, last_ring + following, last_ring + longitude))
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32)


def _geom_mesh(model, geom_id):
    kind = int(model.geom_type[geom_id])
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(model.geom_dataid[geom_id])
        vertex_start, vertex_count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        face_start, face_count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        return (np.array(model.mesh_vert[vertex_start:vertex_start + vertex_count], dtype=float),
                np.array(model.mesh_face[face_start:face_start + face_count], dtype=np.int32),
                "convex_mesh")
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        vertices, faces = _sphere_mesh(float(model.geom_size[geom_id, 0]))
        return vertices, faces, "sphere"
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        vertices, faces = _box_mesh(np.asarray(model.geom_size[geom_id], dtype=float))
        return vertices, faces, "box"
    raise ValueError(f"Unsupported hinge geometry type: {kind} ({model.geom(geom_id).name})")


def _transform(vertices, position, rotation):
    return vertices @ rotation.T + position


def _geom_world_mesh(model, data, geom_id):
    vertices, faces, shape = _geom_mesh(model, geom_id)
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return _transform(vertices, data.geom_xpos[geom_id], rotation), faces, shape


def _geom_world_bounds(model, data, geom_id):
    if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        radius = float(model.geom_size[geom_id, 0])
        return data.geom_xpos[geom_id] - radius, data.geom_xpos[geom_id] + radius
    vertices, _, _ = _geom_world_mesh(model, data, geom_id)
    return vertices.min(axis=0), vertices.max(axis=0)


def _merge_meshes(meshes):
    vertices, faces, offset = [], [], 0
    for mesh_vertices, mesh_faces in meshes:
        vertices.append(np.asarray(mesh_vertices, dtype=float))
        faces.append(np.asarray(mesh_faces, dtype=np.int32) + offset)
        offset += len(mesh_vertices)
    if not vertices:
        raise ValueError("Cannot export an empty articulated body")
    return np.concatenate(vertices), np.concatenate(faces)


def _write_stl(path, vertices_m, faces):
    vertices = np.asarray(vertices_m, dtype=np.float64) * 1000
    faces = np.asarray(faces, dtype=np.int32)
    with path.open("wb") as stream:
        stream.write(b"Swarm hinge native envelope".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(faces)))
        for face in faces:
            triangle = vertices[face]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            length = np.linalg.norm(normal)
            if length:
                normal /= length
            stream.write(struct.pack("<12fH", *normal.astype(np.float32),
                                     *triangle.astype(np.float32).ravel(), 0))


def _write_obj(path, vertices_m, faces, object_name):
    vertices = np.asarray(vertices_m) * 1000
    lines = ["# Native MuJoCo envelope; units: millimeters", f"o {object_name}"]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices)
    lines.extend("f " + " ".join(str(int(index) + 1) for index in face) for face in faces)
    path.write_text("\n".join(lines) + "\n")


def _bounds(vertices):
    vertices = np.asarray(vertices, dtype=float)
    return [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]


def _bounds_mm(vertices):
    return (np.asarray(_bounds(vertices)) * 1000).tolist()


def _body_joint_ids(model, body_id, root_id):
    result = []
    current = int(body_id)
    while current and current != int(model.body_parentid[root_id]):
        for joint_id in range(model.njnt):
            if (int(model.jnt_bodyid[joint_id]) == current
                    and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_HINGE)):
                result.append(joint_id)
        if current == root_id:
            break
        current = int(model.body_parentid[current])
    return sorted(result)


def _body_bounds(model, data, geom_ids):
    bounds = [_geom_world_bounds(model, data, geom_id) for geom_id in geom_ids]
    return np.min([item[0] for item in bounds], axis=0), np.max([item[1] for item in bounds], axis=0)


def _joint_reach(model, data, geom_ids, joint_id):
    anchor = data.xanchor[joint_id]
    reach = 0.
    for geom_id in geom_ids:
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            candidate = np.linalg.norm(data.geom_xpos[geom_id] - anchor) + model.geom_size[geom_id, 0]
        else:
            vertices, _, _ = _geom_world_mesh(model, data, geom_id)
            candidate = np.linalg.norm(vertices - anchor, axis=1).max()
        reach = max(reach, float(candidate))
    return reach


def _sample_body_envelope(model, data, geom_ids, joint_ids, samples):
    neutral = data.qpos.copy()
    grids = [np.linspace(*model.jnt_range[joint_id], samples) for joint_id in joint_ids]
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    pose_count = 0
    for angles in itertools.product(*grids) if grids else [()]:
        data.qpos[:] = neutral
        for joint_id, value in zip(joint_ids, angles):
            data.qpos[int(model.jnt_qposadr[joint_id])] = value
        mujoco.mj_forward(model, data)
        current_low, current_high = _body_bounds(model, data, geom_ids)
        low, high = np.minimum(low, current_low), np.maximum(high, current_high)
        pose_count += 1
    data.qpos[:] = neutral
    mujoco.mj_forward(model, data)
    margin = 0.
    for joint_id in joint_ids:
        step = float(np.ptp(model.jnt_range[joint_id]) / (samples - 1))
        margin += _joint_reach(model, data, geom_ids, joint_id) * step / 2
    return low - margin, high + margin, margin, pose_count


def _verify_body_envelope(model, data, geom_ids, joint_ids, low, high, samples):
    neutral = data.qpos.copy()
    grids = []
    for joint_id in joint_ids:
        edges = np.linspace(*model.jnt_range[joint_id], samples)
        grids.append((edges[:-1] + edges[1:]) / 2)
    exceedance = 0.
    pose_count = 0
    for angles in itertools.product(*grids) if grids else [()]:
        data.qpos[:] = neutral
        for joint_id, value in zip(joint_ids, angles):
            data.qpos[int(model.jnt_qposadr[joint_id])] = value
        mujoco.mj_forward(model, data)
        current_low, current_high = _body_bounds(model, data, geom_ids)
        exceedance = max(exceedance, float(np.max(low - current_low)), float(np.max(current_high - high)), 0.)
        pose_count += 1
    data.qpos[:] = neutral
    mujoco.mj_forward(model, data)
    return exceedance, pose_count


def _actuator_for_joint(model, joint_id):
    matches = [actuator_id for actuator_id in range(model.nu)
               if int(model.actuator_trntype[actuator_id]) == int(mujoco.mjtTrn.mjTRN_JOINT)
               and int(model.actuator_trnid[actuator_id, 0]) == joint_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one actuator for {model.joint(joint_id).name}, found {len(matches)}")
    return matches[0]


def _joint_record(model, joint_id):
    actuator_id = _actuator_for_joint(model, joint_id)
    return {
        "name": model.joint(joint_id).name,
        "body": model.body(int(model.jnt_bodyid[joint_id])).name,
        "axis_body": model.jnt_axis[joint_id].tolist(),
        "anchor_body_m": model.jnt_pos[joint_id].tolist(),
        "limits_rad": model.jnt_range[joint_id].tolist(),
        "armature_kg_m2": float(model.dof_armature[int(model.jnt_dofadr[joint_id])]),
        "actuator": {
            "name": model.actuator(actuator_id).name,
            "ctrlrange_nm": model.actuator_ctrlrange[actuator_id].tolist(),
            "forcerange_nm": model.actuator_forcerange[actuator_id].tolist(),
            "gear": model.actuator_gear[actuator_id].tolist(),
            "max_torque_nm": HINGE_DESIGN["max_motor_torque_nm"],
            "no_load_speed_rad_s": HINGE_DESIGN["motor_no_load_speed_rad_s"],
            "position_gain_nm_rad": HINGE_DESIGN["position_gain_nm_rad"],
            "velocity_gain_nm_s_rad": HINGE_DESIGN["velocity_gain_nm_s_rad"],
        },
    }


def export(out=None, magnetic=False):
    out = Path(out) if out is not None else DEFAULT_OUT
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    root, robot = _scene(magnetic)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root_id = model.body(robot["name"]).id

    geometry, body_records, assembly_meshes = [], [], []
    body_ids = sorted(set(int(model.geom_bodyid[model.geom(name).id]) for name in robot["collider_names"]))
    joint_ids = [joint_id for joint_id in range(model.njnt)
                 if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
    joint_records = [_joint_record(model, joint_id) for joint_id in joint_ids]
    joint_by_body = {}
    for record in joint_records:
        joint_by_body.setdefault(record["body"], []).append(record["name"])

    neutral_low, neutral_high = np.full(3, np.inf), np.full(3, -np.inf)
    swept_low, swept_high = np.full(3, np.inf), np.full(3, -np.inf)
    total_poses = verification_poses = 0
    body_pose_counts = {}
    maximum_margin = maximum_exceedance = 0.
    for body_id in body_ids:
        body_name = model.body(body_id).name
        geom_ids = [int(model.geom(name).id) for name in robot["collider_names"]
                    if int(model.geom_bodyid[model.geom(name).id]) == body_id]
        body_rotation = data.xmat[body_id].reshape(3, 3)
        local_meshes = []
        for geom_id in geom_ids:
            world_vertices, faces, shape = _geom_world_mesh(model, data, geom_id)
            local_vertices = (world_vertices - data.xpos[body_id]) @ body_rotation
            local_meshes.append((local_vertices, faces))
            local_low, local_high = np.asarray(_bounds(local_vertices))
            geometry.append({
                "name": model.geom(geom_id).name,
                "body": body_name,
                "shape": shape,
                "size_m": model.geom_size[geom_id].tolist(),
                "position_body_m": model.geom_pos[geom_id].tolist(),
                "orientation_body_quat_wxyz": model.geom_quat[geom_id].tolist(),
                "local_bounds_m": [local_low.tolist(), local_high.tolist()],
                "mesh_vertices": len(local_vertices),
                "mesh_triangles": len(faces),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "friction": model.geom_friction[geom_id].tolist(),
                "rgba": model.geom_rgba[geom_id].tolist(),
            })
            assembly_meshes.append((world_vertices, faces))
        local_vertices, local_faces = _merge_meshes(local_meshes)
        part_stl = parts / f"{body_name}_local_mm.stl"
        part_obj = parts / f"{body_name}_local_mm.obj"
        _write_stl(part_stl, local_vertices, local_faces)
        _write_obj(part_obj, local_vertices, local_faces, body_name)
        influence = _body_joint_ids(model, body_id, root_id)
        body_swept_low, body_swept_high, margin, poses = _sample_body_envelope(
            model, data, geom_ids, influence, JOINT_SAMPLES)
        exceedance, checked = _verify_body_envelope(
            model, data, geom_ids, influence, body_swept_low, body_swept_high, JOINT_SAMPLES)
        current_low, current_high = _body_bounds(model, data, geom_ids)
        neutral_low, neutral_high = np.minimum(neutral_low, current_low), np.maximum(neutral_high, current_high)
        swept_low, swept_high = np.minimum(swept_low, body_swept_low), np.maximum(swept_high, body_swept_high)
        total_poses += poses
        verification_poses += checked
        body_pose_counts[body_name] = checked
        maximum_margin = max(maximum_margin, margin)
        maximum_exceedance = max(maximum_exceedance, exceedance)
        parent_id = int(model.body_parentid[body_id])
        body_records.append({
            "name": body_name,
            "parent": model.body(parent_id).name,
            "position_parent_m": model.body_pos[body_id].tolist(),
            "orientation_parent_quat_wxyz": model.body_quat[body_id].tolist(),
            "mass_kg": float(model.body_mass[body_id]),
            "local_center_of_mass_m": model.body_ipos[body_id].tolist(),
            "local_inertia_kg_m2": model.body_inertia[body_id].tolist(),
            "geoms": [model.geom(geom_id).name for geom_id in geom_ids],
            "local_bounds_mm": _bounds_mm(local_vertices),
            "parts": {
                "stl": {"path": part_stl.relative_to(out).as_posix(), "sha256": _sha256(part_stl)},
                "obj": {"path": part_obj.relative_to(out).as_posix(), "sha256": _sha256(part_obj)},
            },
            "joints": joint_by_body.get(body_name, []),
            "swept_bounds_root_m": [body_swept_low.tolist(), body_swept_high.tolist()],
            "swept_conservative_margin_m": margin,
            "swept_pose_samples": poses,
        })

    assembly_vertices, assembly_faces = _merge_meshes(assembly_meshes)
    assembly_stl = out / "neutral_assembly_mm.stl"
    assembly_obj = out / "neutral_assembly_mm.obj"
    _write_stl(assembly_stl, assembly_vertices, assembly_faces)
    _write_obj(assembly_obj, assembly_vertices, assembly_faces, "swarm_hinge_neutral_assembly")
    xml_path = out / "robot.xml"
    xml_path.write_text(xml + "\n")
    geometry_path = out / "geometry.json"
    geometry_path.write_text(json.dumps({
        "schema_version": 1,
        "variant": "hinge_magnetic_module" if magnetic else "hinge_module",
        "units": "meters",
        "tessellation": {"sphere_longitudes": SPHERE_LONGITUDES,
                         "sphere_latitudes": SPHERE_LATITUDES},
        "robot": robot,
        "geoms": geometry,
    }, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "variant": "hinge_magnetic_module" if magnetic else "hinge_module",
        "units": "millimeters",
        "source": ["arena_mujoco/swarm_hinge_robot.py"]
                  + (["arena_mujoco/swarm_hinge_scene.py", "arena_mujoco/swarm_magnets.py"] if magnetic else []),
        "scope": "Compiled native collision and visual envelopes for articulated layout reference",
        "total_mass_kg": float(model.body_subtreemass[root_id]),
        "neutral_assembly": {
            "pose": {"joint_angles_rad": [0.] * len(joint_ids)},
            "bounds_mm": (np.asarray([neutral_low, neutral_high]) * 1000).tolist(),
            "stl": {"path": assembly_stl.relative_to(out).as_posix(), "sha256": _sha256(assembly_stl)},
            "obj": {"path": assembly_obj.relative_to(out).as_posix(), "sha256": _sha256(assembly_obj)},
        },
        "bodies": body_records,
        "joints": joint_records,
        "motor_assumptions": {key: HINGE_DESIGN[key] for key in (
            "physical_motor_count", "max_motor_torque_nm", "motor_no_load_speed_rad_s",
            "position_gain_nm_rad", "velocity_gain_nm_s_rad", "joint_armature_kg_m2")},
        "swept_envelope": {
            "method": ("Each body's influencing joints use a 41-point tensor grid. The sampled AABB is expanded "
                       "by sum(reach_from_joint_axis * half_grid_step) to contain angles between samples."),
            "joint_samples": {model.joint(joint_id).name: JOINT_SAMPLES for joint_id in joint_ids},
            "total_pose_samples": total_poses,
            "conservative_margin_m": maximum_margin,
            "bounds_root_m": [swept_low.tolist(), swept_high.tolist()],
            "verification": {
                "method": "Independent midpoint grid for every articulated body",
                "pose_samples": verification_poses,
                "body_pose_counts": body_pose_counts,
                "max_bound_exceedance_m": maximum_exceedance,
                "passed": maximum_exceedance <= 1e-12,
            },
        },
        "files": {
            "robot_xml": {"path": xml_path.name, "sha256": _sha256(xml_path)},
            "geometry": {"path": geometry_path.name, "sha256": _sha256(geometry_path)},
        },
        "engineering_remaining": [
            "Motor and transmission cavities", "Bearings and shafts", "Fasteners and assembly clearances",
            "PCB, battery and wiring", "Flexible stops and cable routing", "Thermal duty-cycle validation",
        ],
    }
    manifest_path = out / "cad_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def motion_coupon_1ms():
    from scripts.diagnose_swarm_hinge import Gait, run

    cases = {
        "forward": Gait(direction=-1),
        "reverse": Gait(direction=1),
        "left": Gait(direction=-1, steering_rad=-math.radians(9), steering_phase_rad=math.pi / 2),
        "right": Gait(direction=-1, steering_rad=math.radians(9), steering_phase_rad=math.pi / 2),
    }
    results = {name: run(gait, seconds=6., timestep=.001) for name, gait in cases.items()}
    left, right = results["left"], results["right"]
    passed = bool(
        results["forward"]["forward_mm"] > 10 and results["reverse"]["forward_mm"] < -10
        and left["forward_mm"] > 10 and right["forward_mm"] > 10
        and left["yaw_change_deg"] > 3 and right["yaw_change_deg"] < -3
        and abs(left["yaw_change_deg"] + right["yaw_change_deg"]) < 1e-5
        and abs(left["lateral_mm"] + right["lateral_mm"]) < 1e-5
        and all(result["external_applied_force_max"] == 0 and result["warnings"] == 0
                and result["peak_motor_torque_nm"] <= HINGE_DESIGN["max_motor_torque_nm"]
                and result["maximum_tilt_deg"] < 10
                for result in results.values()))
    return {"schema_version": 1, "timestep_s": .001, "duration_per_case_s": 6.,
            "passed": passed, "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output", dest="out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--magnetic", action="store_true", help="include four moving lobe docking housings")
    parser.add_argument("--validate", action="store_true", help="run four motor-only 1 ms motion coupons")
    args = parser.parse_args()
    out = export(args.out, args.magnetic)
    if args.validate:
        result = motion_coupon_1ms()
        validation_path = out / "motion_validation_1ms.json"
        validation_path.write_text(json.dumps(result, indent=2) + "\n")
        manifest_path = out / "cad_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["motion_validation_1ms"] = {
            "path": validation_path.name, "sha256": _sha256(validation_path)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if not result["passed"]:
            raise SystemExit("The 1 ms hinge motion coupon failed")
    print(out)


if __name__ == "__main__":
    main()
