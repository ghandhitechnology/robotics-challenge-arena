#!/usr/bin/env python3
"""Check the hinge module's simulation-derived CAD export."""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_mujoco.swarm_hinge_robot import HINGE_DESIGN
from scripts.export_swarm_hinge_robot import export


def _obj_mesh(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields[:1] == ["v"]:
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[:1] == ["f"]:
            face = []
            for value in fields[1:]:
                index = int(value.split("/", 1)[0])
                face.append(index - 1 if index > 0 else len(vertices) + index)
            faces.append(face)
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not faces:
        raise AssertionError(f"OBJ has no usable mesh: {path}")
    if not np.isfinite(points).all():
        raise AssertionError(f"OBJ has a non-finite vertex: {path}")
    if any(len(face) < 3 or min(face) < 0 or max(face) >= len(points) for face in faces):
        raise AssertionError(f"OBJ has an invalid face: {path}")
    return points, faces


def _stl_mesh(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    raw = path.read_bytes()
    triangles: list[list[list[float]]] = []
    if len(raw) >= 84 and len(raw) == 84 + 50 * struct.unpack_from("<I", raw, 80)[0]:
        for record in struct.iter_unpack("<12fH", raw[84:]):
            triangles.append([list(record[start:start + 3]) for start in (3, 6, 9)])
    else:
        vertices = []
        for line in raw.decode("ascii").splitlines():
            fields = line.split()
            if fields[:1] == ["vertex"] and len(fields) == 4:
                vertices.append([float(value) for value in fields[1:]])
        if len(vertices) % 3:
            raise AssertionError(f"ASCII STL has an incomplete triangle: {path}")
        triangles = [vertices[start:start + 3] for start in range(0, len(vertices), 3)]
    points = np.asarray(triangles, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (3, 3) or not len(points):
        raise AssertionError(f"STL has no usable triangles: {path}")
    if not np.isfinite(points).all():
        raise AssertionError(f"STL has a non-finite vertex: {path}")
    flat = points.reshape(-1, 3)
    return flat, [[index, index + 1, index + 2] for index in range(0, len(flat), 3)]


def _assert_closed_mesh(test: unittest.TestCase, vertices: np.ndarray,
                        faces: list[list[int]], label: str) -> None:
    edges: Counter[tuple[tuple[float, ...], tuple[float, ...]]] = Counter()
    area_tolerance = 1e-10
    for face in faces:
        origin = vertices[face[0]]
        for offset in range(1, len(face) - 1):
            area = np.linalg.norm(np.cross(vertices[face[offset]] - origin,
                                           vertices[face[offset + 1]] - origin)) / 2
            test.assertGreater(area, area_tolerance, f"{label} has a degenerate face")
        for index, start in enumerate(face):
            end = face[(index + 1) % len(face)]
            a = tuple(np.round(vertices[start], 6))
            b = tuple(np.round(vertices[end], 6))
            edges[tuple(sorted((a, b)))] += 1
    test.assertTrue(edges, f"{label} has no edges")
    test.assertTrue(all(count == 2 for count in edges.values()), f"{label} is not closed")


def _bounds(vertices: np.ndarray) -> np.ndarray:
    return np.stack((vertices.min(axis=0), vertices.max(axis=0)))


class SwarmHingeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temporary.name) / "hinge_export"
        result = export(cls.out)
        if Path(result).resolve() != cls.out.resolve():
            raise AssertionError("export() did not return its output directory")
        cls.manifest = json.loads((cls.out / "cad_manifest.json").read_text())
        cls.geometry = json.loads((cls.out / "geometry.json").read_text())
        cls.model = mujoco.MjModel.from_xml_path(str(cls.out / "robot.xml"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _checked_artifact(self, record: dict, expected: str) -> Path:
        self.assertEqual(record["path"], expected)
        path = self.out / record["path"]
        self.assertTrue(path.is_file(), f"Missing exported mesh: {record['path']}")
        self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        return path

    def _mesh_pair(self, records: dict, stem: str) -> tuple[np.ndarray, np.ndarray]:
        stl_path = self._checked_artifact(records["stl"], stem + ".stl")
        obj_path = self._checked_artifact(records["obj"], stem + ".obj")
        stl_vertices, stl_faces = _stl_mesh(stl_path)
        obj_vertices, obj_faces = _obj_mesh(obj_path)
        _assert_closed_mesh(self, stl_vertices, stl_faces, stl_path.name)
        _assert_closed_mesh(self, obj_vertices, obj_faces, obj_path.name)
        np.testing.assert_allclose(_bounds(stl_vertices), _bounds(obj_vertices), atol=1e-5, rtol=0)
        return stl_vertices, obj_vertices

    def test_neutral_assembly_is_a_finite_closed_stl_and_obj(self):
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(self.manifest["units"], "millimeters")
        self.assertEqual(self.geometry["units"], "meters")
        self.assertIn("arena_mujoco/swarm_hinge_robot.py", self.manifest["source"])
        self.assertFalse(any(token in geom["name"] for geom in self.geometry["geoms"]
                             for token in ("wheel", "tire", "lift", "pad", "skid")))
        self._checked_artifact(self.manifest["files"]["robot_xml"], "robot.xml")
        self._checked_artifact(self.manifest["files"]["geometry"], "geometry.json")
        self.assertEqual(self.manifest["neutral_assembly"]["pose"],
                         {"joint_angles_rad": [0, 0, 0]})

        stl_vertices, _ = self._mesh_pair(
            self.manifest["neutral_assembly"], "neutral_assembly_mm")
        mesh_bounds = _bounds(stl_vertices)
        np.testing.assert_allclose(mesh_bounds,
                                   self.manifest["neutral_assembly"]["bounds_mm"],
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(mesh_bounds / 1000,
                                   HINGE_DESIGN["neutral_shell_bounds_m"],
                                   atol=2.5e-4, rtol=0)

    def test_local_parts_follow_the_compiled_body_hierarchy(self):
        bodies = {body["name"]: body for body in self.manifest["bodies"]}
        model_names = {self.model.body(body_id).name for body_id in range(1, self.model.nbody)}
        self.assertEqual(set(bodies), model_names)
        self.assertEqual(len(bodies), 3)

        local_bounds = {}
        for name, body in bodies.items():
            body_id = self.model.body(name).id
            parent_id = int(self.model.body_parentid[body_id])
            expected_parent = self.model.body(parent_id).name
            self.assertEqual(body["parent"], expected_parent)
            np.testing.assert_allclose(body["position_parent_m"], self.model.body_pos[body_id],
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(body["orientation_parent_quat_wxyz"],
                                       self.model.body_quat[body_id], atol=1e-12, rtol=0)
            self.assertAlmostEqual(body["mass_kg"], self.model.body_mass[body_id], places=12)
            np.testing.assert_allclose(body["local_center_of_mass_m"],
                                       self.model.body_ipos[body_id], atol=1e-12, rtol=1e-9)
            np.testing.assert_allclose(body["local_inertia_kg_m2"],
                                       self.model.body_inertia[body_id], atol=1e-12, rtol=1e-9)
            expected_geoms = {
                self.model.geom(geom_id).name for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) == body_id
            }
            self.assertEqual(set(body["geoms"]), expected_geoms)
            expected_joints = {
                self.model.joint(joint_id).name for joint_id in range(self.model.njnt)
                if int(self.model.jnt_bodyid[joint_id]) == body_id
                and self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
            }
            self.assertEqual(set(body["joints"]), expected_joints)

            stl_vertices, _ = self._mesh_pair(body["parts"], f"parts/{name}_local_mm")
            local_bounds[name] = _bounds(stl_vertices)
            np.testing.assert_allclose(local_bounds[name], body["local_bounds_mm"],
                                       atol=1e-5, rtol=0)

        self.assertAlmostEqual(sum(body["mass_kg"] for body in bodies.values()),
                               HINGE_DESIGN["mass_kg"], places=12)
        np.testing.assert_allclose(local_bounds["hinge_00_front"],
                                   local_bounds["hinge_00_rear"], atol=1e-5, rtol=0)

    def test_joint_records_match_compiled_axes_limits_and_motor_assumptions(self):
        joints = {joint["name"]: joint for joint in self.manifest["joints"]}
        expected_names = {
            self.model.joint(joint_id).name for joint_id in range(self.model.njnt)
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        }
        self.assertEqual(set(joints), expected_names)
        self.assertEqual(len(joints), HINGE_DESIGN["physical_motor_count"])

        for name, record in joints.items():
            joint_id = self.model.joint(name).id
            self.assertEqual(record["body"], self.model.body(self.model.jnt_bodyid[joint_id]).name)
            np.testing.assert_allclose(record["axis_body"], self.model.jnt_axis[joint_id],
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(record["anchor_body_m"], self.model.jnt_pos[joint_id],
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(record["limits_rad"], self.model.jnt_range[joint_id],
                                       atol=1e-10, rtol=0)

            actuator = record["actuator"]
            actuator_id = self.model.actuator(actuator["name"]).id
            self.assertEqual(int(self.model.actuator_trnid[actuator_id, 0]), joint_id)
            np.testing.assert_allclose(actuator["ctrlrange_nm"],
                                       self.model.actuator_ctrlrange[actuator_id], atol=1e-12, rtol=0)
            np.testing.assert_allclose(actuator["forcerange_nm"],
                                       self.model.actuator_forcerange[actuator_id], atol=1e-12, rtol=0)
            np.testing.assert_allclose(actuator["gear"], self.model.actuator_gear[actuator_id],
                                       atol=1e-12, rtol=0)
            assumptions = {
                "max_torque_nm": "max_motor_torque_nm",
                "no_load_speed_rad_s": "motor_no_load_speed_rad_s",
                "position_gain_nm_rad": "position_gain_nm_rad",
                "velocity_gain_nm_s_rad": "velocity_gain_nm_s_rad",
            }
            for actuator_key, design_key in assumptions.items():
                self.assertAlmostEqual(actuator[actuator_key], HINGE_DESIGN[design_key], places=12)

        expected_assumptions = {key: HINGE_DESIGN[key] for key in (
            "physical_motor_count", "max_motor_torque_nm", "motor_no_load_speed_rad_s",
            "position_gain_nm_rad", "velocity_gain_nm_s_rad", "joint_armature_kg_m2")}
        self.assertEqual(self.manifest["motor_assumptions"], expected_assumptions)
        for record in joints.values():
            joint_id = self.model.joint(record["name"]).id
            dof_id = int(self.model.jnt_dofadr[joint_id])
            self.assertAlmostEqual(record["armature_kg_m2"], self.model.dof_armature[dof_id],
                                   places=12)

    def test_swept_envelope_contains_an_independent_dense_joint_grid(self):
        swept = self.manifest["swept_envelope"]
        joint_samples = swept["joint_samples"]
        joint_names = [joint["name"] for joint in self.manifest["joints"]]
        self.assertEqual(set(joint_samples), set(joint_names))
        self.assertTrue(all(isinstance(count, int) and count >= 3
                            for count in joint_samples.values()))
        expected_pose_count = sum(
            math.prod(joint_samples[name] for name in body["joints"])
            if body["joints"] else 1
            for body in self.manifest["bodies"]
        )
        self.assertEqual(swept["total_pose_samples"], expected_pose_count)
        self.assertIsInstance(swept["method"], str)
        self.assertTrue(swept["method"])
        margin = swept["conservative_margin_m"]
        self.assertTrue(math.isfinite(margin) and 0 <= margin <= .002)

        declared = np.asarray(swept["bounds_root_m"], dtype=float)
        self.assertEqual(declared.shape, (2, 3))
        self.assertTrue(np.isfinite(declared).all())
        self.assertTrue(np.all(declared[0] < declared[1]))
        neutral = np.asarray(HINGE_DESIGN["neutral_shell_bounds_m"])
        self.assertTrue(np.all(declared[0] <= neutral[0]))
        self.assertTrue(np.all(declared[1] >= neutral[1]))

        sampled = self._sample_root_bounds(7)
        exceedance = max(float(np.max(declared[0] - sampled[0])),
                         float(np.max(sampled[1] - declared[1])), 0.)
        self.assertLessEqual(exceedance, 2e-6)
        self.assertTrue(np.all(declared[0] >= sampled[0] - margin - .001))
        self.assertTrue(np.all(declared[1] <= sampled[1] + margin + .001))

        verification = swept["verification"]
        self.assertTrue(verification["passed"])
        self.assertLessEqual(verification["max_bound_exceedance_m"], 2e-6)
        expected_verification_counts = {
            body["name"]: (math.prod(joint_samples[name] - 1 for name in body["joints"])
                           if body["joints"] else 1)
            for body in self.manifest["bodies"]
        }
        self.assertEqual(verification["body_pose_counts"], expected_verification_counts)
        self.assertEqual(verification["pose_samples"], sum(expected_verification_counts.values()))

    def test_magnetic_export_adds_lobe_housings_without_extra_bodies_or_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "hinge_magnetic_export"
            export(out, magnetic=True)
            manifest = json.loads((out / "cad_manifest.json").read_text())
            geometry = json.loads((out / "geometry.json").read_text())
            model = mujoco.MjModel.from_xml_path(str(out / "robot.xml"))
            self.assertEqual(manifest["variant"], "hinge_magnetic_module")
            self.assertEqual(len(manifest["bodies"]), 3)
            self.assertAlmostEqual(manifest["total_mass_kg"], HINGE_DESIGN["mass_kg"], places=12)
            np.testing.assert_allclose(manifest["neutral_assembly"]["bounds_mm"],
                                       [[-12, -27, -5], [12, 27, 11]], atol=2e-6, rtol=0)
            docking = geometry["robot"]["magnetic_docking"]
            self.assertEqual(docking["ports"], 4)
            self.assertEqual({frame["body"] for frame in docking["port_frames"]},
                             {"hinge_00_front", "hinge_00_rear"})
            housings = [geom for geom in geometry["geoms"] if "_magnet_" in geom["name"]]
            self.assertEqual(len(housings), 4)
            self.assertTrue(all(geom["shape"] == "box" for geom in housings))
            self.assertTrue(all(model.geom(geom["name"]).bodyid[0] != model.body("hinge_00").id
                                for geom in housings))
            self.assertTrue(manifest["swept_envelope"]["verification"]["passed"])

    def _sample_root_bounds(self, samples_per_joint: int) -> np.ndarray:
        data = mujoco.MjData(self.model)
        root_id = self.model.body("hinge_00").id
        geom_ids = [self.model.geom(name).id for body in self.manifest["bodies"]
                    for name in body["geoms"]]
        joints = [self.model.joint(record["name"]).id for record in self.manifest["joints"]]
        grids = [np.linspace(*self.model.jnt_range[joint_id], samples_per_joint)
                 for joint_id in joints]
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        for angles in itertools.product(*grids):
            data.qpos[:] = self.model.qpos0
            for joint_id, angle in zip(joints, angles):
                data.qpos[self.model.jnt_qposadr[joint_id]] = angle
            mujoco.mj_forward(self.model, data)
            root_position = data.xpos[root_id]
            root_rotation = data.xmat[root_id].reshape(3, 3)
            for geom_id in geom_ids:
                geom_position = (data.geom_xpos[geom_id] - root_position) @ root_rotation
                if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    radius = self.model.geom_size[geom_id, 0]
                    lower = np.minimum(lower, geom_position - radius)
                    upper = np.maximum(upper, geom_position + radius)
                elif self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_id = self.model.geom_dataid[geom_id]
                    start = self.model.mesh_vertadr[mesh_id]
                    stop = start + self.model.mesh_vertnum[mesh_id]
                    vertices = self.model.mesh_vert[start:stop]
                    world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
                    local = (world - root_position) @ root_rotation
                    lower = np.minimum(lower, local.min(axis=0))
                    upper = np.maximum(upper, local.max(axis=0))
                else:
                    self.fail(f"Unsupported robot geom type: {self.model.geom_type[geom_id]}")
        return np.stack((lower, upper))


if __name__ == "__main__":
    unittest.main()
