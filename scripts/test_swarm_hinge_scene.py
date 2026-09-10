#!/usr/bin/env python3
"""Check articulated packing, scene ownership, and real arena contact dynamics."""
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.swarm_hinge_robot import hinge_motor_torques
from arena_mujoco.swarm_hinge_scene import build_hinge_swarm_scene, hinge_packing
from arena_mujoco.swarm_magnets import MagneticCoupling
from arena_mujoco.swarm_flow import connected_components


class HingeSceneTests(unittest.TestCase):
    def test_all_forty_free_modules_fit_the_start_zone(self):
        xml, metadata = build_hinge_swarm_scene(num_objects=1)
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual(model.nu, 120)
        self.assertEqual(model.neq, 0)
        for spec in metadata["robots"]:
            root = model.body(spec["name"]).id
            owned = [root] + [model.body(spec["name"]+"_"+side).id for side in ("front", "rear")]
            self.assertAlmostEqual(float(model.body_mass[owned].sum()), .040)
            self.assertEqual(model.joint(spec["freejoint"]).type[0], mujoco.mjtJoint.mjJNT_FREE)
            self.assertEqual([model.joint(n).type[0] for n in spec["joint_names"]],
                             [mujoco.mjtJoint.mjJNT_HINGE]*3)
            for joint in spec["joint_names"]:
                self.assertTrue(model.joint(joint).limited[0])
        poses = hinge_packing()
        lower = poses[:, :2] - [.027, .012]
        upper = poses[:, :2] + [.027, .012]
        self.assertTrue(np.all(lower >= [.86, .72]))
        self.assertTrue(np.all(upper <= [1.14, 1.20]))

    def test_moving_ports_connect_the_whole_neutral_packing(self):
        xml, metadata = build_hinge_swarm_scene()
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        magnets = MagneticCoupling(model, metadata["robots"])
        magnets.apply(data, np.ones(40, bool))
        _, sizes = connected_components(magnets.load_bearing_graph(.002))
        self.assertEqual(max(sizes), 40)
        self.assertFalse(np.any(np.isin(magnets.force_bodies, magnets.robot_bodies)))
        force = data.xfrc_applied[magnets.force_bodies]
        moment = force[:, 3:] + np.cross(data.xipos[magnets.force_bodies], force[:, :3])
        np.testing.assert_allclose(force[:, :3].sum(0), 0, atol=1e-12)
        np.testing.assert_allclose(moment.sum(0), 0, atol=1e-12)

    def test_small_body_rises_from_flat_pose_using_only_hinge_torques(self):
        xml, metadata = build_hinge_swarm_scene(2)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        specs = metadata["robots"]
        magnets = MagneticCoupling(model, specs)
        qa = np.array([[model.joint(n).qposadr[0] for n in s["joint_names"]] for s in specs])
        da = np.array([[model.joint(n).dofadr[0] for n in s["joint_names"]] for s in specs])
        ua = np.array([[model.actuator(n).id for n in s["motor_names"]] for s in specs])
        untouched = np.setdiff1d(np.arange(model.nbody), magnets.force_bodies)
        initial = data.qpos.copy()
        for _ in range(2000):
            target = np.array([0., -.35, -.35])*min(data.time/1.5, 1.)
            mujoco.mj_step1(model, data)
            data.ctrl[ua] = hinge_motor_torques(data.qpos[qa], data.qvel[da], target)
            magnets.apply(data, [True, True])
            mujoco.mj_step2(model, data)
        self.assertFalse(np.any(data.warning.number))
        self.assertFalse(np.any(data.qfrc_applied))
        self.assertFalse(np.any(data.xfrc_applied[untouched]))
        for spec in specs:
            adr = model.joint(spec["freejoint"]).qposadr[0]
            self.assertGreater(data.qpos[adr+2]-initial[adr+2], .003)
        self.assertTrue(magnets.load_bearing_graph(.002)[0, 1])


if __name__ == "__main__":
    unittest.main()
