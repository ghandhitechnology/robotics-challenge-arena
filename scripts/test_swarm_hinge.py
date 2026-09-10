#!/usr/bin/env python3
"""Check free-body hinge mechanics and motor-only directional locomotion."""
from pathlib import Path
import math
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.swarm_hinge_robot import HINGE_DESIGN, add_hinge_robot, hinge_motor_torques
from arena_mujoco.swarm_robot import add_swarm_robot
from scripts.diagnose_swarm_hinge import Gait, run, scene


class SwarmHingeTests(unittest.TestCase):
    def test_geometry_has_only_original_shells_and_one_free_root(self):
        xml, spec = scene()
        root = ET.fromstring(xml)
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual((model.nq, model.nv, model.nu, model.neq), (10, 9, 3, 0))
        self.assertEqual(model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertTrue(np.all(model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE))
        self.assertTrue(model.jnt_limited[1:].all())
        self.assertAlmostEqual(model.body_mass.sum(), .040, places=12)
        self.assertEqual(len(spec['colliders']), 3)
        self.assertEqual(len(root.findall('.//geom')), 4)  # Three shells and the board.
        for motor in root.findall('./actuator/motor'):
            self.assertIn(motor.get('joint'), spec['joint_names'])
        baseline = ET.Element('mujoco')
        add_swarm_robot(baseline, 0)
        np.testing.assert_allclose(
            np.fromstring(root.find('./asset/mesh').get('vertex'), sep=' '),
            np.fromstring(baseline.find('./asset/mesh').get('vertex'), sep=' '))

    def test_builder_preserves_joint_travel_with_default_degree_compiler(self):
        root = ET.Element('mujoco')
        add_hinge_robot(root)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        np.testing.assert_allclose(model.jnt_range[1:], HINGE_DESIGN['joint_limits_rad'], atol=1e-10)

    def test_motor_drive_respects_torque_speed_and_passive_braking(self):
        speed = HINGE_DESIGN['motor_no_load_speed_rad_s']
        torque = hinge_motor_torques(np.zeros(3), [speed, speed/2, -speed], np.ones(3))
        self.assertEqual(torque[0], 0.)
        self.assertLessEqual(abs(torque[1]), .004)
        self.assertGreater(torque[2], 0.)
        self.assertLessEqual(abs(torque).max(), .008)
        with self.assertRaises(ValueError):
            hinge_motor_torques([0, 0, float('nan')], [0, 0, 0], [0, 0, 0])

    def test_phase_order_reverses_net_travel_without_external_forces(self):
        forward = run(Gait(direction=-1), seconds=6)
        reverse = run(Gait(direction=1), seconds=6)
        stationary = run(Gait(amplitude_rad=0), seconds=6)
        self.assertGreater(forward['forward_mm'], 10.)
        self.assertLess(reverse['forward_mm'], -10.)
        self.assertLess(abs(stationary['forward_mm']), .01)
        for result in (forward, reverse, stationary):
            self.assertEqual(result['external_applied_force_max'], 0.)
            self.assertEqual(result['warnings'], 0)
            self.assertLessEqual(result['peak_motor_torque_nm'], .008)
            self.assertLess(result['maximum_shell_height_mm'], 25.)
            self.assertLess(result['maximum_tilt_deg'], 10.)

    def test_yaw_hinge_produces_mirrored_turns_while_crawling(self):
        common = dict(direction=-1, steering_phase_rad=math.pi/2)
        left = run(Gait(steering_rad=-math.radians(9), **common), seconds=6)
        right = run(Gait(steering_rad=math.radians(9), **common), seconds=6)
        self.assertGreater(left['yaw_change_deg'], 3.)
        self.assertLess(right['yaw_change_deg'], -3.)
        self.assertGreater(left['forward_mm'], 10.)
        self.assertGreater(right['forward_mm'], 10.)
        self.assertAlmostEqual(left['yaw_change_deg'], -right['yaw_change_deg'], places=5)
        self.assertAlmostEqual(left['lateral_mm'], -right['lateral_mm'], places=5)


if __name__ == '__main__':
    unittest.main()
