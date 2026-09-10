#!/usr/bin/env python3
"""Check native batch isolation, force accounting, and read-only observations."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.swarm_hinge_dynamics import HingeBatchDynamics


class HingeDynamicsTests(unittest.TestCase):
    def test_parallel_worlds_match_serial_physics_and_reset_is_isolated(self):
        serial = HingeBatchDynamics(2, 2, native_workers=1)
        parallel = HingeBatchDynamics(2, 2, native_workers=2)
        try:
            for engine in (serial, parallel):
                engine.reset(friction=[.6, .8], motor_strength=[.9, 1.])
            for step in range(40):
                target = np.zeros((2, 2, 4))
                target[..., 1:3] = -.4*min(step/30, 1.)
                target[..., 3] = 1.
                target[1, :, 0] = .05*np.sin(step*.2)
                a, b = serial.step(target), parallel.step(target)
            np.testing.assert_array_equal(a["qpos"], b["qpos"])
            np.testing.assert_array_equal(a["qvel"], b["qvel"])
            before = parallel.native_data[1].qpos.copy()
            time_before = parallel.native_data[1].time
            parallel.reset([0], friction=.55)
            np.testing.assert_array_equal(parallel.native_data[1].qpos, before)
            self.assertEqual(parallel.native_data[1].time, time_before)
            self.assertTrue(np.all(parallel.models[0].geom_friction[parallel.shell_geoms, 0] == .55))
            self.assertTrue(np.all(parallel.models[1].geom_friction[parallel.shell_geoms, 0] == .8))
        finally:
            serial.close()
            parallel.close()

    def test_state_reads_preserve_trajectory_and_magnetic_partners(self):
        engine = HingeBatchDynamics(1, 4, 1)
        try:
            action = np.broadcast_to([0., -.4, -.4, 1.], (1, 4, 4))
            for _ in range(20):
                engine.step(action)
            before = engine.read_state()
            partners = engine.magnets[0].interacting_pairs.copy()
            for _ in range(3):
                after = engine.read_state()
                for key in before:
                    np.testing.assert_array_equal(before[key], after[key])
            self.assertEqual(partners, engine.magnets[0].interacting_pairs)
            self.assertEqual(float(after["external_force_max"].max()), 0.)
            self.assertLessEqual(float(after["peak_torque_nm"].max()), .008)
        finally:
            engine.close()

    def test_stationary_payload_weight_is_measured_as_environment_support(self):
        engine = HingeBatchDynamics(1, 2, 1)
        try:
            for _ in range(30):
                state = engine.step(np.zeros((1, 2, 4)))
            weight = engine.model.body_mass[engine.object_ids[0]]*9.81
            self.assertAlmostEqual(state["object_environment_support_n"][0, 0], weight, delta=weight*.02)
            self.assertLess(abs(state["object_clearance"][0, 0]), .0002)
            self.assertFalse(state["object_contact_force_n"].any())
            self.assertFalse(state["magnetic_graph"].any())
            self.assertEqual(float(state["external_force_max"].max()), 0.)
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
