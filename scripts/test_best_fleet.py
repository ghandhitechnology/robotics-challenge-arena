#!/usr/bin/env python3
"""Native checks for fleet starting geometry and gravity-only kit discharge."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from arena_mujoco.best_fleet import FleetSimulation


class FleetMechanicsTests(unittest.TestCase):
    def test_full_fleet_start_and_actuators(self):
        sim = FleetSimulation()
        self.assertTrue(sim.setup["valid"], sim.setup["errors"])
        self.assertEqual(len(sim.setup["preloaded_kits"]), 4)
        self.assertEqual(sim.model.nu, 22)
        self.assertEqual(len(sim.robots), 5)
        for robot in sim.robots.values():
            self.assertEqual(sim.model.joint(robot["freejoint"]).type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertFalse(np.isin(sim.model.eq_type, [mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT]).any())

    def test_gravity_gates_release_all_four_free_kits(self):
        sim = FleetSimulation(only=["kit"])
        self.assertTrue(sim.setup["valid"], sim.setup["errors"])
        for _ in range(25):
            sim.step()
        sim.gate_targets["kit"][:] = np.pi / 2
        for _ in range(175):
            sim.step()
        for index in range(1, 5):
            body = sim.data.body(f"Medical_Kit_{index:02d}")
            self.assertLess(body.xpos[2], .021)
        score = sim.score()
        for name, record in score["objects"].items():
            if name.startswith("Medical_Kit"):
                self.assertTrue(record["released"], name)
                self.assertTrue(record["stable"], name)
        self.assertFalse(np.any(sim.data.xfrc_applied))
        self.assertLessEqual(sim.max_torque, .025 + 1e-12)


if __name__ == "__main__":
    unittest.main()
