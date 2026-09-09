"""Material changes must affect native contact dynamics in the fast arena."""
import unittest

import mujoco

from arena_mujoco.runtime import ArenaSimulation
from arena_mujoco.builder import build_arena


class SurfaceFrictionTests(unittest.TestCase):
    def test_unstable_overlap_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Overlapping flex layers failed"):
            build_arena(profile={"tape":{"joins":"overlap"}})

    def test_wood_slides_farther_on_vinyl_than_paper(self):
        speeds = {}
        for mode in ("none", "rigid"):
            sim = ArenaSimulation(robot=False, tape_mode=mode)
            joint = sim.model.joint("Cylinder_Yellow_01_free")
            qpos, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
            # The 20 mm cylinder fits the 20 mm vertical strip, away from crossings.
            top = 0.00015 if mode == "rigid" else 0.0
            sim.data.qpos[qpos:qpos + 3] = [0.34, 0.7, 0.0102 + top]
            mujoco.mj_forward(sim.model, sim.data)
            sim.step(nstep=200)
            sim.data.qvel[dof + 1] = 0.2
            mujoco.mj_forward(sim.model, sim.data)
            sim.step(nstep=50)
            speeds[mode] = float(sim.data.qvel[dof + 1])
            self.assertEqual(int(sim.data.warning.number.sum()), 0)
        self.assertGreater(speeds["rigid"], speeds["none"] + 0.008)
        self.assertLess(speeds["rigid"], 0.2)


if __name__ == "__main__":
    unittest.main()
