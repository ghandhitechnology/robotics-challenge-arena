"""World-level option values and robot self-contact isolation."""
import unittest

import mujoco

from arena_mujoco.builder import build_arena


class WorldPhysicsTests(unittest.TestCase):
    def test_solver_profile_reaches_the_compiled_world(self):
        xml, _ = build_arena(tape_mode="none", robot=False,
                             profile={"solver": {"solver": "Newton", "cone": "elliptic",
                                                 "impratio": 100}})
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual(int(model.opt.solver), int(mujoco.mjtSolver.mjSOL_NEWTON))
        self.assertEqual(int(model.opt.cone), int(mujoco.mjtCone.mjCONE_ELLIPTIC))
        self.assertAlmostEqual(float(model.opt.impratio), 100)

    def test_reference_world_keeps_its_solver_defaults(self):
        xml, _ = build_arena(tape_mode="none", robot=False)
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual(int(model.opt.solver), int(mujoco.mjtSolver.mjSOL_CG))
        self.assertEqual(int(model.opt.cone), int(mujoco.mjtCone.mjCONE_PYRAMIDAL))

    def test_unknown_solver_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported option keys"):
            build_arena(profile={"solver": {"cone": "elliptic", "warmstart": 1}})

    def test_robot_parts_do_not_get_explicit_contact_pairs(self):
        _, metadata = build_arena(tape_mode="rigid", robot=True)
        names = {item["name"] for item in metadata["robot"]["colliders"]}
        for pair in metadata["contact_pairs"]:
            self.assertFalse(pair["geom1"] in names and pair["geom2"] in names, pair["name"])


if __name__ == "__main__":
    unittest.main()
