"""Competition outcomes require complete geometry, released pieces, and history."""
import unittest

import numpy as np

from arena_mujoco.competition_scoring import score_competition
from arena_mujoco.runtime import ArenaSimulation


class CompetitionScoringTests(unittest.TestCase):
    def setUp(self):
        self.sim = ArenaSimulation(tape_mode="none")
        self.sim.metadata["competition_events"] = {"robot_outside_count": 0,
                                                   "human_intervention": False,
                                                   "initial_setup_valid": True}
        self.groups = {}
        for item in self.sim.metadata["objects"]:
            self.groups.setdefault(item["category"], []).append(item["name"])

    def place(self, name, xyz, quat=(1, 0, 0, 0)):
        joint = self.sim.model.joint(name + "_free")
        qpos, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.sim.data.qpos[qpos:qpos + 3] = xyz
        self.sim.data.qpos[qpos + 3:qpos + 7] = quat
        self.sim.data.qvel[dof:dof + 6] = 0

    def full_layout(self):
        for name, y in zip(self.groups["sample"], (0.3905, 0.4905, 0.5905)):
            self.place(name, (1.068, y, 0.0025))
        for name, xy in zip(self.groups["medical_kit"], ((0.05, 0.5), (0.1, 0.5), (0.05, 0.1), (0.05, 1.0))):
            self.place(name, (*xy, 0.01))
        for name, x in zip(self.groups["patient"][:3], (0.04, 0.09, 0.14)):
            self.place(name, (x, 0.4, 0.01))
        for name, xy in zip(self.groups["observation"][:3], ((0.09, 0.1), (0.14, 0.1), (0.1, 1.0))):
            self.place(name, (*xy, 0.01))
        for name, x in zip(self.groups["low_risk"][:3], (1.02, 1.07, 1.12)):
            self.place(name, (x, 0.94, 0.01))

    def score(self, limit=None):
        return score_competition(self.sim.model, self.sim.data, self.sim.metadata, time_limit_s=limit)

    def test_full_score_and_official_time_are_separate(self):
        self.full_layout()
        self.sim.data.time = 119
        result = self.score(120)
        self.assertEqual(result["score"], 160)
        self.assertTrue(result["official_success"])
        self.assertEqual(len(result["sample_slot_assignments"]), 3)
        self.sim.data.time = 121
        result = self.score(120)
        self.assertEqual(result["task_score"], 160)
        self.assertFalse(result["official_success"])
        self.assertIsNone(result["official_score"])
        result = self.score()
        self.assertTrue(result["feasibility_success"])
        self.assertFalse(result["official_success"])

    def test_wrong_color_cancels_destination_patient_points_only(self):
        self.full_layout()
        self.place(self.groups["low_risk"][3], (0.1, 0.6, 0.01))
        result = self.score()
        self.assertEqual(result["per_task"]["red_patients"]["points"], 0)
        self.assertEqual(result["per_task"]["kits"]["points"], 40)
        self.assertEqual(result["score"], 130)
        self.assertIn("hospital", result["contaminated_zones"])

    def test_yellow_requires_both_pccs(self):
        self.full_layout()
        self.place(self.groups["observation"][2], (0.1, 0.2, 0.01))
        result = self.score()
        self.assertFalse(result["yellow_distribution_satisfied"])
        self.assertEqual(result["per_task"]["yellow_patients"]["points"], 0)

    def test_rotated_kit_crossing_boundary_does_not_score(self):
        self.full_layout()
        name = self.groups["medical_kit"][0]
        angle = np.pi / 4
        self.place(name, (0.017, 0.5, 0.01), (np.cos(angle/2), 0, 0, np.sin(angle/2)))
        result = self.score()
        self.assertEqual(result["per_task"]["kits"]["points"], 30)
        self.assertLess(result["objects"][name]["zone_clearance_m"]["hospital"], 0)

    def test_sample_center_only_floating_and_moving_checks(self):
        self.full_layout()
        first, second, third = self.groups["sample"]
        self.place(first, (1.0701, 0.3905, 0.0025))
        self.place(second, (1.068, 0.4905, 0.08))
        dof = int(self.sim.model.joint(third + "_free").dofadr[0])
        self.sim.data.qvel[dof] = 0.02
        result = self.score()
        self.assertEqual(result["per_task"]["samples"]["points"], 0)
        self.assertLess(result["objects"][first]["slot_clearance_m"]["lab_1"], 0)
        self.assertFalse(result["objects"][second]["seated"])
        self.assertFalse(result["objects"][third]["stable"])

    def test_robot_contact_prevents_release_credit(self):
        self.full_layout()
        qpos = int(self.sim.model.joint("robot_free").qposadr[0])
        self.sim.data.qpos[qpos:qpos + 3] = (0.09, 0.5, 0.045)
        name = self.groups["patient"][0]
        self.place(name, (0.09, 0.5799, 0.01))
        result = self.score()
        self.assertFalse(result["objects"][name]["released"])
        self.assertEqual(result["per_task"]["red_patients"]["points"], 20)

    def test_missing_history_and_prior_exit_prevent_proof(self):
        self.full_layout()
        self.sim.metadata.pop("competition_events")
        result = self.score()
        self.assertEqual(result["task_score"], 160)
        self.assertFalse(result["event_history_complete"])
        self.assertFalse(result["success"])
        self.sim.metadata["competition_events"] = {"robot_outside_count": 2,
                                                   "human_intervention": False,
                                                   "initial_setup_valid": True}
        result = self.score()
        self.assertEqual(result["score"], 140)
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
