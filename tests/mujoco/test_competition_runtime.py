import copy
import math
import unittest

import numpy as np

from arena_mujoco.competition_events import initialize_competition_events, update_competition_events
from arena_mujoco.competition_policy import ExpertPolicy
from arena_mujoco.competition_runtime import CompetitionSimulation


class CompetitionRuntimeTests(unittest.TestCase):
    def simulation(self, **kwargs):
        sim = CompetitionSimulation(tape_mode="none", **kwargs)
        setup = initialize_competition_events(sim.model, sim.data, sim.metadata)
        self.assertTrue(setup["valid"], setup["errors"])
        return sim

    def controls(self, sim):
        policy = ExpertPolicy()
        for goals in ((.08, .12, .03, .014), (-.04, -.07, .01, .022), (.02, 0, .04, .008)):
            sim.control(policy, *goals)

    def test_checkpoint_restores_controller_history_and_replays_physics(self):
        sim = self.simulation()
        self.controls(sim)
        checkpoint = sim.get_state()
        self.controls(sim)
        expected = sim.get_state()
        update_competition_events(sim.model, sim.data, sim.metadata, human_intervention=True)
        sim.set_state(checkpoint)
        self.assertFalse(sim.metadata["competition_events"]["human_intervention"])
        self.assertEqual(len(sim.observations), 3)
        np.testing.assert_array_equal(sim.motor_integrals, checkpoint["competition_controller"]["motor_integrals"])
        self.controls(sim)
        actual = sim.get_state()
        np.testing.assert_allclose(actual["physics"], expected["physics"], rtol=0, atol=1e-10)
        self.assertEqual(actual["competition_events"], expected["competition_events"])
        for name, value in expected["competition_controller"].items():
            np.testing.assert_allclose(actual["competition_controller"][name], value, rtol=0, atol=1e-10)
        # Runtime changes must not mutate a checkpoint retained for another replay.
        checkpoint_events = copy.deepcopy(checkpoint["competition_events"])
        update_competition_events(sim.model, sim.data, sim.metadata, human_intervention=True)
        self.assertEqual(checkpoint["competition_events"], checkpoint_events)

    def test_reset_clears_controller_and_history_then_matches_fresh_run(self):
        sim = self.simulation()
        self.controls(sim)
        update_competition_events(sim.model, sim.data, sim.metadata, human_intervention=True)
        sim.reset()
        self.assertEqual(float(sim.data.time), 0)
        np.testing.assert_array_equal(sim.motor_integrals, np.zeros(2))
        self.assertEqual((sim.lift_target, sim.jaw_target), (0, .025))
        self.assertEqual((sim.distance_m, sim.max_tilt_rad, sim.max_wheel_command), (0, 0, 0))
        self.assertEqual((sim.observations, sim.actions), ([], []))
        self.assertFalse(sim.metadata["competition_events"]["human_intervention"])
        self.assertNotIn("initial_setup_valid", sim.metadata["competition_events"])
        setup = initialize_competition_events(sim.model, sim.data, sim.metadata)
        self.assertTrue(setup["valid"], setup["errors"])
        fresh = self.simulation()
        self.controls(sim)
        self.controls(fresh)
        np.testing.assert_allclose(sim.data.qpos, fresh.data.qpos, rtol=0, atol=1e-12)
        np.testing.assert_allclose(sim.data.qvel, fresh.data.qvel, rtol=0, atol=1e-12)

    def test_restoring_another_physics_profile_refreshes_robot_metadata(self):
        original = self.simulation(seed=31, randomize=True)
        self.controls(original)
        restored = self.simulation(seed=32, randomize=True)
        restored.set_state(original.get_state())
        self.assertTrue(restored.meta is restored.metadata["robot"],
                        "Robot metadata cache must follow the recompiled checkpoint model")
        self.assertEqual(restored.meta["max_motor_torque_nm"], original.meta["max_motor_torque_nm"])
        self.controls(original)
        self.controls(restored)
        np.testing.assert_allclose(restored.data.qpos, original.data.qpos, rtol=0, atol=1e-12)

    def test_seeded_physics_varies_between_seeds_and_repeats_after_reset(self):
        first = self.simulation(seed=11, randomize=True)
        second = self.simulation(seed=12, randomize=True)
        repeat = self.simulation(seed=11, randomize=True)
        self.assertNotEqual(first.xml, second.xml)
        self.assertNotEqual(first.profile["wood_density_kg_m3"], second.profile["wood_density_kg_m3"])
        self.assertNotEqual(first.metadata["robot"]["max_motor_torque_nm"],
                            second.metadata["robot"]["max_motor_torque_nm"])
        self.assertEqual(first.xml, repeat.xml)
        self.controls(first)
        self.controls(repeat)
        np.testing.assert_array_equal(first.data.qpos, repeat.data.qpos)
        first.reset(seed=12)
        self.assertEqual(first.xml, second.xml)
        self.assertTrue(initialize_competition_events(first.model, first.data, first.metadata)["valid"])
        self.controls(first)
        self.controls(second)
        np.testing.assert_array_equal(first.data.qpos, second.data.qpos)

    def test_metrics_use_current_pose_after_each_one_millisecond_step_batch(self):
        sim = self.simulation(timestep=.001)
        previous = sim.pose()[0]
        distance, maximum_tilt = 0., 0.
        for _ in range(4):
            sim.control(ExpertPolicy(), .10, .15, .025, .02)
            position = sim.pose()[0]
            distance += float(np.linalg.norm(position - previous))
            previous = position
            rotation = sim.data.body("competition_robot").xmat.reshape(3, 3)
            maximum_tilt = max(maximum_tilt, math.acos(np.clip(rotation[2, 2], -1, 1)))
            np.testing.assert_allclose(sim.last_axle, position, rtol=0, atol=1e-14)
            self.assertAlmostEqual(sim.distance_m, distance, places=14)
            self.assertAlmostEqual(sim.max_tilt_rad, maximum_tilt, places=14)
            self.assertEqual(sim.metadata["competition_events"]["last_observed_time_s"], float(sim.data.time))
        self.assertGreater(distance, .001)


if __name__ == "__main__":
    unittest.main()
