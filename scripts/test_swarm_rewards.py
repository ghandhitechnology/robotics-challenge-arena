#!/usr/bin/env python3
"""Check one-time physical drop rewards through airborne support loss."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.swarm_env import SwarmVectorEnv


class DropRewardTests(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(drop_armed=torch.zeros(1, 1, dtype=torch.bool),
                                     phase=torch.tensor([[2]]), object_active=torch.ones(1, 1, dtype=torch.bool),
                                     object_floor_contact=torch.zeros(1, 1))

    def event(self, supported=False, clearance=.008, floor=False, phase=2):
        self.state.phase.fill_(phase)
        self.state.object_floor_contact.fill_(float(floor))
        return bool(SwarmVectorEnv._drop_events(self.state, torch.tensor([[supported]]),
                                                torch.tensor([[clearance]])).item())

    def test_airborne_contact_loss_is_not_a_drop(self):
        self.assertFalse(self.event(supported=True))
        for _ in range(30):
            self.assertFalse(self.event())
        self.assertTrue(self.state.drop_armed.item())

    def test_delayed_grounding_is_detected_after_recovery_starts(self):
        self.event(supported=True)
        self.assertFalse(self.event(phase=0))
        self.assertTrue(self.event(floor=True, phase=0))
        self.assertFalse(self.state.drop_armed.item())

    def test_low_clearance_detects_grounding_without_a_contact_flag(self):
        self.event(supported=True)
        self.assertFalse(self.event(clearance=.003))
        self.assertTrue(self.event(clearance=.001))

    def test_bounces_do_not_repeat_until_supported_regrasp(self):
        self.event(supported=True)
        self.assertTrue(self.event(floor=True))
        for _ in range(3):
            self.assertFalse(self.event(clearance=.006))
            self.assertFalse(self.event(floor=True, clearance=.001))
        self.assertFalse(self.event(supported=True))
        self.assertTrue(self.event(floor=True))

    def test_planned_lowering_disarms_before_grounding(self):
        self.event(supported=True)
        self.assertFalse(self.event(phase=3))
        self.assertFalse(self.state.drop_armed.item())
        self.assertFalse(self.event(phase=3, floor=True, clearance=0))
        self.assertFalse(self.event(phase=4, floor=True, clearance=0))

    def test_pickup_and_inactive_objects_do_not_arm_transport_drop(self):
        self.assertFalse(self.event(supported=True, phase=1))
        self.assertFalse(self.event(floor=True, phase=1))
        self.state.object_active.zero_()
        self.assertFalse(self.event(supported=True))
        self.assertFalse(self.event(floor=True))

    def test_native_step_charges_once_and_world_reset_clears_latch(self):
        env = SwarmVectorEnv(num_envs=2, num_robots=4, num_objects=1, backend="native")
        try:
            env.drop_armed.fill_(True)
            env.reset_done(torch.tensor([True, False]))
            self.assertEqual(env.drop_armed[:, 0].tolist(), [False, True])
            action = torch.zeros(2, 4, 3)
            action[..., 2] = -1
            _, _, _, _, info = env.step(action)
            self.assertEqual(info["reward_components"]["drop"].tolist(), [[0., 0., 0., 0.], [-2., -2., 0., 0.]])
            _, _, _, _, info = env.step(action)
            self.assertEqual(float(info["reward_components"]["drop"].abs().sum()), 0)
            env.drop_armed.fill_(True)
            env.reset()
            self.assertFalse(env.drop_armed.any())
        finally:
            env.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
