#!/usr/bin/env python3
"""Check the body environment's physical-action and reset boundaries."""
import math
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.swarm_body_env import SwarmBodyEnv


class BodyEnvironmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = SwarmBodyEnv(num_envs=2, num_robots=40, num_objects=2, native_workers=2)

    def tearDown(self):
        self.env.close()

    def test_actions_and_partial_reset(self):
        env = self.env
        obs = env.reset()
        self.assertEqual(tuple(obs['local'].shape), (2, 40, 40))
        self.assertEqual(tuple(obs['neighbors'].shape), (2, 40, 6, 12))
        self.assertEqual(tuple(obs['global'].shape), (2, 88))
        command = torch.zeros((2, 40, 4))
        command[..., 2] = -1
        command[1, :, 3] = 1
        _, reward, _, _, info = env.step(command)
        self.assertFalse(env.body_links[0].any())
        self.assertGreater(int(info['largest_magnetic_component'][1]), 1)
        self.assertTrue(torch.allclose(reward, sum(info['reward_components'].values())*env.active))
        saved = env.native_data[1].qpos.copy()
        links = env.body_links[1].copy()
        env.reset_done(torch.tensor([True, False]))
        self.assertEqual(int(env.steps[0]), 0)
        self.assertEqual(int(env.steps[1]), 1)
        np.testing.assert_array_equal(saved, env.native_data[1].qpos)
        np.testing.assert_array_equal(links, env.body_links[1])
        self.assertFalse(env.magnets[0].interaction_graph().any())
        for data in env.native_data:
            for obj in env.metadata['objects']:
                np.testing.assert_array_equal(data.xfrc_applied[env.model.body(obj['name']).id], 0)

    def test_waiting_gripper_does_not_push_without_partner(self):
        env = self.env
        env.body_phase[:] = 1
        env.body_approach_stage[:, 0] = 4
        env.body_approach_stage[:, 1] = 1
        for world, data in enumerate(env.native_data):
            adr = env.robot_qadr_np[0]
            center = data.qpos[env.object_qadr_np[0]:env.object_qadr_np[0]+2]
            data.qpos[adr:adr+2] = center + [float(env.object_radius[0])+.042, 0]
            data.qpos[adr+3:adr+7] = [math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)]
            data.qvel[:] = 0
            mujoco.mj_forward(env.model, data)
            env.qpos[world] = torch.as_tensor(data.qpos)
            env.qvel[world] = torch.as_tensor(data.qvel)
        waiting = env.teacher_action()[:, 0]
        self.assertLess(float(waiting[:, :2].abs().max()), 1e-5)
        self.assertTrue(torch.all(waiting[:, 2:] == -1))
        env.body_approach_stage[:, :2] = 5
        pressing = env.teacher_action()[:, 0, :2]
        self.assertGreater(float(pressing.mean()), .1)

    def test_outside_gripper_leaves_before_its_waiting_partner(self):
        env = self.env
        env.body_phase[:] = 1
        command = env.teacher_action()
        self.assertTrue(torch.all(command[:, [0, 2], :2] == 0))
        self.assertTrue(torch.all(command[:, :4, 3] == -1))
        self.assertGreater(float(command[:, [1, 3], :2].mean()), 0)
        env.step(command)
        self.assertTrue(torch.all(env.body_approach_stage[:, [0, 2]] == -1))
        env.body_approach_stage[:, [1, 3]] = 1
        env.step(env.teacher_action())
        self.assertTrue(torch.all(env.body_approach_stage[:, [0, 2]] == 2))


if __name__ == '__main__':
    unittest.main()
