#!/usr/bin/env python3
"""Check the body environment's physical-action and reset boundaries."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

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
        self.assertEqual(tuple(obs['local'].shape), (2, 40, 42))
        self.assertEqual(tuple(obs['neighbors'].shape), (2, 40, 6, 12))
        self.assertEqual(tuple(obs['global'].shape), (2, 92))
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

    def test_carry_teams_wait_until_both_payloads_are_supported(self):
        env = self.env
        env.body_phase[:] = 1
        env.phase[:, 0] = 2
        waiting = env.teacher_action()[:, :2, :2].mean(-1)
        torch.testing.assert_close(waiting, torch.full_like(waiting, .15))
        env.body_phase[:] = 2
        carrying = env.teacher_action()[:, :2, :2].mean(-1)
        torch.testing.assert_close(carrying, torch.tensor([[.25, .05], [.25, .05]]))

    def test_docking_updates_once_and_observation_reads_preserve_plans(self):
        env = self.env
        plans = env.body_docking
        with patch.object(plans[0], 'update', wraps=plans[0].update) as update:
            env.step(env.teacher_action())
            self.assertEqual(update.call_count, 1)
            # A pending alignment plan must survive repeated policy reads.
            plans[0].stage[0] = 1
            plans[0].neighbor[0] = 4
            plans[0].port[0] = 0
            plans[0].own_port[0] = 2
            saved = {key: value.copy() for key, value in vars(plans[0]).items()
                     if isinstance(value, np.ndarray)}
            first = env._observation()
            self.assertAlmostEqual(float(first['local'][0, 0, 41]), 1/3, places=6)
            env.teacher_action()
            second = env._observation()
            self.assertEqual(update.call_count, 1)
            for key, expected in saved.items():
                np.testing.assert_array_equal(getattr(plans[0], key), expected)
            for key in first:
                torch.testing.assert_close(first[key], second[key])
        plans[1].stage[0] = 2
        env.reset_done(torch.tensor([True, False]))
        self.assertTrue(np.all(plans[0].stage == -1))
        self.assertEqual(plans[1].stage[0], 2)

    def test_programmed_control_stages_are_visible_to_the_actor(self):
        env = self.env
        env.body_approach_stage[:, 0] = 3
        before = env._observation()['local']
        env.body_approach_stage[:, 0] = 4
        env.body_docking[0].release_steps[4] = 30
        after = env._observation()['local']
        self.assertAlmostEqual(float(before[0, 0, 40]), 3/5, places=6)
        self.assertAlmostEqual(float(after[0, 0, 40]), 4/5, places=6)
        self.assertEqual(float(after[0, 4, 40]), 0.)
        self.assertEqual(float(after[0, 4, 41]), -1.)


if __name__ == '__main__':
    unittest.main()
