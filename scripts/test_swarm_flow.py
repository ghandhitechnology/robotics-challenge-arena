#!/usr/bin/env python3
"""Check start fit, steering convention, and physical-connectivity reporting."""
from pathlib import Path
import math
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.swarm_flow import (body_telemetry, compact_packing, flow_actions,
                                     local_velocity_field, velocity_actions)


class SwarmFlowTests(unittest.TestCase):
    def test_full_footprint_and_shoulders_fit_start(self):
        poses = compact_packing()
        self.assertEqual(poses.shape, (40, 4))
        lower = poses[:, :2] - [.028, .012]
        upper = poses[:, :2] + [.027, .012]
        self.assertTrue(np.all(lower >= [.863, .701]))
        self.assertTrue(np.all(upper <= [1.143, 1.181]))
        # Adjacent staggered rows align opposite shoulder sites at 0.8 mm gap.
        delta = poses[4, :2] - poses[0, :2]
        np.testing.assert_allclose(delta, [.028, .0248])

    def test_forward_and_turn_use_runtime_wheel_sign(self):
        straight = velocity_actions([[-.04, 0]], [math.pi / 2])[0]
        self.assertGreater(straight[0], 0)
        self.assertAlmostEqual(straight[0], straight[1])
        turn = velocity_actions([[-.04, 0]], [0])[0]
        self.assertLess(turn[0], turn[1])
        stopped = velocity_actions([[0, 0]], [math.pi / 2])[0]
        np.testing.assert_allclose(stopped[:2], 0, atol=1e-12)

    def test_packing_flow_is_finite_bounded_and_permutation_equivariant(self):
        poses = compact_packing()
        pos, yaw = poses[:, :2], poses[:, 3]
        vel = np.zeros_like(pos)
        actions, desired = flow_actions(pos, vel, yaw, [.72, .885])
        self.assertTrue(np.isfinite(actions).all())
        self.assertLessEqual(abs(actions).max(), 1.)
        self.assertTrue(np.all(desired[:, 0] < 0))
        self.assertTrue(np.all(np.linalg.norm(desired, axis=-1) <= .04000001))
        permutation = np.random.default_rng(8).permutation(40)
        other, _ = flow_actions(pos[permutation], vel[permutation], yaw[permutation], [.72, .885])
        np.testing.assert_allclose(other, actions[permutation], atol=1e-12)

    def test_backup_preserves_heading_and_magnetic_connection(self):
        links = np.array([[False, True], [True, False]])
        desired = np.array([[.04, 0.], [.04, 0.]])
        yaw = np.full(2, math.pi / 2)
        reverse = velocity_actions(desired, yaw, links=links)
        self.assertTrue(np.all(reverse[:, :2] < 0))
        np.testing.assert_allclose(reverse[:, 0], reverse[:, 1], atol=1e-12)
        self.assertTrue(np.all(reverse[:, 3] == 1))
        forward_only = velocity_actions(desired, yaw, links=links, allow_reverse=False)
        self.assertTrue(np.all(forward_only[:, 0] != forward_only[:, 1]))
        self.assertTrue(np.all(forward_only[:, 3] == -1))

    def test_reverse_steers_toward_the_same_world_velocity(self):
        # A desired vector slightly right of the robot's rear needs a small
        # positive yaw change while both wheels drive backward.
        action = velocity_actions([[.01, -.04]], [0.])[0]
        self.assertLess(action[:2].mean(), 0.)
        self.assertGreater(action[1] - action[0], 0.)
        poses = compact_packing(1)
        reverse, _ = flow_actions(poses[:, :2], np.zeros((1, 2)), poses[:, 3],
                                  [1.05, .77], allow_reverse=True)
        forward, _ = flow_actions(poses[:, :2], np.zeros((1, 2)), poses[:, 3],
                                  [1.05, .77], allow_reverse=False)
        self.assertLess(reverse[0, :2].mean(), 0.)
        self.assertGreater(abs(forward[0, 0] - forward[0, 1]), .01)

    def test_close_rectangle_deflects_flow_outward(self):
        kwargs = dict(positions=np.array([[.55, .5]]), velocities=np.zeros((1, 2)),
                      yaw=np.array([0.]), waypoint=np.array([.55, .7]))
        clear = local_velocity_field(**kwargs)
        blocked = local_velocity_field(**kwargs, obstacles=[(.5, .505, .6, .56)])
        self.assertLess(blocked[0, 1], clear[0, 1])
        self.assertGreater(abs(blocked[0, 0]), 0.)

    def test_graph_reporting_requires_actual_links_and_forces(self):
        pos = np.array([[.4, .4], [.425, .4], [.45, .4], [.475, .4]])
        links = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], bool)
        force = links.astype(float) * .01
        force[1, 2] = force[2, 1] = .001
        report = body_telemetry(pos, np.zeros_like(pos), links, link_forces=force)
        self.assertEqual(report['component_sizes'], [3, 1])
        self.assertEqual(report['largest_component_fraction'], .75)
        self.assertEqual(report['force_bearing_largest_fraction'], .5)
        self.assertEqual(report['isolated_count'], 1)
        self.assertEqual(report['force_bearing_edge_count'], 1)
        self.assertEqual(body_telemetry(pos, np.zeros_like(pos), np.zeros_like(links))['component_count'], 4)

    def test_release_commands_do_not_modify_graph_or_state(self):
        links = np.array([[False, True], [True, False]])
        old = links.copy()
        desired = np.array([[-.04, 0.], [-.04, 0.]])
        actions = velocity_actions(desired, [0., 0.], links=links)
        self.assertTrue(np.all(actions[:, 3] == -1))
        np.testing.assert_array_equal(links, old)
        np.testing.assert_array_equal(desired, [[-.04, 0.], [-.04, 0.]])

    def test_tiny_cohesion_corrections_keep_docks_enabled(self):
        links = np.array([[False, True], [True, False]])
        tiny = velocity_actions([[-.0005, 0.], [-.0005, 0.]], [0., 0.], links=links)
        large = velocity_actions([[-.035, 0.], [-.035, 0.]], [0., 0.], links=links)
        self.assertTrue(np.all(tiny[:, 3] == 1))
        self.assertTrue(np.all(large[:, 3] == -1))
        self.assertLess(abs(tiny[:, :2]).max(), .005)
        self.assertGreater(abs(large[:, :2]).max(), .05)


if __name__ == '__main__':
    unittest.main()
