#!/usr/bin/env python3
"""Check start fit, steering convention, and physical-connectivity reporting."""
from pathlib import Path
import math
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.swarm_flow import (DockingState, FlowConfig, body_telemetry, compact_packing, flow_actions,
                                     local_velocity_field, velocity_actions, _clear_docking_poses, _docking_route)


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

    def test_docking_reads_do_not_advance_and_ids_survive_subset_reordering(self):
        pos = np.array([[.5, .5], [.5248, .528], [.5496, .5], [.59, .50]])
        yaw = np.array([0., 0., 0., .3])
        links = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], bool)
        ids = np.array([5, 7, 9, 12])
        state = DockingState(40)
        state.update(pos, yaw, links, module_ids=ids)
        first = state.guidance(pos, yaw, module_ids=ids)
        self.assertEqual(first['stage'][3], 0)
        self.assertTrue(first['active'][3])
        target = first['positions'][3].copy()
        # Arriving at the staging pose must not advance the state during reads.
        moved = pos.copy()
        moved[3] = target
        for _ in range(3):
            _, _, read = flow_actions(moved, np.zeros_like(pos), yaw, pos.mean(0),
                                     links=links, docking=True, docking_state=state,
                                     module_ids=ids, return_guidance=True)
            self.assertEqual(read['stage'][3], 0)
        state.update(moved, yaw, links, module_ids=ids)
        self.assertEqual(state.guidance(moved, yaw, module_ids=ids)['stage'][3], 1)
        order = np.array([3, 1, 0, 2])
        shuffled = state.guidance(moved[order], yaw[order], module_ids=ids[order])
        self.assertEqual(shuffled['stage'][0], 1)
        np.testing.assert_allclose(shuffled['final_positions'][0], first['final_positions'][3])
        state.reset()
        self.assertFalse(state.guidance(moved, yaw, module_ids=ids)['active'].any())

    def test_docking_port_latches_until_measured_physical_attachment(self):
        pos = np.array([[.5, .5], [.5248, .528], [.5496, .5], [.59, .50]])
        yaw = np.array([0., 0., 0., .3])
        links = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], bool)
        state = DockingState(4)
        state.update(pos, yaw, links)
        initial = state.guidance(pos, yaw)
        pos[3] += [.002, .004]
        yaw[3] += .5
        state.update(pos, yaw, links)
        updated = state.guidance(pos, yaw)
        self.assertEqual(updated['neighbor'][3], initial['neighbor'][3])
        self.assertEqual(updated['port'][3], initial['port'][3])
        np.testing.assert_allclose(updated['final_positions'][3], initial['final_positions'][3])
        neighbor = initial['neighbor'][3]
        links[3, neighbor] = links[neighbor, 3] = True
        state.update(pos, yaw, links)
        self.assertFalse(state.guidance(pos, yaw)['active'][3])

    def test_linked_velocity_exchange_reduces_local_steering_disagreement(self):
        poses = compact_packing(4)
        pos, yaw = poses[:, :2], poses[:, 3]
        graph = np.ones((4, 4), bool)
        np.fill_diagonal(graph, False)
        velocity = np.array([[0., .03], [0., 0.], [0., -.03], [0., 0.]])
        raw = local_velocity_field(pos, velocity, yaw, [.72, .78], links=graph,
                                   config=FlowConfig(consensus_steps=0))
        shared = local_velocity_field(pos, velocity, yaw, [.72, .78], links=graph)
        self.assertLess(np.ptp(shared[:, 1]), np.ptp(raw[:, 1])*.01)
        # Disconnected robots retain their own local navigation directive.
        separate = local_velocity_field(pos, velocity, yaw, [.72, .78])
        separate_raw = local_velocity_field(pos, velocity, yaw, [.72, .78],
                                             config=FlowConfig(consensus_steps=0))
        np.testing.assert_allclose(separate, separate_raw)

    def test_detached_pair_releases_while_largest_core_stays_enabled(self):
        pos = np.array([[.5, .5], [.5248, .528], [.5496, .5], [.63, .553], [.6548, .553]])
        yaw = np.zeros(5)
        graph = np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]], bool)
        state = DockingState(5)
        state.update(pos, yaw, graph)
        before = state.release_steps.copy()
        actions, _, guide = flow_actions(pos, np.zeros_like(pos), yaw, pos.mean(0),
                                        links=graph, docking=True, docking_state=state,
                                        return_guidance=True)
        np.testing.assert_array_equal(guide['release'], [False, False, False, True, True])
        np.testing.assert_array_equal(guide['stage'][3:], [-3, -3])
        self.assertTrue(np.all(actions[:3, 3] == 1))
        self.assertTrue(np.all(actions[3:, 3] == -1))
        self.assertLess(actions[3, :2].mean(), 0.)
        self.assertGreater(actions[4, :2].mean(), 0.)
        np.testing.assert_array_equal(state.release_steps, before)

    def test_docking_does_not_release_a_large_detached_body(self):
        poses = compact_packing(12)
        graph = np.zeros((12, 12), bool)
        graph[:7, :7] = True
        graph[7:, 7:] = True
        np.fill_diagonal(graph, False)
        state = DockingState(12)
        state.update(poses[:, :2], poses[:, 3], graph)
        self.assertFalse(state.guidance(poses[:, :2], poses[:, 3])['release'].any())

    def test_clear_dock_endpoint_does_not_imply_a_clear_rolling_lane(self):
        core = np.array([[.5, .5], [.5248, .528], [.5248, .584]])
        yaw = np.zeros(3)
        target = np.array([[.5248, .472]])
        self.assertTrue(_clear_docking_poses(target, np.zeros(1), core, yaw, [])[0])
        # The north approach crosses two occupied chassis even though its final
        # pose is free; the south approach reaches the same dock unobstructed.
        offsets = np.array([.208, -.072])
        swept = target+np.column_stack((np.zeros(2), offsets/2))
        clear = _clear_docking_poses(swept, np.zeros(2), core, yaw, [],
                                     half_lengths=.0276+np.abs(offsets)/2)
        np.testing.assert_array_equal(clear, [False, True])

    def test_reserved_port_is_replanned_if_another_module_occupies_it(self):
        pos = np.array([[.5, .5], [.5248, .528], [.5496, .5], [.59, .50], [.2, .2]])
        yaw = np.array([0., 0., 0., .3, 0.])
        graph = np.zeros((5, 5), bool)
        graph[0, 1] = graph[1, 0] = graph[1, 2] = graph[2, 1] = True
        state = DockingState(5)
        state.update(pos, yaw, graph)
        initial = state.guidance(pos, yaw)
        owner, port = initial['neighbor'][3], initial['port'][3]
        self.assertGreaterEqual(owner, 0)
        pos[4], yaw[4] = initial['final_positions'][3], initial['yaw'][3]
        graph[owner, 4] = graph[4, owner] = True
        state.update(pos, yaw, graph)
        updated = state.guidance(pos, yaw)
        self.assertFalse(updated['neighbor'][3] == owner and updated['port'][3] == port)

    def test_unplanned_arrival_holds_until_a_docking_lane_is_assigned(self):
        pos = np.array([[.5, .5], [.5248, .528], [.5496, .5], [.95, .9]])
        yaw = np.array([0., 0., 0., .3])
        graph = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], bool)
        state = DockingState(4)
        state.update(pos, yaw, graph)
        actions, _, guide = flow_actions(pos, np.zeros_like(pos), yaw, [.5, .5],
                                        links=graph, docking=True, docking_state=state,
                                        return_guidance=True)
        self.assertTrue(state.waiting[3])
        self.assertEqual(guide['stage'][3], -2)
        np.testing.assert_allclose(actions[3], [0., 0., -1., 1.])
        pos[3] = [.59, .5]
        state.update(pos, yaw, graph)
        self.assertGreaterEqual(state.stage[3], 0)
        self.assertFalse(state.waiting[3])
        state.update(pos, yaw, graph)
        self.assertFalse(state.waiting[3])

    def test_docking_route_preserves_a_clear_target_inside_the_core_bounding_box(self):
        pos = np.array([[.5, .5], [.55, .55], [.6, .6]])
        yaw = np.full(3, math.pi/4)
        target = np.array([.5, .6])
        route = _docking_route(np.array([.4, .6]), target, pos, yaw)
        np.testing.assert_allclose(route, target)
        permutation = np.array([2, 0, 1])
        np.testing.assert_allclose(_docking_route(np.array([.4, .6]), target,
                                                pos[permutation], yaw[permutation]), route)

    def test_docking_route_advances_around_payload_corners(self):
        start, target = np.array([.3, .5]), np.array([.7, .5])
        pos, yaw = np.array([[.9, .9]]), np.zeros(1)
        obstacles = [[.48, .47, .52, .53]]
        first = _docking_route(start, target, pos, yaw, obstacles)
        second = _docking_route(first, target, pos, yaw, obstacles)
        self.assertAlmostEqual(first[0], .46)
        self.assertAlmostEqual(second[0], .54)
        self.assertAlmostEqual(abs(first[1]-.5), .05)
        self.assertAlmostEqual(second[1], first[1])
        np.testing.assert_allclose(_docking_route(second, target, pos, yaw, obstacles), target)

    def test_docking_route_escapes_overlapping_arrival_clearances_together(self):
        start, target = np.array([.49, .9301]), np.array([.5745, 1.0181])
        pos, yaw = np.array([[.7, .885]]), np.zeros(1)
        others = np.array([[.4776, .8866], [.4709, .9897]])
        headings = np.array([-.5635, .6476])
        next_point = _docking_route(start, target, pos, yaw,
                                    other_positions=others, other_yaw=headings)
        self.assertGreater(next_point[0], start[0]+.01)
        self.assertGreater(next_point[1], start[1])
        forward = np.column_stack((-np.sin(headings), np.cos(headings)))
        right = np.column_stack((np.cos(headings), np.sin(headings)))
        delta = next_point-others-.0005*forward
        gap = np.maximum(np.abs(np.sum(delta*right, axis=-1))-.0121,
                         np.abs(np.sum(delta*forward, axis=-1))-.0276)
        self.assertTrue(np.all(gap >= .0359))

    def test_docking_staging_reservation_uses_the_other_clear_lane(self):
        from arena_mujoco.swarm_flow import docking_targets
        pos = np.array([[.5, .5], [.5248, .528], [.5496, .5], [.59, .5]])
        yaw = np.array([0., 0., 0., .3])
        graph = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], bool)
        first = docking_targets(pos, yaw, graph)
        forward = np.array([-np.sin(first['yaw'][3]), np.cos(first['yaw'][3])])
        reserved = first['positions'][3]+first['staging_offset'][3]*forward
        second = docking_targets(pos, yaw, graph, reserved_staging=[reserved])
        self.assertTrue(second['active'][3])
        forward = np.array([-np.sin(second['yaw'][3]), np.cos(second['yaw'][3])])
        alternate = second['positions'][3]+second['staging_offset'][3]*forward
        self.assertGreaterEqual(np.linalg.norm(alternate-reserved), .07)


if __name__ == '__main__':
    unittest.main()
