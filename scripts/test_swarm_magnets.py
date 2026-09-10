#!/usr/bin/env python3
"""Check docking forces, conservation, native joining and wheel-driven release."""
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.swarm_robot import add_swarm_robots
from arena_mujoco.swarm_magnets import MagneticCoupling, PARAMETERS, add_magnetic_docks


def scene(gap=.0008, floor=True, yaw=0., timestep=.001, poses=None):
    root = ET.fromstring(f'<mujoco><option timestep="{timestep}" integrator="implicitfast" gravity="0 0 {-9.81 if floor else 0}" cone="elliptic"/><worldbody/></mujoco>')
    if floor:
        ET.SubElement(root.find('worldbody'), 'geom', name='floor', type='plane',
                      size='1 1 .1', contype='1', conaffinity='7', friction='.85 .00001 .000001')
    if poses is None:
        poses = [(-(.024+gap)/2, 0, .0072, 0), ((.024+gap)/2, 0, .0072, yaw)]
    specs = add_swarm_robots(root, len(poses), poses)
    add_magnetic_docks(root, specs)
    # The payload is deliberately present to verify that coupling never writes
    # its applied forces, even when it has an unrelated pre-existing load.
    body = ET.SubElement(root.find('worldbody'), 'body', name='payload', pos='.3 .3 .02')
    ET.SubElement(body, 'freejoint')
    ET.SubElement(body, 'geom', type='sphere', size='.01', mass='.1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    coupling = MagneticCoupling(model, specs)
    return model, data, coupling, specs


def run(model, data, coupling, specs, seconds, enabled=(True, True), speeds=None, pull_n=0.):
    motors = np.array([[model.actuator(n).id for n in s['motor_names']] for s in specs])
    dofs = np.array([[model.joint(n).dofadr[0] for n in s['wheel_joints']] for s in specs])
    for _ in range(round(seconds/model.opt.timestep)):
        mujoco.mj_step1(model, data)
        velocity = data.qvel[dofs]
        desired = np.zeros((2, 2)) if speeds is None else -20*np.asarray(speeds)
        request = .00015*(desired-velocity)
        limit = np.where(request*velocity > 0, .002*np.maximum(0, 1-np.abs(velocity)/26.1799388), .002)
        data.ctrl[motors] = np.clip(request, -limit, limit)
        coupling.apply(data, enabled)
        data.xfrc_applied[coupling.robot_bodies[0], 0] -= pull_n
        data.xfrc_applied[coupling.robot_bodies[1], 0] += pull_n
        mujoco.mj_step2(model, data)
    mujoco.mj_forward(model, data)
    coupling.apply(data, enabled)
    if any(w.number for w in data.warning):
        raise AssertionError('MuJoCo warning during magnetic test')


def articulated_scene():
    root = ET.fromstring('<mujoco><option gravity="0 0 0"/><worldbody/></mujoco>')
    world = root.find('worldbody')
    specs = []
    for index, side in enumerate((1, -1)):
        name = f'module_{index}'
        body = ET.SubElement(world, 'body', name=name, pos=f'{-side*.035} 0 0')
        ET.SubElement(body, 'freejoint', name=name+'_free')
        ET.SubElement(body, 'geom', type='sphere', size='.006', mass='.02')
        lobe = ET.SubElement(body, 'body', name=name+'_lobe', pos=f'{side*.02} 0 0')
        ET.SubElement(lobe, 'joint', name=name+'_hinge', axis='0 1 0')
        ET.SubElement(lobe, 'geom', type='sphere', size='.007', mass='.01')
        ET.SubElement(lobe, 'site', name=name+'_port', pos=f'{side*.0146} 0 .004',
                      quat=f'.707106781187 0 {side*.707106781187} 0')
        specs.append(dict(name=name, magnetic_sites=[name+'_port']))
    payload = ET.SubElement(world, 'body', name='payload', pos='.3 0 0')
    ET.SubElement(payload, 'freejoint')
    ET.SubElement(payload, 'geom', type='sphere', size='.01', mass='.1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    data.qpos[model.joint('module_0_hinge').qposadr[0]] = .015
    data.qpos[model.joint('module_1_hinge').qposadr[0]] = .015
    mujoco.mj_forward(model, data)
    return model, data, MagneticCoupling(model, specs), specs


class MagneticTests(unittest.TestCase):
    def test_articulated_ports_apply_balanced_wrenches_to_their_lobes(self):
        model, data, coupling, _ = articulated_scene()
        untouched = np.setdiff1d(np.arange(model.nbody), coupling.force_bodies)
        data.xfrc_applied[untouched] = np.arange(6) + .25
        coupling.apply(data, [True, True])
        self.assertTrue(coupling.interacting_pairs)
        bodies = coupling.force_bodies
        loads = data.xfrc_applied[bodies]
        np.testing.assert_allclose(loads[:, :3].sum(0), 0, atol=1e-14)
        moment = loads[:, 3:] + np.cross(data.xipos[bodies], loads[:, :3])
        np.testing.assert_allclose(moment.sum(0), 0, atol=1e-14)
        np.testing.assert_array_equal(data.xfrc_applied[untouched],
                                     np.tile(np.arange(6)+.25, (len(untouched), 1)))
        coupling.apply(data, [False, True])
        self.assertFalse(np.any(data.xfrc_applied[bodies]))

    def test_articulated_port_potential_gradient_matches_hinge_torque(self):
        model, data, coupling, _ = articulated_scene()
        coupling.apply(data, [True, True])
        generalized = np.zeros(model.nv)
        for body in coupling.force_bodies:
            mujoco.mj_applyFT(model, data, data.xfrc_applied[body, :3],
                             data.xfrc_applied[body, 3:], data.xipos[body],
                             int(body), generalized)
        initial = data.qpos.copy()
        for index in range(2):
            joint = model.joint(f'module_{index}_hinge')
            adr, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
            self.assertGreater(abs(generalized[dof]), 1e-5)
            energies = []
            for sign in (-1, 1):
                data.qpos[:] = initial
                data.qpos[adr] += sign*1e-7
                mujoco.mj_forward(model, data)
                coupling.reset()
                coupling.apply(data, [True, True])
                energies.append(coupling.last_potential_j)
            self.assertAlmostEqual(-(energies[1]-energies[0])/2e-7,
                                   generalized[dof], delta=2e-8)

    def test_articulated_broadphase_tracks_moving_ports(self):
        class AllPairsCoupling(MagneticCoupling):
            def _candidate_ports(self, data):
                return self._pair_a, self._pair_b

        model, data, fast, specs = articulated_scene()
        reference = AllPairsCoupling(model, specs)
        initial = data.qpos.copy()
        tested_interactions = 0
        for angle in np.linspace(-1.2, 1.2, 49):
            data.qpos[:] = initial
            for index in range(2):
                data.qpos[model.joint(f'module_{index}_hinge').qposadr[0]] = angle
            mujoco.mj_forward(model, data)
            # Translate the second root to retain a small face gap at each
            # hinge angle. These are broadphase fixtures, not a motion test.
            normal = data.site_xmat[fast.site_ids[0, 0]].reshape(3, 3)[:, 2]
            offset = (data.site_xpos[fast.site_ids[0, 0]] + .0008*normal
                      - data.site_xpos[fast.site_ids[1, 0]])
            adr = model.joint('module_1_free').qposadr[0]
            data.qpos[adr:adr+3] += offset
            mujoco.mj_forward(model, data)
            fast.apply(data, [True, True])
            wrench = data.xfrc_applied.copy()
            reference.apply(data, [True, True])
            np.testing.assert_array_equal(data.xfrc_applied, wrench)
            self.assertEqual(fast.interacting_pairs, reference.interacting_pairs)
            tested_interactions += len(fast.interacting_pairs)
        self.assertEqual(tested_interactions, 49)

    def test_ports_cannot_belong_to_another_module_or_payload(self):
        model, _, _, specs = articulated_scene()
        specs[0]['magnetic_sites'] = specs[1]['magnetic_sites']
        with self.assertRaisesRegex(ValueError, 'module root or its descendants'):
            MagneticCoupling(model, specs)

    def test_broadphase_matches_all_pairs_through_three_dimensional_motion(self):
        from arena_mujoco.swarm_flow import compact_packing

        class AllPairsCoupling(MagneticCoupling):
            def _candidate_ports(self, data):
                return self._pair_a, self._pair_b

        model, data, fast, specs = scene(floor=False, poses=compact_packing())
        reference = AllPairsCoupling(model, specs)
        initial = data.qpos.copy()
        addresses = np.array([model.joint(s['freejoint']).qposadr[0] for s in specs])
        rng = np.random.default_rng(6714)
        tested_interactions = 0
        for sample in range(36):
            data.qpos[:] = initial
            if sample % 3 == 1:
                # Small displacements retain/release existing partners. An
                # arbitrary rigid 3D rotation checks the orientation-free bound.
                offsets = rng.uniform(-.0015, .0015, (40, 3))
                rotation = rng.normal(size=4)
                rotation /= np.linalg.norm(rotation)
                matrix = np.empty(9)
                mujoco.mju_quat2Mat(matrix, rotation)
                for index, adr in enumerate(addresses):
                    data.qpos[adr:adr+3] = matrix.reshape(3, 3) @ (initial[adr:adr+3]+offsets[index])
                    mujoco.mju_mulQuat(data.qpos[adr+3:adr+7], rotation, initial[adr+3:adr+7])
            elif sample % 3 == 2:
                for adr in addresses:
                    data.qpos[adr:adr+3] = rng.uniform(-.055, .055, 3)
                    rotation = rng.normal(size=4)
                    data.qpos[adr+3:adr+7] = rotation/np.linalg.norm(rotation)
            mujoco.mj_forward(model, data)
            enabled = rng.random((40, 4)) > .1
            enabled[:] = True if sample == 0 else enabled
            fast.apply(data, enabled)
            wrench = data.xfrc_applied.copy()
            reference.apply(data, enabled)
            np.testing.assert_array_equal(data.xfrc_applied, wrench)
            self.assertEqual(fast.interacting_pairs, reference.interacting_pairs)
            self.assertEqual(fast.links, reference.links)
            self.assertEqual(fast.last_potential_j, reference.last_potential_j)
            self.assertEqual(fast.last_peak_force_n, reference.last_peak_force_n)
            np.testing.assert_array_equal(fast.pair_force_vectors_n, reference.pair_force_vectors_n)
            np.testing.assert_array_equal(fast.pair_force_magnitudes_n, reference.pair_force_magnitudes_n)
            tested_interactions += len(fast.interacting_pairs)
        self.assertGreater(tested_interactions, 200)
        data.qpos[:] = initial
        mujoco.mj_forward(model, data)
        self.assertLess(len(fast._candidate_ports(data)[0]), len(fast._pair_a)//5)

    def test_internal_wrenches_are_balanced_and_payload_untouched(self):
        model, data, coupling, _ = scene(yaw=.025, floor=False)
        payload = model.body('payload').id
        data.xfrc_applied[payload] = np.arange(6) + .25
        coupling.apply(data, [True, True])
        self.assertTrue(coupling.interacting_pairs)
        loads = data.xfrc_applied[coupling.robot_bodies]
        np.testing.assert_allclose(loads[:, :3].sum(0), 0, atol=1e-14)
        moment = loads[:, 3:] + np.cross(data.xipos[coupling.robot_bodies], loads[:, :3])
        np.testing.assert_allclose(moment.sum(0), 0, atol=1e-14)
        np.testing.assert_array_equal(data.xfrc_applied[payload], np.arange(6)+.25)

    def test_no_long_range_force_or_disabled_connection(self):
        model, data, coupling, _ = scene(gap=.01)
        coupling.apply(data, [True, True])
        self.assertFalse(coupling.interacting_pairs)
        self.assertFalse(np.any(data.xfrc_applied[coupling.robot_bodies]))
        model, data, coupling, _ = scene()
        coupling.apply(data, [True, True])
        self.assertTrue(coupling.interacting_pairs)
        coupling.apply(data, [True, False])
        self.assertFalse(coupling.interacting_pairs)
        self.assertFalse(coupling.graph().any())
        self.assertFalse(np.any(data.xfrc_applied[coupling.robot_bodies]))

    def test_force_bound_and_zero_at_range_boundary(self):
        model, data, coupling, specs = scene(floor=False)
        adr = model.joint(specs[1]['freejoint']).qposadr[0]
        x0 = data.qpos[adr]
        measured = []
        for gap in np.linspace(0, PARAMETERS.capture_gap_m, 100):
            data.qpos[adr] = x0 + gap - .0008
            mujoco.mj_forward(model, data)
            coupling.reset()
            coupling.apply(data, [True, True])
            measured.append(coupling.last_peak_force_n)
        self.assertGreater(max(measured), .149)
        self.assertLessEqual(max(measured), PARAMETERS.peak_force_n + 1e-12)
        self.assertLess(measured[-1], 1e-12)

    def test_potential_gradient_matches_force_and_angular_wrench(self):
        model, data, coupling, specs = scene(floor=False, yaw=.015)
        joint = model.joint(specs[1]['freejoint'])
        adr, dof = joint.qposadr[0], joint.dofadr[0]
        initial = data.qpos.copy()
        coupling.apply(data, [True, True])
        body = coupling.robot_bodies[1]
        force = data.xfrc_applied[body, :3].copy()
        # qpos rotation moves the body around its origin, while applied wrench
        # is reported around its COM. Convert before comparing gradients.
        torque = (data.xfrc_applied[body, 3:] +
                  np.cross(data.xipos[body]-data.xpos[body], force))
        for axis in range(6):
            energies = []
            eps = 1e-7
            for sign in (-1, 1):
                data.qpos[:] = initial
                dq = np.zeros(model.nv)
                dq[dof+axis] = sign
                mujoco.mj_integratePos(model, data.qpos, dq, eps)
                mujoco.mj_forward(model, data)
                coupling.reset()
                coupling.apply(data, [True, True])
                energies.append(coupling.last_potential_j)
            gradient = -(energies[1]-energies[0])/(2*eps)
            # Free-joint angular velocity uses the body's local frame.
            expected = force[axis] if axis < 3 else (data.xmat[body].reshape(3, 3).T @ torque)[axis-3]
            self.assertAlmostEqual(gradient, expected, delta=2e-6)
        data.qpos[:] = initial

    def test_interaction_and_load_graphs_preserve_close_dock_definition(self):
        model, data, coupling, _ = scene(gap=.0008)
        coupling.apply(data, [True, True])
        self.assertFalse(coupling.graph().any())
        np.testing.assert_array_equal(coupling.contact_graph(), coupling.graph())
        self.assertTrue(coupling.interaction_graph()[0, 1])
        self.assertTrue(coupling.load_bearing_graph()[0, 1])
        self.assertFalse(coupling.load_bearing_graph(.31).any())
        with self.assertRaises(ValueError):
            coupling.load_bearing_graph(float("nan"))
        coupling.apply(data, [False, True])
        self.assertFalse(coupling.interaction_graph().any())
        self.assertFalse(coupling.load_bearing_graph().any())
        self.assertFalse(coupling.pair_force_vectors_n.any())
        self.assertFalse(coupling.pair_force_magnitudes_n.any())

    def test_pair_force_telemetry_matches_actual_module_loads(self):
        model, data, coupling, _ = scene(yaw=.025, floor=False)
        coupling.apply(data, [True, True])
        vectors = coupling.pair_force_vectors_n
        np.testing.assert_allclose(vectors, -vectors.transpose(1, 0, 2), atol=1e-14)
        np.testing.assert_allclose(vectors.sum(1), data.xfrc_applied[coupling.robot_bodies, :3], atol=1e-14)
        np.testing.assert_allclose(coupling.pair_force_magnitudes_n, np.linalg.norm(vectors, axis=-1))
        self.assertEqual(float(coupling.pair_force_magnitudes_n.trace()), 0.)
        endpoints = [port for pair in coupling.interacting_pairs for port in pair]
        self.assertEqual(len(endpoints), len(set(endpoints)))
        coupling.reset()
        self.assertFalse(coupling.pair_force_vectors_n.any())
        self.assertFalse(coupling.pair_force_magnitudes_n.any())

    def test_native_attraction_joins_and_release_removes_force(self):
        model, data, coupling, specs = scene()
        run(model, data, coupling, specs, .5)
        self.assertTrue(coupling.graph()[0, 1])
        before = data.qpos.copy()
        coupling.apply(data, [False, True])
        np.testing.assert_array_equal(before, data.qpos)
        self.assertFalse(coupling.graph().any())
        self.assertFalse(np.any(data.xfrc_applied[coupling.robot_bodies]))

    def test_native_overload_breaks_finite_links_on_floor(self):
        model, data, coupling, specs = scene(gap=.00005)
        run(model, data, coupling, specs, .1)
        self.assertTrue(coupling.graph()[0, 1])
        run(model, data, coupling, specs, .25, pull_n=.8)
        self.assertFalse(coupling.graph().any())
        separation = np.linalg.norm(data.xpos[coupling.robot_bodies[1]] - data.xpos[coupling.robot_bodies[0]])
        self.assertGreater(separation, .04)

    def test_engaged_ports_hold_opposing_wheels_before_release(self):
        model, data, coupling, specs = scene(gap=.00005)
        run(model, data, coupling, specs, .1)
        run(model, data, coupling, specs, 1.5, speeds=[[1, 1], [-1, -1]])
        self.assertTrue(coupling.graph()[0, 1])
        self.assertEqual(len(coupling.links), 2)
        run(model, data, coupling, specs, 1., enabled=[False, True], speeds=[[1, 1], [-1, -1]])
        self.assertFalse(coupling.interacting_pairs)
        separation = np.linalg.norm(data.xpos[coupling.robot_bodies[1]] - data.xpos[coupling.robot_bodies[0]])
        self.assertGreater(separation, .04)

    def test_port_release_allows_wheel_driven_shear(self):
        model, data, coupling, specs = scene(gap=.00005)
        run(model, data, coupling, specs, .1)
        self.assertTrue(coupling.graph()[0, 1])
        run(model, data, coupling, specs, 1., enabled=[False, True], speeds=[[.5, .5], [-.5, -.5]])
        self.assertFalse(coupling.interacting_pairs)
        relative_y = abs(data.xpos[coupling.robot_bodies[1], 1] - data.xpos[coupling.robot_bodies[0], 1])
        self.assertGreater(relative_y, .04)


if __name__ == '__main__':
    unittest.main()
