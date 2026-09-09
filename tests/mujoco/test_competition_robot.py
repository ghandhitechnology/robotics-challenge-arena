"""Physical grasp coupons for the competition robot's real contact geometry."""
import unittest
import xml.etree.ElementTree as ET
import math

import mujoco
import numpy as np

from arena_mujoco.competition_robot import add_competition_robot


def coupon(kind, lab_ring=False, forward_offset=0):
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep='.0002', integrator='implicitfast',
                  solver='CG', cone='pyramidal', iterations='80', tolerance='1e-9')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='1 1 .01',
                  friction='.7 .0001 .00001', solref='.001 1', solimp='.98 .999 .0001')
    meta = add_competition_robot(root, {})
    world.find("body[@name='competition_robot']").set('pos', '0 0 .020')
    dimensions = {'cylinder': ('cylinder', '.010 .010', .010, math.pi*.01**2*.02),
                  'kit': ('box', '.0125 .0125 .010', .010, .025*.025*.020),
                  'sample': ('cylinder', '.028 .0025', .0025, math.pi*.028**2*.005)}
    shape, size, half_height, volume = dimensions[kind]
    obj = ET.SubElement(world, 'body', name='object', pos=f'0 {.105+forward_offset} {half_height+.00002}')
    ET.SubElement(obj, 'freejoint', name='object_free')
    ET.SubElement(obj, 'geom', name='object_geom', type=shape, size=size,
                  mass=str(volume*600), friction='.5 .00005 .000005',
                  solref='.001 1', solimp='.98 .999 .0001')
    if lab_ring:
        asset = ET.SubElement(root, 'asset')
        for i in range(64):
            a, b = i*2*math.pi/64, (i+1)*2*math.pi/64
            vertices = [[r*math.cos(t), .305+r*math.sin(t), z]
                        for z in (0, .003) for r,t in ((.030,a),(.080,a),(.080,b),(.030,b))]
            name = f'ring_{i}'
            ET.SubElement(asset, 'mesh', name=name, vertex=' '.join(map(str,np.ravel(vertices))),
                          face='0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7')
            ET.SubElement(world, 'geom', name=name, type='mesh', mesh=name,
                          solref='.001 1', solimp='.98 .999 .0001')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data = mujoco.MjData(model)
    for joint in meta['jaw_joints']:
        data.qpos[model.joint(joint).qposadr[0]] = .025
    data.ctrl[model.actuator(meta['grip_motor']).id] = .025
    mujoco.mj_forward(model,data)
    return model,data,meta


def advance(model,data,meta,seconds,lift=None,jaw=None,target_y=None,target_yaw=None):
    targets = ((meta['lift_motor'],lift,meta['max_lift_speed_m_s']),
               (meta['grip_motor'],jaw,meta['max_jaw_speed_m_s']))
    for _ in range(round(seconds/model.opt.timestep)):
        for name,target,speed in targets:
            if target is not None:
                actuator = model.actuator(name).id
                data.ctrl[actuator] += np.clip(target-data.ctrl[actuator],
                                               -speed*model.opt.timestep,speed*model.opt.timestep)
        # Wheel motor feedback prevents free rolling during the bench coupon.
        for side,(joint,motor) in enumerate(zip(meta['wheel_joints'],meta['motor_names'])):
            velocity = data.qvel[model.joint(joint).dofadr[0]]
            forward = 0 if target_y is None else np.clip(5*(target_y-data.xpos[model.body('competition_robot').id,1]),-.15,.15)
            rotation = data.body('competition_robot').xmat.reshape(3,3)
            yaw = math.atan2(rotation[1,0],rotation[0,0])
            error = 0 if target_yaw is None else (target_yaw-yaw+math.pi)%(2*math.pi)-math.pi
            turn = np.clip(4*error,-1,1)
            wanted = -(forward+(-1 if side==0 else 1)*turn*meta['wheel_track_m']/2)/meta['wheel_radius_m']
            data.ctrl[model.actuator(motor).id] = np.clip(.008*(wanted-velocity),-.025,.025)
        mujoco.mj_step(model,data)
        if any(w.number for w in data.warning):
            raise AssertionError('Native MuJoCo warning during physical grasp')
    mujoco.mj_forward(model,data)


class CompetitionRobotTests(unittest.TestCase):
    def test_four_motors_mass_and_symmetric_jaws(self):
        model,data,meta = coupon('cylinder')
        robot_id = model.body('competition_robot').id
        self.assertEqual(model.nu,4)
        self.assertAlmostEqual(data.subtree_com[robot_id,0],0)
        self.assertAlmostEqual(model.body_subtreemass[robot_id],.800)
        self.assertEqual(model.neq,1)
        self.assertEqual(int(model.eq_type[0]),int(mujoco.mjtEq.mjEQ_JOINT))
        advance(model,data,meta,.3,jaw=.015)
        positions = [data.qpos[model.joint(j).qposadr[0]] for j in meta['jaw_joints']]
        self.assertLess(abs(positions[0]-positions[1]),.00005)

    def test_cylinder_kit_and_sample_lift_and_release_by_contact(self):
        for kind in ('cylinder','kit','sample'):
            with self.subTest(kind=kind):
                model,data,meta = coupon(kind)
                obj = model.body('object').id
                initial_height = data.xpos[obj,2]
                advance(model,data,meta,.15)
                advance(model,data,meta,.7,jaw=0)
                advance(model,data,meta,.9,lift=.050)
                self.assertGreater(data.xpos[obj,2]-initial_height,.035)
                self.assertLess(np.linalg.norm(data.xpos[obj,:2]-data.site('gripper_center').xpos[:2]),.008)
                advance(model,data,meta,.7,jaw=.025)
                self.assertLess(data.xpos[obj,2]-initial_height,.001)
                self.assertFalse(np.any(model.geom_adhesion))
                self.assertFalse(np.any(data.qfrc_applied))

    def test_off_center_kit_survives_reverse_drive_and_quarter_turn(self):
        model,data,meta = coupon('kit',forward_offset=.010)
        model.opt.timestep=.001
        advance(model,data,meta,.15)
        advance(model,data,meta,.7,jaw=0)
        advance(model,data,meta,.9,lift=.050)
        advance(model,data,meta,2.0,target_y=-.120)
        advance(model,data,meta,3.0,target_yaw=math.pi/2)
        self.assertGreater(data.body('object').xpos[2],.035)
        self.assertLess(np.linalg.norm(data.body('object').xpos[:2]-data.site('gripper_center').xpos[:2]),.015)
        rotation = data.body('competition_robot').xmat.reshape(3,3)
        self.assertAlmostEqual(math.atan2(rotation[1,0],rotation[0,0]),math.pi/2,delta=.01)

    def test_thin_sample_stays_in_pads_for_thirty_seconds(self):
        model,data,meta = coupon('sample')
        model.opt.timestep = .001
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.impratio = 100
        advance(model,data,meta,.15)
        advance(model,data,meta,.7,jaw=0)
        advance(model,data,meta,.9,lift=.050)
        relative = data.body('object').xpos[2]-data.site('gripper_center').xpos[2]
        advance(model,data,meta,30)
        after = data.body('object').xpos[2]-data.site('gripper_center').xpos[2]
        self.assertLess(abs(after-relative),.0002)
        self.assertGreater(data.body('object').xpos[2],.035)

    def test_sample_releases_inside_60mm_hole_with_jaws_above_plate(self):
        model,data,meta = coupon('sample',lab_ring=True)
        obj = model.body('object').id
        advance(model,data,meta,.15)
        advance(model,data,meta,.7,jaw=0)
        advance(model,data,meta,.9,lift=.050)
        self.assertGreater(data.xpos[obj,2],.035)
        advance(model,data,meta,3.0,target_y=.200)
        advance(model,data,meta,.9,lift=0,target_y=.200)
        advance(model,data,meta,.7,jaw=.025,target_y=.200)
        self.assertLess(np.linalg.norm(data.xpos[obj,:2]-[0,.305]),.0018)
        self.assertLess(data.xpos[obj,2],.003)
        pad_ids = {model.geom(f'robot_comp_pad_{side}_sample').id for side in ('left','right')}
        ring_ids = {model.geom(f'ring_{i}').id for i in range(64)}
        self.assertFalse(any((int(c.geom[0]) in pad_ids and int(c.geom[1]) in ring_ids)
                             or (int(c.geom[1]) in pad_ids and int(c.geom[0]) in ring_ids)
                             for c in data.contact))


if __name__ == '__main__':
    unittest.main()
