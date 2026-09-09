"""Native contact/shell behavior checks; these do not certify material calibration."""
import json
import unittest
import xml.etree.ElementTree as ET
import mujoco,numpy as np
from arena_mujoco.tape import TapeController
from arena_mujoco.geometry import rectangle_grid

def _coupon(spacing=.01,curvature=0,first_row_only=False,grip=False):
 h=.00015;r=h/2;g=rectangle_grid([0,0,.02,.04],spacing,r)
 flat=g['vertices'].copy();flat[:,2]=r
 rest=flat.copy();s=flat[:,1]-flat[:,1].min()
 if curvature:
  rest[:,1]=flat[:,1].min()+np.sin(curvature*s)/curvature
  rest[:,2]=r+(1-np.cos(curvature*s))/curvature
 a=g['areas']*(g['nominal_area_m2']/g['mesh_area_m2'])
 root=ET.Element('mujoco');ET.SubElement(root,'option',timestep='.0001',integrator='implicitfast',solver='CG',cone='pyramidal',iterations='100',tolerance='1e-8',gravity='0 0 0' if curvature else '0 0 -9.81')
 w=ET.SubElement(root,'worldbody');ET.SubElement(w,'geom',name='floor',type='box',size='.1 .1 .005',pos='0 0 -.005',contype='1',conaffinity='1')
 ct=ET.SubElement(root,'contact');deform=ET.SubElement(root,'deformable');names=[];meta={'tape_nodes':[]}
 nums=lambda x:' '.join(map(str,np.asarray(x).ravel()))
 for i,(pos,area) in enumerate(zip(rest,a)):
  name=f'node_{i}';names.append(name);b=ET.SubElement(w,'body',name=name,pos=nums(pos))
  for j in range(3):ET.SubElement(b,'joint',type='slide',axis=nums(np.eye(3)[j]))
  mass=1350*h*area;ET.SubElement(b,'inertial',pos='0 0 0',mass=str(mass),diaginertia=nums([mass*1e-8]*3))
  ET.SubElement(b,'geom',name=f'proxy_{i}',type='sphere',size=str(r),mass='0',contype='0',conaffinity='0')
  ET.SubElement(ct,'pair',name=f'bond_{i}',geom1='floor',geom2=f'proxy_{i}',condim='3',adhesion=str(30000*area),gap='.008',solref='.0004 1',solimp='.99 .999 .00001')
  meta['tape_nodes'].append({'body_name':name,'pair_name':f'bond_{i}','area_m2':area,'rest_relative_m':[0,0,r],'support_anchor_local_m':[pos[0],flat[i,1],0], 'initial_damage':1 if first_row_only and flat[i,1]>flat[:,1].min()+1e-6 else 0})
 f=ET.SubElement(deform,'flex',name='tape',dim='2',body=' '.join(names),vertex=nums(np.zeros_like(rest)),element=nums(g['triangles']),radius=str(r))
 ET.SubElement(f,'elasticity',young='10000000',poisson='.45',thickness=str(h),elastic2d='both',damping='.00001')
 ET.SubElement(f,'contact',contype='2',conaffinity='2',condim='3',passive='false',selfcollide='auto')
 if grip:
  eq=ET.SubElement(root,'equality')
  for i in np.flatnonzero(flat[:,1]>flat[:,1].max()-1e-6):
   ET.SubElement(w,'body',name=f'grip_{i}',mocap='true',pos=nums(flat[i]))
   ET.SubElement(eq,'connect',name=f'grip_link_{i}',body1=names[i],body2=f'grip_{i}',anchor='0 0 0',solref='.0004 1',solimp='.99 .999 .00001')
 m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
 d.qpos[:]=(flat-rest).ravel()
 c=TapeController(m,meta,{});c.initialize(d)
 return m,d,c,g

def _bead(parameters=None):
    xml='''<mujoco><option timestep=".0001" integrator="implicitfast" cone="pyramidal" solver="CG"/>
    <worldbody><geom name="floor" type="box" size=".1 .1 .005" pos="0 0 -.005"/>
    <body name="node" pos="0 0 .000075"><freejoint/><geom name="proxy" type="sphere"
    size=".000075" mass=".00003" contype="0" conaffinity="0"/></body></worldbody>
    <contact><pair name="bond" geom1="floor" geom2="proxy" condim="3"
    solref=".0004 1" gap=".0001"/></contact></mujoco>'''
    metadata={'tape_nodes':[{'body_name':'node','geom_name':'proxy','pair_name':'bond',
        'area_m2':1e-6,'rest_relative_m':[0,0,.000075],
        'support_body_name':'world','support_anchor_local_m':[0,0,0]}]}
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    controller=TapeController(model,metadata,parameters)
    controller.initialize(data)
    return model,data,controller


def _step(model,data,controller,count):
    for _ in range(count):
        controller.before_step(data)
        mujoco.mj_step(model,data)
        controller.after_step(data)
    if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all():
        raise AssertionError('Native simulation produced a numerical warning')


class TapePhysicsTests(unittest.TestCase):
    def test_native_capacity_holds_then_releases_on_finite_floor(self):
        model,data,tape=_bead()
        # XML began with zero adhesion; the runtime flag must also be enabled.
        self.assertTrue(model.flg_adhesion)
        data.xfrc_applied[1,2]=.01
        _step(model,data,tape,1000)
        self.assertLess(data.qpos[2],.0002)
        self.assertEqual(tape.damage[0],0)
        data.xfrc_applied[1,2]=.06
        _step(model,data,tape,100)
        self.assertEqual(tape.damage[0],1)
        self.assertEqual(model.pair_adhesion[0],0)
        self.assertFalse(np.any(model.geom_adhesion))
        self.assertFalse(np.any(data.qfrc_applied))

    def test_shear_damage_is_irreversible_and_state_is_json_roundtrippable(self):
        model,data,tape=_bead()
        # Prescribed displacement isolates the history law from shell mechanics.
        data.qpos[0]=.004;data.time=.0001;tape.after_step(data)
        self.assertGreater(tape.damage[0],.45)
        self.assertLess(tape.damage[0],.55)
        damage=tape.damage.copy()
        data.qpos[0]=0;data.time=.0002;tape.after_step(data)
        np.testing.assert_array_equal(tape.damage,damage)
        state=json.loads(json.dumps(tape.state_dict()))
        capacity=model.pair_adhesion.copy()
        tape.initialize(data)
        self.assertEqual(tape.damage[0],0)
        tape.load_state_dict(state)
        np.testing.assert_array_equal(tape.damage,damage)
        np.testing.assert_array_equal(model.pair_adhesion,capacity)
        self.assertEqual(tape.state_dict(),state)

    def test_flipped_underside_releases_and_rebond_requires_pressure(self):
        model,data,tape=_bead()
        data.qpos[3:7]=[0,1,0,0]
        data.time=.0001;tape.after_step(data)
        self.assertEqual(tape.damage[0],1)
        self.assertEqual(model.pair_adhesion[0],0)
        model,data,tape=_bead({'allow_rebond':True,'rebond_dwell_s':.002,
                              'rebond_pressure_pa':1,'rebond_max_slip_speed_m_s':1})
        tape.damage[:]=1;tape.max_separation_ratio[:]=1
        data.xfrc_applied[1,2]=-.001
        _step(model,data,tape,100)
        self.assertEqual(tape.rebond_count,1)
        self.assertAlmostEqual(tape.damage[0],.25)

    def test_shell_at_rest_and_area_scaling(self):
        capacities=[]
        for spacing in (.01,.005):
            model,data,tape,_=_coupon(spacing)
            capacities.append(model.pair_adhesion[tape.pair_ids].sum())
            _step(model,data,tape,300)
            self.assertEqual(float(tape.damage.max()),0)
            self.assertLess(tape.metrics()['tape_max_slip_m'],.00002)
            self.assertLess(float(np.max(abs(data.qvel))),.001)
        np.testing.assert_allclose(capacities,[24,24],rtol=1e-12)

    def test_native_reference_curvature_lifts_released_edge(self):
        model,data,tape,_=_coupon(.01,20,True)
        before=float(data.flexvert_xpos[:,2].max())
        _step(model,data,tape,1000)
        self.assertGreater(float(data.flexvert_xpos[:,2].max())-before,.00005)
        # This short physical check detects curl; it is not a mesh-convergence assertion.

    def test_material_envelope_uses_reference_edge_extension(self):
        model,data,tape,_=_coupon()
        self.assertAlmostEqual(tape.metrics()['tape_max_tensile_strain'],0)
        initial=data.flexvert_xpos.copy()
        # Uniform 12% x extension must exceed the 10% small-strain envelope.
        data.qpos.reshape(-1,3)[:,0]=.12*initial[:,0]
        tape.before_step(data)
        self.assertAlmostEqual(tape.metrics()['tape_max_tensile_strain'],.12)
        self.assertTrue(tape.metrics()['tape_material_limit_exceeded'])
        state=json.loads(json.dumps(tape.state_dict()))
        tape.initialize(data)
        tape.load_state_dict(state)
        self.assertEqual(tape.state_dict(),state)
        data.qpos[:]=0
        tape.before_step(data)
        self.assertFalse(tape.metrics()['tape_material_limit_exceeded'])

    def test_controlled_native_grip_peels_shell(self):
        model,data,tape,_=_coupon(.01,grip=True)
        maximum_force=0.0
        saw_partial_damage=False
        for _ in range(1100):
            data.mocap_pos[:,2]=.000075+.2*data.time
            _step(model,data,tape,1)
            force=abs(float(np.sum(data.efc_force[:3*model.neq].reshape(-1,3)[:,2])))
            maximum_force=max(maximum_force,force)
            saw_partial_damage |= bool(np.any((tape.damage>0)&(tape.damage<1)))
        self.assertTrue(saw_partial_damage)
        self.assertGreater(maximum_force,.1)
        self.assertTrue(np.all(tape.damage==1))
        self.assertGreater(tape.contact_work_proxy_j,0)
        self.assertFalse(np.any(data.qfrc_applied))

    def test_peel_force_and_work_trend_under_mesh_refinement(self):
        results=[]
        for spacing in (.005,.0025):
            model,data,tape,_=_coupon(spacing,grip=True)
            model.opt.timestep=.00005
            middle_forces=[]
            work=0.0
            for _ in range(2200):
                data.mocap_pos[:,2]=.000075+.2*data.time
                _step(model,data,tape,1)
                force=abs(float(np.sum(data.efc_force[:3*model.neq].reshape(-1,3)[:,2])))
                work+=force*.2*model.opt.timestep
                if .2<float(tape.damage.mean())<.8:
                    middle_forces.append(force)
            self.assertTrue(np.all(tape.damage==1))
            results.append((np.mean(middle_forces),work))
        coarse,fine=np.asarray(results)
        self.assertLess(abs(coarse[0]/fine[0]-1),.15)
        self.assertLess(abs(coarse[1]/fine[1]-1),.1)


if __name__=='__main__':
    unittest.main()
