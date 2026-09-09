"""Geometric task planning over a learned native-physics velocity controller."""
from __future__ import annotations
import json
import math
import time
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import mujoco
from .competition_runtime import CompetitionSimulation, wrap
from .competition_policy import (NeuralPolicy, ExpertPolicy, ZeroPolicy, ERROR_SCALES,
                                VELOCITY_LIMITS, OBSERVATION_NAMES, ACTION_NAMES)
from .competition_events import initialize_competition_events
from .competition_scoring import score_competition

ROOT=Path(__file__).resolve().parents[1]
CENTER_Y=.5905
CORRIDORS={'left':.245,'right':.835}


class Mission:
    def __init__(self, policy='neural', *, seed=0, tape_mode='rigid', record=True,
                 randomize=False,start_jitter_m=0.):
        self.sim=CompetitionSimulation(seed=seed,tape_mode=tape_mode,randomize=randomize)
        self.policy=NeuralPolicy(ROOT/'output/competition/policy/weights.npz') if policy=='neural' else ExpertPolicy() if policy=='expert' else ZeroPolicy()
        self.policy_name=policy
        self.seed=seed
        self.start_jitter_m=float(start_jitter_m)
        self.lift=.045
        self.jaw=.025
        self.phase='setup'
        self.records=[]
        self.events=[]
        self.record=record
        self.controls=0
        self.side='right'
        self._configure_start()
        self.initial_qpos=self.sim.data.qpos.copy()

    def _configure_start(self):
        # Choose the robot heading inside start. The task pieces retain the
        # dimension-audited diagram positions, including the four close kits.
        s=self.sim
        adr=s.model.joint('competition_robot_free').qposadr[0]
        rng=np.random.default_rng(self.seed)
        jitter=rng.uniform(-self.start_jitter_m,self.start_jitter_m,2)
        yaw=math.pi/2+rng.uniform(-self.start_jitter_m*2,self.start_jitter_m*2)
        s.data.qpos[adr:adr+7]=[.995+jitter[0],.800+jitter[1],.020,math.cos(yaw/2),0,0,math.sin(yaw/2)]
        mujoco.mj_forward(s.model,s.data)
        s.last_axle=s.pose()[0].copy()
        setup=initialize_competition_events(s.model,s.data,s.metadata)
        if not setup['valid']:
            raise RuntimeError(f'Invalid starting setup: {setup}')

    def tick(self, forward=0., heading=0.):
        s=self.sim
        s.control(self.policy,forward,heading,self.lift,self.jaw)
        self.controls+=1
        if self.record and self.controls%5==0:
            self.records.append((float(s.data.time),s.data.qpos.copy(),self.phase))
        if s.data.time>1200:
            raise RuntimeError('Feasibility run exceeded 1200 simulated seconds')

    def hold(self, seconds):
        for _ in range(round(seconds/self.sim.control_dt)):
            self.tick()

    def turn(self, heading, tolerance=.003):
        start=self.sim.data.time
        anchor=self.sim.pose()[0]
        settled=0
        while settled<5:
            position,current=self.sim.pose()
            error=wrap(heading-current)
            u=np.array([math.cos(current),math.sin(current)])
            self.tick(float((anchor-position)@u),error)
            settled=settled+1 if abs(error)<tolerance else 0
            if self.sim.data.time-start>10:
                raise RuntimeError(f'Turn failed at {self.phase}: error={error:.5f}')

    def line(self, target, *, heading=None, tolerance=.001):
        target=np.asarray(target,dtype=float)
        position,initial_heading=self.sim.pose()
        if np.linalg.norm(target-position)<tolerance:
            if heading is not None:
                self.turn(heading)
            return
        if heading is None:
            delta=target-position
            heading=math.atan2(delta[1],delta[0])
            self.turn(heading)
        u=np.array([math.cos(heading),math.sin(heading)])
        normal=np.array([-u[1],u[0]])
        start=self.sim.data.time
        settled=0
        while settled<4:
            position,current=self.sim.pose()
            error=target-position
            forward=float(error@u)
            lateral=float(error@normal)
            sign=1 if forward>=0 else -1
            aim=heading+np.clip(math.atan2(lateral,max(abs(forward),.025))*sign,-.35,.35)
            yaw=wrap(aim-current)
            self.tick(forward*max(0,math.cos(yaw)),yaw)
            settled=settled+1 if abs(forward)<tolerance and abs(lateral)<max(tolerance,.0012) else 0
            if self.sim.data.time-start>20:
                raise RuntimeError(f'Line failed at {self.phase}: target={target.tolist()}, pos={position.tolist()}, error={error.tolist()}')
        self.turn(heading)

    def corridor(self, side, y):
        x=CORRIDORS[side]
        if self.side!=side:
            self.line([CORRIDORS[self.side],CENTER_Y])
            self.line([x,CENTER_Y])
            self.side=side
        self.line([x,y])

    def object_position(self,name):
        return self.sim.data.body(name).xpos.copy()

    def pickup(self,name,side):
        self.phase=f'pick {name}'
        s=self.sim
        pos=self.object_position(name)
        heading=0. if side=='right' and not name.startswith('Cylinder') else math.pi if side=='right' else 0.
        is_kit=name.startswith('Medical')
        if is_kit:
            heading=math.pi/2
        u=np.array([math.cos(heading),math.sin(heading)])
        width=.056 if name.startswith('Biological') else .025 if name.startswith('Medical') else .020
        self.lift=.045
        self.jaw=.009 if is_kit else min(.025,(width+.014-.018)/2)
        if is_kit:
            self.corridor(side,.800)
            self.line([pos[0],.800])
            retreat=np.array([pos[0],.800])
        else:
            self.corridor(side,float(pos[1]))
            retreat=np.array([CORRIDORS[side],pos[1]])
        self.turn(heading)
        pos=self.object_position(name)
        reach=s.meta['grasp_center_body_m'][1]-s.meta['wheel_axle_y_m']
        if name.startswith('Biological'):
            staging=pos[:2]-(reach+.075)*u
            # The first disc lies close to the transit lane. Align with the
            # fingers raised, then back away before lowering around its edge.
            if float((pos[:2]-s.pose()[0])@u)<reach+.06:
                self.line(staging,heading=heading)
        self.lift=0.
        lowering_start=s.data.time
        while abs(s.joint_position('comp_lift'))>.0002 or abs(s.joint_position('comp_jaw_left')-self.jaw)>.0003:
            self.tick()
            if s.data.time-lowering_start>4:
                raise RuntimeError(f'Manipulator did not reach pickup setup for {name}')
        self.hold(.15)
        pos=self.object_position(name)
        axle_target=pos[:2]-reach*u
        self.line(axle_target,heading=heading,tolerance=.0006)
        self.hold(.2)
        self.jaw=0.
        self.hold(.7)
        self.lift=.045
        self.hold(2.0)
        lifted=self.object_position(name)
        if lifted[2]<.025:
            raise RuntimeError(f'Grasp failed {name}: object {lifted.tolist()}, tool {s.tool_position().tolist()}')
        self.events.append({'event':'grasp','object':name,'time_s':float(s.data.time),'object_z_m':float(lifted[2])})
        self.line(retreat,heading=heading)
        if is_kit:
            self.line([CORRIDORS[side],.800])

    def deliver(self,name,destination,side):
        self.phase=f'deliver {name}'
        s=self.sim
        heading=math.pi if side=='left' else 0.
        u=np.array([math.cos(heading),math.sin(heading)])
        target=np.asarray(destination,float)
        self.corridor(side,float(target[1]))
        self.turn(heading)
        if self.object_position(name)[2]<.025:
            raise RuntimeError(f'Object slipped during transport: {name}')
        # The observed held-object offset includes small real contact slip.
        # An onboard pose estimator would provide these poses on the hardware.
        for _ in range(2):
            object_offset=self.object_position(name)[:2]-s.pose()[0]
            self.line(target-object_offset,heading=heading,tolerance=.0005)
            self.hold(.25)
            if np.linalg.norm(self.object_position(name)[:2]-target)<.0007:
                break
        self.lift=0.
        self.hold(2.0)
        if name.startswith('Biological') and np.linalg.norm(self.object_position(name)[:2]-target)>.001:
            offset=self.object_position(name)[:2]-s.pose()[0]
            self.line(target-offset,heading=heading,tolerance=.00035)
            self.hold(.3)
        self.jaw=.025
        release_start=s.data.time
        while s.joint_position('comp_jaw_left')<.024:
            self.tick()
            if s.data.time-release_start>4:
                raise RuntimeError(f'Jaws did not release {name}')
        self.hold(.2)
        self.events.append({'event':'release','object':name,'time_s':float(s.data.time),'position_m':self.object_position(name).tolist(),'target_xy':target.tolist()})
        self.line([CORRIDORS[side],target[1]],heading=heading)
        self.lift=.045
        self.hold(.3)

    def tasks(self):
        result=[]
        kits=[([.09,.47],'left'),([.09,.62],'left'),([.09,1.04],'left'),([.09,.18],'left')]
        for i,(dest,side) in enumerate(kits,1):
            result.append((f'Medical_Kit_{i:02d}','right',dest,side))
        for i,y in ((1,.5905),(2,.4905),(3,.3905)):
            result.append((f'Biological_Sample_{i:02d}','right',[1.068,y],'right'))
        result.extend([
            ('Cylinder_Red_05','left',[.12,.39],'left'),
            ('Cylinder_Red_07','left',[.12,.72],'left'),
            ('Cylinder_Red_08','right',[.12,.80],'left'),
            ('Cylinder_Yellow_01','left',[.12,.08],'left'),
            ('Cylinder_Yellow_11','left',[.12,.93],'left'),
            ('Cylinder_Yellow_12','right',[.12,1.13],'left'),
            ('Cylinder_Green_03','left',[1.07,.91],'right'),
            ('Cylinder_Green_09','left',[1.07,1.03],'right'),
            ('Cylinder_Green_10','right',[1.07,1.13],'right'),
        ])
        return result

    def run(self, *, output=None, max_tasks=None):
        started=time.perf_counter()
        failure=None
        complete=0
        declared_score=None
        final_view_scores=[]
        declaration_time=None
        declaration_state=None
        try:
            self.phase='leave start'
            self.hold(1.0)
            self.line([CORRIDORS['right'],.800],heading=math.pi)
            for name,pick_side,target,drop_side in self.tasks()[:max_tasks]:
                self.pickup(name,pick_side)
                self.deliver(name,target,drop_side)
                complete+=1
                print(json.dumps({'task':complete,'object':name,'sim_seconds':float(self.sim.data.time),'wall_seconds':time.perf_counter()-started,'position':self.object_position(name).tolist()}),flush=True)
            self.phase='finished'
            declaration_time=float(self.sim.data.time)
            declared_score=score_competition(self.sim.model,self.sim.data,self.sim.metadata)
            declaration_state={key:np.array(getattr(self.sim.data,key)).copy() for key in ('qpos','qvel','ctrl','time')}
            for _ in range(10):
                self.hold(.5)
                view_score=score_competition(self.sim.model,self.sim.data,self.sim.metadata)
                final_view_scores.append({'time_s':float(self.sim.data.time),'task_score':view_score['task_score'],'success':view_score['success']})
        except (RuntimeError,FloatingPointError) as error:
            failure=str(error)
            print('MISSION_FAILURE '+failure,flush=True)
        score=declared_score or score_competition(self.sim.model,self.sim.data,self.sim.metadata)
        full_delivery_count=complete==len(self.tasks())
        final_view_valid=len(final_view_scores)==10 and all(v['success'] for v in final_view_scores)
        success=failure is None and full_delivery_count and score['success'] and final_view_valid
        report={'policy':self.policy_name,'seed':self.seed,'randomize_physics':self.sim.randomize,'start_jitter_m':self.start_jitter_m,
                'completed_deliveries':complete,'planned_deliveries':len(self.tasks()),'failure':failure,'success':success,
                'completion_seconds':declaration_time if success else None,'final_view_valid':final_view_valid,
                'final_view_checks':final_view_scores,
                'final_view_seconds':max(0,float(self.sim.data.time)-(declaration_time or float(self.sim.data.time))),
                'simulation_seconds':float(self.sim.data.time),'wall_seconds':time.perf_counter()-started,
                'travel_distance_m':self.sim.distance_m,'maximum_tilt_degrees':math.degrees(self.sim.max_tilt_rad),
                'policy_calls':self.policy.calls,'maximum_normalized_wheel_command':self.sim.max_wheel_command,
                'tape_mode':self.sim.tape_mode,'timestep_s':float(self.sim.model.opt.timestep),'slip_friction_update_seconds':.01,
                'events':self.events,'diagnostics':self.sim.diagnostics(),'score':score,
                'physics_solver':{'solver':'Newton','cone':'elliptic','impratio':float(self.sim.model.opt.impratio)},
                'controller':{'network':'Four independent learned goal-feedback channels','planner':'Scripted geometric task and corridor planner',
                              'observations':'Exact simulator robot, joint and object poses; onboard perception is not implemented',
                              'wheel_servo':'Encoder PI with torque-speed and torque limits','control_dt_s':self.sim.control_dt,
                              'event_sample_interval_s':self.sim.control_dt},
                'scene_sha256':hashlib.sha256(self.sim.xml.encode()).hexdigest(),
                'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
        if self.policy_name=='neural':
            report['weights_sha256']=hashlib.sha256(Path(self.policy.path).read_bytes()).hexdigest()
        sources=list((ROOT/'arena_mujoco').glob('*.py'))+[ROOT/name for name in ('arena_spec.json','competition_rules.json','robot_design.json')]
        report['source_files_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)}
        if output:
            out=Path(output);out.mkdir(parents=True,exist_ok=True)
            (out/'scene.xml').write_text(self.sim.xml)
            (out/'metadata.json').write_text(json.dumps(self.sim.metadata,indent=2)+'\n')
            np.savez_compressed(out/'trajectory.npz',times=np.array([r[0] for r in self.records]),qpos=np.array([r[1] for r in self.records]),phases=np.array([r[2] for r in self.records]),initial_qpos=self.initial_qpos)
            np.savez_compressed(out/'policy_trace.npz',time_s=np.array(self.sim.action_times),
                                observation=np.array(self.sim.observations),action=np.array(self.sim.actions),
                                velocity_limits=VELOCITY_LIMITS,error_scales=ERROR_SCALES,
                                observation_names=np.array(OBSERVATION_NAMES),action_names=np.array(ACTION_NAMES))
            report['trajectory_sha256']=hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest()
            np.savez_compressed(out/'final_state.npz',qpos=self.sim.data.qpos,qvel=self.sim.data.qvel,ctrl=self.sim.data.ctrl,time=self.sim.data.time)
            if declaration_state is not None:
                np.savez_compressed(out/'declaration_state.npz',**declaration_state)
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
