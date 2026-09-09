"""Native competition robot control with torque limits and contact-only grasps."""
import math
import copy
import numpy as np
import mujoco
from .runtime import ArenaSimulation
from .competition_policy import observation, VELOCITY_LIMITS
from .competition_events import update_competition_events


def wrap(angle):
    return (angle+math.pi)%(2*math.pi)-math.pi


class CompetitionSimulation(ArenaSimulation):
    def __init__(self, *, tape_mode='rigid', seed=0, randomize=False, timestep=None):
        timestep=(.0002 if tape_mode=='flex' else .001) if timestep is None else timestep
        super().__init__(robot='competition', tape_mode=tape_mode, seed=seed,
                         randomize=randomize, profile={'timestep_s': timestep})
        self.meta=self.metadata['robot']
        self.metadata['competition_events']={'robot_outside_count':0,'human_intervention':False}
        self.lift_target=0.
        self.jaw_target=.025
        self.motor_integrals=np.zeros(2)
        self._outside=False
        self.control_dt=.02
        self.distance_m=0.
        self.max_tilt_rad=0.
        self.max_wheel_command=0.
        self.contact_impulses=0.
        self.last_axle=self.pose()[0].copy()
        self.observations=[]
        self.actions=[]
        self.action_times=[]

    def get_state(self):
        state=super().get_state()
        names=('lift_target','jaw_target','motor_integrals','distance_m','max_tilt_rad',
               'max_wheel_command','last_axle','observations','actions','action_times')
        state['competition_controller']={name:copy.deepcopy(getattr(self,name)) for name in names}
        state['competition_events']=copy.deepcopy(self.metadata.get('competition_events'))
        return state

    def set_state(self,state):
        super().set_state(state)
        self.meta=self.metadata['robot']
        for name,value in state['competition_controller'].items():
            setattr(self,name,copy.deepcopy(value))
        self.metadata['competition_events']=copy.deepcopy(state['competition_events'])

    def reset(self,seed=None):
        super().reset(seed)
        self.meta=self.metadata['robot']
        self.lift_target=0.;self.jaw_target=.025
        self.motor_integrals=np.zeros(2)
        self.distance_m=0.;self.max_tilt_rad=0.;self.max_wheel_command=0.
        self.last_axle=self.pose()[0].copy()
        self.observations=[];self.actions=[];self.action_times=[]
        self.metadata['competition_events']={'robot_outside_count':0,'human_intervention':False}
        return self.data

    def pose(self):
        body=self.data.body('competition_robot')
        rotation=body.xmat.reshape(3,3)
        axle=body.xpos+rotation@np.array([0,self.metadata['robot']['wheel_axle_y_m'],0])
        heading=math.atan2(rotation[1,1],rotation[0,1])
        return axle[:2].copy(), heading

    def joint_position(self, name):
        return float(self.data.qpos[self.model.joint(name).qposadr[0]])

    def tool_position(self):
        return self.data.site('gripper_center').xpos.copy()

    def _motors(self, action):
        if action is None:
            action=np.zeros(2)
        robot=self.metadata['robot']
        for i,(value,joint,actuator) in enumerate(zip(np.asarray(action)[:2],robot['wheel_joints'],robot['motor_names'])):
            velocity=self.data.qvel[self.model.joint(joint).dofadr[0]]
            desired=robot['wheel_direction_sign']*float(value)*robot['max_wheel_speed_rad_s']
            error=desired-velocity
            # Encoder PI feedback overcomes bearing/caster stiction at the very
            # low wheel speeds needed for a 2 mm-clearance sample placement.
            self.motor_integrals[i]=np.clip(self.motor_integrals[i]+.020*error*self.model.opt.timestep,-.008,.008)
            request=robot['velocity_gain']*error+self.motor_integrals[i]
            limit=robot['max_motor_torque_nm']
            if request*velocity>0:
                limit*=max(0,1-abs(velocity)/robot['motor_no_load_speed_rad_s'])
            self.data.ctrl[self.model.actuator(actuator).id]=np.clip(request,-limit,limit)
        if hasattr(self,'lift_target'):
            self.data.ctrl[self.model.actuator('comp_lift_motor').id]=self.lift_target
            self.data.ctrl[self.model.actuator('comp_grip_motor').id]=self.jaw_target

    def control(self, policy, forward_error, heading_error, lift_goal, jaw_goal):
        action_time=float(self.data.time)
        errors=[forward_error,heading_error,
                lift_goal-self.joint_position('comp_lift'),
                jaw_goal-self.joint_position('comp_jaw_left')]
        obs=observation(errors)
        action=np.asarray(policy(obs))
        if action.shape!=(4,) or not np.isfinite(action).all():
            raise ValueError('Policy must return four finite normalized controls')
        velocities=np.clip(action,-1,1)*VELOCITY_LIMITS
        lift_position=self.joint_position('comp_lift')
        jaw_position=self.joint_position('comp_jaw_left')
        # Anti-windup keeps the integral targets within their force-limited
        # servo error. Opening can then unload a grasp immediately.
        self.lift_target=float(np.clip(np.clip(self.lift_target+velocities[2]*self.control_dt,lift_position-.005,lift_position+.005),0,.080))
        self.jaw_target=float(np.clip(np.clip(self.jaw_target+velocities[3]*self.control_dt,jaw_position-.001,jaw_position+.001),0,.025))
        v,w=velocities[:2]
        wheel=np.array([v-w*.170/2,v+w*.170/2])/(.020*20)
        wheel=np.clip(wheel,-1,1)
        self.max_wheel_command=max(self.max_wheel_command,float(abs(wheel).max()))
        if self.tape_mode=='flex':
            self.step(wheel, nstep=round(self.control_dt/self.model.opt.timestep))
        else:
            # The native contact solver still runs at 1 kHz. Refresh the
            # slip-dependent material coefficients at 100 Hz in the fast tier.
            for i in range(round(self.control_dt/self.model.opt.timestep)):
                if i%max(1,round(.01/self.model.opt.timestep))==0:
                    self._contact_friction()
                self._motors(wheel)
                mujoco.mj_step(self.model,self.data)
            if not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning):
                raise FloatingPointError('Numerical warning in competition simulation')
        mujoco.mj_forward(self.model,self.data)
        position,_=self.pose()
        self.distance_m+=float(np.linalg.norm(position-self.last_axle))
        self.last_axle=position
        rotation=self.data.body('competition_robot').xmat.reshape(3,3)
        self.max_tilt_rad=max(self.max_tilt_rad,math.acos(np.clip(rotation[2,2],-1,1)))
        update_competition_events(self.model,self.data,self.metadata)
        self.observations.append(obs)
        self.actions.append(action)
        self.action_times.append(action_time)
        return action
