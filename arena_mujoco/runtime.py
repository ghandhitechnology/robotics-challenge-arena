"""Native MuJoCo stepping, material contacts, motor limits and complete state."""
from __future__ import annotations

import copy
import numpy as np
import mujoco

from .builder import build_arena
from .materials import load_profile
from .materials import material_pair
from .tape import TapeController


class ArenaSimulation:
    def __init__(self, config="senior_preliminary", *, tape_mode="flex", robot=True,
                 randomize=False, seed=None, profile=None):
        self.config, self.tape_mode, self.robot_enabled = config, tape_mode, robot
        self.randomize, self.seed, self.base_profile = randomize, seed, copy.deepcopy(profile)
        self._compile(seed)

    def _compile(self, seed):
        self.profile = load_profile(self.base_profile, seed=seed, randomize=self.randomize)
        self.xml, self.metadata = build_arena(self.config, tape_mode=self.tape_mode,
                                             robot=self.robot_enabled, profile=self.profile)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.tape = TapeController(self.model, self.metadata, self.profile)
        self._dry_pairs = {}
        for pair in self.metadata["contact_pairs"]:
            pid = self.model.pair(pair["name"]).id
            self._dry_pairs[tuple(sorted((int(self.model.pair_geom1[pid]),int(self.model.pair_geom2[pid]))))] = (pid,pair["parameters"])
        self._default_friction = self.model.pair_friction.copy()
        self._default_geom_friction = self.model.geom_friction.copy()
        self._vinyl_materials = {self.model.geom(item["name"]).id: material_pair(self.profile,item["material"],"vinyl")
                                 for item in self.metadata["rigid_geoms"] if item["dynamic"]}
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._reset_data()

    def _reset_data(self):
        mujoco.mj_resetData(self.model,self.data)
        self.model.pair_friction[:] = self._default_friction
        self.model.geom_friction[:] = self._default_geom_friction
        for joint,value in self.metadata["flatten_qpos"].items():
            self.data.qpos[self.model.joint(joint).qposadr[0]]=value
        mujoco.mj_forward(self.model,self.data)
        self.tape.initialize(self.data)
        self.tape.before_step(self.data)
        mujoco.mj_forward(self.model,self.data)

    def reset(self, seed=None):
        if seed is not None:
            self.seed=seed
        if self.randomize:
            self._compile(self.seed)
        else:
            self._reset_data()
        return self.data

    def _contact_friction(self):
        # Material updates precede mj_step, which rebuilds the constraint cache.
        # Slip is from the previous accepted step, a documented one-step lag.
        speeds={}
        vinyl_speeds={}
        body_velocity={}
        data,model=self.data,self.model
        for contact in data.contact:
            g1,g2=int(contact.geom[0]),int(contact.geom[1])
            if g1<0 or g2<0:
                gid=max(g1,g2)
                if gid not in self._vinyl_materials:
                    continue
                bid=int(model.geom_bodyid[gid])
                velocity=np.zeros(6)
                mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,velocity,0)
                relative=velocity[3:]+np.cross(velocity[:3],contact.pos-data.xpos[bid])
                side=0 if g1<0 else 1
                fid,eid=int(contact.flex[side]),int(contact.elem[side])
                vid=int(contact.vert[side])
                if fid>=0:
                    if vid>=0:
                        vertices=[int(model.flex_vertadr[fid])+vid]
                    elif eid>=0:
                        address=int(model.flex_elemdataadr[fid])+3*eid
                        vertices=model.flex_elem[address:address+3]+model.flex_vertadr[fid]
                    else:
                        vertices=[]
                    if len(vertices):
                        # The full-DOF tape nodes have three world-axis slide joints.
                        bodies=model.flex_vertbodyid[vertices]
                        addresses=model.body_dofadr[bodies]
                        relative-=np.mean([data.qvel[a:a+3] for a in addresses],axis=0)
                tangent=relative-contact.frame[:3]*np.dot(relative,contact.frame[:3])
                vinyl_speeds[gid]=max(vinyl_speeds.get(gid,0),float(np.linalg.norm(tangent)))
                continue
            key=tuple(sorted((g1,g2)))
            if key not in self._dry_pairs:
                continue
            relative=np.zeros(3)
            for sign,gid in ((-1,g1),(1,g2)):
                bid=int(model.geom_bodyid[gid])
                if bid not in body_velocity:
                    velocity=np.zeros(6)
                    mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,velocity,0)
                    body_velocity[bid]=velocity
                v=body_velocity[bid]
                relative+=sign*(v[3:]+np.cross(v[:3],contact.pos-data.xpos[bid]))
            tangent=relative-contact.frame[:3]*np.dot(relative,contact.frame[:3])
            speeds[key]=max(speeds.get(key,0),float(np.linalg.norm(tangent)))
        self.model.pair_friction[:]=self._default_friction
        self.model.geom_friction[:]=self._default_geom_friction
        threshold=self.profile["contact"]["slip_speed_m_s"]
        for key,speed in speeds.items():
            pid,values=self._dry_pairs[key]
            mu=values[1]+(values[0]-values[1])*np.exp(-(speed/threshold)**2)
            model.pair_friction[pid,:2]=mu
        for gid,speed in vinyl_speeds.items():
            values=self._vinyl_materials[gid]
            model.geom_friction[gid,0]=values[1]+(values[0]-values[1])*np.exp(-(speed/threshold)**2)

    def _motors(self, action):
        if "robot" not in self.metadata:
            if action is not None and len(action):
                raise ValueError("This arena has no robot actuators")
            return
        robot=self.metadata["robot"]
        action=np.zeros(2) if action is None else np.asarray(action,dtype=float)
        if action.shape!=(2,) or not np.isfinite(action).all():
            raise ValueError("action must contain two finite normalized wheel commands")
        action=np.clip(action,-1,1)
        names=robot.get("wheel_joints",["wheel_left","wheel_right"])
        if isinstance(names,dict):
            names=[names[side] for side in ("left","right")]
        actuators=robot.get("motor_names",["motor_left","motor_right"])
        if isinstance(actuators,dict):
            actuators=[actuators[side] for side in ("left","right")]
        speed=float(robot.get("max_wheel_speed_rad_s",20.0))
        limit=float(robot.get("max_motor_torque_nm",.12))
        gain=float(robot.get("velocity_gain",.04))
        direction=float(robot.get("wheel_direction_sign",-1))
        scale=float(self.profile["robot"].get("motor_strength_scale",1))
        for value,joint,actuator in zip(action,names,actuators):
            velocity=self.data.qvel[self.model.joint(joint).dofadr[0]]
            # Linear torque-speed envelope approximates a DC motor at finite voltage.
            torque_limit=limit*max(0,1-abs(velocity)/(speed*1.25))
            torque=np.clip(gain*(direction*value*speed-velocity),-torque_limit,torque_limit)
            self.data.ctrl[self.model.actuator(actuator).id]=torque

    def step(self, action=None, nstep=1):
        if isinstance(nstep,bool) or int(nstep)!=nstep or nstep<1:
            raise ValueError("nstep must be a positive integer")
        for _ in range(int(nstep)):
            # Refresh derived contacts consistently, including after checkpoint restore.
            mujoco.mj_forward(self.model,self.data)
            self._contact_friction()
            self.tape.before_step(self.data)
            self._motors(action)
            mujoco.mj_step(self.model,self.data)
            self.tape.after_step(self.data)
            if not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning):
                raise FloatingPointError("MuJoCo numerical warning; reduce timestep or revise the sampled material profile")
        return self.data

    def get_state(self):
        state=np.empty(mujoco.mj_stateSize(self.model,self._state_spec))
        mujoco.mj_getState(self.model,self.data,state,self._state_spec)
        return {"physics":state,"tape":self.tape.state_dict(),
                "pair_friction":self.model.pair_friction.copy(),
                "geom_friction":self.model.geom_friction.copy(),
                "profile":copy.deepcopy(self.profile),"xml":self.xml}

    def set_state(self,state):
        if state["xml"]!=self.xml:
            raise ValueError("State belongs to a different model/profile; rebuild with its profile first")
        mujoco.mj_setState(self.model,self.data,np.asarray(state["physics"]),self._state_spec)
        self.model.pair_friction[:]=state["pair_friction"]
        self.model.geom_friction[:]=state["geom_friction"]
        self.tape.load_state_dict(state["tape"])
        mujoco.mj_forward(self.model,self.data)
        self.tape.before_step(self.data)
        # Forward overwrites warmstart-related arrays; restore the integrator state after cache refresh.
        mujoco.mj_setState(self.model,self.data,np.asarray(state["physics"]),self._state_spec)

    def diagnostics(self):
        return {"time_s":float(self.data.time),"contacts":int(self.data.ncon),
                "max_abs_qvel":float(np.max(np.abs(self.data.qvel),initial=0)),
                "warnings":sum(int(w.number) for w in self.data.warning),**self.tape.metrics()}
