"""Native batched hinge dynamics with measured contact and magnetic loads."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from .swarm_flow import connected_components
from .swarm_hinge_robot import HINGE_DESIGN, hinge_motor_torques
from .swarm_hinge_scene import build_hinge_swarm_scene
from .swarm_magnets import MagneticCoupling


class HingeBatchDynamics:
    """One free-root articulated model per world; actions are joint targets.

    The first three action channels request bounded yaw/front/rear angles. The
    fourth enables the module's EPM ports when positive. Only hinge actuators,
    passive collisions, gravity and balanced inter-module magnetic forces act
    during a step. Pose and velocity writes are confined to reset().
    """

    def __init__(self, num_envs=1, num_robots=40, num_objects=0, *, timestep=.001,
                 control_dt=.02, native_workers=1, solver="CG"):
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(native_workers, int) or isinstance(native_workers, bool) or native_workers < 1:
            raise ValueError("native_workers must be a positive integer")
        self.dt, self.timestep = float(control_dt), float(timestep)
        if not np.isfinite([self.dt, self.timestep]).all() or min(self.dt, self.timestep) <= 0:
            raise ValueError("control_dt and timestep must be finite and positive")
        self.substeps = round(self.dt/self.timestep)
        if self.substeps < 1 or abs(self.substeps*self.timestep-self.dt) > 1e-10:
            raise ValueError("control_dt must be a positive integer multiple of timestep")
        self.xml, self.metadata = build_hinge_swarm_scene(num_robots, num_objects, timestep, solver)
        self.num_envs, self.num_robots, self.num_objects = num_envs, num_robots, num_objects
        self.models = [mujoco.MjModel.from_xml_string(self.xml) for _ in range(num_envs)]
        self.native_data = [mujoco.MjData(model) for model in self.models]
        self.model = self.models[0]
        specs, objects, model = self.metadata["robots"], self.metadata["objects"], self.model
        self.magnets = [MagneticCoupling(model, specs) for model in self.models]
        self.root_ids = np.array([model.body(spec["name"]).id for spec in specs])
        self.robot_qadr = np.array([model.joint(spec["freejoint"]).qposadr[0] for spec in specs])
        self.robot_dadr = np.array([model.joint(spec["freejoint"]).dofadr[0] for spec in specs])
        self.joint_qadr = np.array([[model.joint(name).qposadr[0] for name in spec["joint_names"]]
                                   for spec in specs])
        self.joint_dadr = np.array([[model.joint(name).dofadr[0] for name in spec["joint_names"]]
                                   for spec in specs])
        self.motor_ids = np.array([[model.actuator(name).id for name in spec["motor_names"]]
                                  for spec in specs])
        self.object_ids = np.array([model.body(obj["name"]).id for obj in objects], dtype=int)
        self.object_geom_ids = np.array([model.geom(obj["geom"]).id for obj in objects], dtype=int)
        self.object_qadr = np.array([model.joint(obj["freejoint"]).qposadr[0] for obj in objects], dtype=int)
        self.object_dadr = np.array([model.joint(obj["freejoint"]).dofadr[0] for obj in objects], dtype=int)
        self.geom_robot = np.full(model.ngeom, -1, dtype=int)
        self.geom_part = np.full(model.ngeom, -1, dtype=int)
        self.geom_object = np.full(model.ngeom, -1, dtype=int)
        shell_geoms = []
        for robot, spec in enumerate(specs):
            for collider in spec["colliders"]:
                gid = model.geom(collider["name"]).id
                self.geom_robot[gid] = robot
                self.geom_part[gid] = (1 if collider["body"].endswith("_front") else
                                       2 if collider["body"].endswith("_rear") else 0)
                if collider["material"] != "magnet_housing":
                    shell_geoms.append(gid)
        for obj, body_id in enumerate(self.object_ids):
            self.geom_object[model.geom_bodyid == body_id] = obj
        self.shell_geoms = np.asarray(shell_geoms, dtype=int)
        self.non_magnetic_bodies = np.setdiff1d(np.arange(model.nbody), self.magnets[0].force_bodies)
        self.angle_scale = np.asarray(HINGE_DESIGN["joint_limits_rad"])[:, 1]
        self.motor_strength = np.ones(num_envs)
        self.previous_actions = np.zeros((num_envs, num_robots, 4))
        self.magnet_enabled = np.ones((num_envs, num_robots), dtype=bool)
        self.control_metrics = [dict() for _ in range(num_envs)]
        self.native_workers = min(native_workers, num_envs)
        self._pool = ThreadPoolExecutor(self.native_workers) if self.native_workers > 1 else None
        self.reset()

    def reset(self, ids=None, *, friction=None, motor_strength=None):
        ids = np.arange(self.num_envs) if ids is None else np.asarray(ids, dtype=int)
        if ids.ndim != 1 or np.any((ids < 0) | (ids >= self.num_envs)):
            raise ValueError("reset ids must name existing worlds")
        frictions = None if friction is None else np.broadcast_to(np.asarray(friction, float), (len(ids),))
        strengths = None if motor_strength is None else np.broadcast_to(np.asarray(motor_strength, float), (len(ids),))
        if frictions is not None and (not np.isfinite(frictions).all() or np.any(frictions < 0)):
            raise ValueError("friction must be finite and nonnegative")
        if strengths is not None and (not np.isfinite(strengths).all() or np.any((strengths <= 0) | (strengths > 1))):
            raise ValueError("motor strength must be in (0, 1]")
        for local, world in enumerate(ids):
            model, data, magnets = self.models[world], self.native_data[world], self.magnets[world]
            if frictions is not None:
                model.geom_friction[self.shell_geoms, 0] = frictions[local]
            if strengths is not None:
                self.motor_strength[world] = strengths[local]
            mujoco.mj_resetData(model, data)
            self.previous_actions[world] = [0., 0., 0., 1.]
            self.magnet_enabled[world] = True
            magnets.reset()
            mujoco.mj_forward(model, data)
            magnets.apply(data, self.magnet_enabled[world])
            self.control_metrics[world] = dict(min_component=self.num_robots, connected_fraction=1.,
                                               positive_work_j=0., peak_torque_nm=0.,
                                               peak_penetration_m=0., external_force_max=0.)
        return self.read_state()

    def _advance_world(self, world, action):
        model, data, magnets = self.models[world], self.native_data[world], self.magnets[world]
        targets = action[:, :3]*self.angle_scale
        enabled = action[:, 3] > 0
        metrics = dict(min_component=self.num_robots, connected_fraction=0., positive_work_j=0.,
                       peak_torque_nm=0., peak_penetration_m=0., external_force_max=0.)
        for _ in range(self.substeps):
            mujoco.mj_step1(model, data)
            torque = hinge_motor_torques(data.qpos[self.joint_qadr], data.qvel[self.joint_dadr], targets)
            torque *= self.motor_strength[world]
            data.ctrl[self.motor_ids] = torque
            magnets.apply(data, enabled)
            metrics["positive_work_j"] += float(np.maximum(torque*data.qvel[self.joint_dadr], 0).sum())*self.timestep
            metrics["peak_torque_nm"] = max(metrics["peak_torque_nm"], float(np.abs(torque).max()))
            _, sizes = connected_components(magnets.load_bearing_graph(.002))
            largest = max(sizes)
            metrics["min_component"] = min(metrics["min_component"], largest)
            metrics["connected_fraction"] += float(largest == self.num_robots)/self.substeps
            metrics["external_force_max"] = max(metrics["external_force_max"],
                                                float(np.abs(data.xfrc_applied[self.non_magnetic_bodies]).max()),
                                                float(np.abs(data.qfrc_applied).max()))
            mujoco.mj_step2(model, data)
            if data.ncon:
                relevant = ((self.geom_robot[data.contact.geom1[:data.ncon]] >= 0)
                            | (self.geom_robot[data.contact.geom2[:data.ncon]] >= 0))
                if relevant.any():
                    metrics["peak_penetration_m"] = max(metrics["peak_penetration_m"],
                                                        float(np.maximum(-data.contact.dist[:data.ncon][relevant], 0).max()))
        if np.any(data.warning.number) or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError(f"Invalid MuJoCo state in hinge world {world}")
        mujoco.mj_forward(model, data)
        magnets.apply(data, enabled)
        self.previous_actions[world] = action
        self.magnet_enabled[world] = enabled
        self.control_metrics[world] = metrics

    def step(self, actions):
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.num_envs, self.num_robots, 4) or not np.isfinite(actions).all():
            raise ValueError("actions must be finite [world, module, 4] arrays")
        actions = np.clip(actions, -1., 1.)
        if self._pool is None:
            for world, action in enumerate(actions):
                self._advance_world(world, action)
        else:
            futures = [self._pool.submit(self._advance_world, world, action)
                       for world, action in enumerate(actions)]
            for future in futures:
                future.result()
        return self.read_state()

    def _contacts(self, model, data):
        ground = np.zeros((self.num_robots, 3))
        object_normals = np.zeros((self.num_robots, self.num_objects, 3))
        object_forces = np.zeros((self.num_robots, self.num_objects, 3))
        environment_support = np.zeros(self.num_objects)
        for index, contact in enumerate(data.contact[:data.ncon]):
            first, second = int(contact.geom1), int(contact.geom2)
            ra, rb = self.geom_robot[[first, second]]
            oa, ob = self.geom_object[[first, second]]
            if max(ra, rb, oa, ob) < 0:
                continue
            wrench = np.empty(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            if wrench[0] <= 0:
                continue
            force_on_second = contact.frame.reshape(3, 3).T @ wrench[:3]
            if ra >= 0 and ob >= 0:
                object_normals[ra, ob, self.geom_part[first]] += wrench[0]
                object_forces[ra, ob] += force_on_second
            elif rb >= 0 and oa >= 0:
                object_normals[rb, oa, self.geom_part[second]] += wrench[0]
                object_forces[rb, oa] -= force_on_second
            if ra >= 0 and rb < 0 and ob < 0:
                ground[ra, self.geom_part[first]] += max(0., -force_on_second[2])
            if rb >= 0 and ra < 0 and oa < 0:
                ground[rb, self.geom_part[second]] += max(0., force_on_second[2])
            if oa >= 0 and rb < 0 and ob < 0:
                environment_support[oa] += max(0., -force_on_second[2])
            if ob >= 0 and ra < 0 and oa < 0:
                environment_support[ob] += max(0., force_on_second[2])
        return ground, object_normals, object_forces, environment_support

    def read_state(self):
        """Read the last completed control state without changing physics."""
        rows = []
        for world, (model, data, magnets) in enumerate(zip(self.models, self.native_data, self.magnets)):
            qr = data.qpos[self.robot_qadr[:, None]+np.arange(7)]
            vr = data.qvel[self.robot_dadr[:, None]+np.arange(6)]
            rotation = data.xmat[self.root_ids].reshape(-1, 3, 3)
            yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
            graph = magnets.load_bearing_graph(.002)
            labels, sizes = connected_components(graph)
            components = np.asarray(sizes)[labels]
            ground, normals, forces, support = self._contacts(model, data)
            qo = data.qpos[self.object_qadr[:, None]+np.arange(7)]
            vo = data.qvel[self.object_dadr[:, None]+np.arange(6)]
            clearance = np.zeros(self.num_objects)
            for obj, geom in enumerate(self.object_geom_ids):
                vertical = data.geom_xmat[geom].reshape(3, 3)[2]
                size = model.geom_size[geom]
                extent = (np.abs(vertical) @ size if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
                          else size[0]*np.linalg.norm(vertical[:2])+size[1]*abs(vertical[2]))
                clearance[obj] = data.geom_xpos[geom, 2]-extent
            rows.append(dict(time=float(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                             positions=qr[:, :3], quaternions=qr[:, 3:], velocity=vr[:, :3],
                             angular_velocity=vr[:, 3:], yaw=yaw, up_local=rotation[:, 2, :],
                             joint_angles=data.qpos[self.joint_qadr].copy(),
                             joint_speeds=data.qvel[self.joint_dadr].copy(),
                             ground_force_n=ground, object_contact_normal_n=normals,
                             object_contact_force_n=forces, object_environment_support_n=support,
                             object_positions=qo[:, :3], object_quaternions=qo[:, 3:],
                             object_velocity=vo[:, :3], object_clearance=clearance,
                             magnetic_graph=graph, contact_graph=magnets.graph(),
                             interaction_graph=magnets.interaction_graph(),
                             pair_force_n=magnets.pair_force_magnitudes_n.copy(),
                             component_size=components, previous_actions=self.previous_actions[world].copy(),
                             magnet_enabled=self.magnet_enabled[world].copy(),
                             **self.control_metrics[world]))
        return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
