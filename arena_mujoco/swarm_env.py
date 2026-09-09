"""Batched contact-transport curriculum with an explicit geometric task allocator.

The shared policy commands all wheel speeds and lift targets. A state machine
supplies grasp poses, transport goals, and queue poses. Object attachment,
external object forces, and in-episode state edits are deliberately absent.
"""
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from .builder import build_arena, numbers
from .swarm_robot import DESIGN, add_swarm_robots

LOCAL_DIM = 32
NEIGHBOR_DIM = 8
GLOBAL_DIM = 72
FEATURES = [
    "target_right", "target_forward", "heading_sin", "heading_cos",
    "object_right", "object_forward", "goal_right", "goal_forward",
    "velocity_right", "velocity_forward", "yaw_velocity", "lift_fraction",
    "own_pad_contact", "partner_pad_contact", "object_clearance", "carrier",
    "phase_approach", "phase_lift", "phase_transport", "phase_lower",
    "phase_retreat", "phase_done", "previous_left", "previous_right",
    "previous_lift", "axis_right", "axis_forward", "object_radius",
    "object_velocity_right", "object_velocity_forward", "delivery_fraction",
    "episode_fraction",
]


def build_swarm_scene(num_robots=40, num_objects=4, timestep=.002):
    """Reuse the audited field and payload geometries in a transport benchmark.

    The benchmark relocates a subset of the competition payloads for repeated
    trials. Its success measure is transport, not the official competition score.
    """
    if not 2 <= num_robots <= 40 or num_robots % 2:
        raise ValueError("num_robots must be an even number from 2 to 40")
    if not 1 <= num_objects <= num_robots // 2:
        raise ValueError("Every payload needs an opposing pair of modules")
    if timestep not in (.001, .002):
        raise ValueError("Swarm timestep must be a contact-validated 1 or 2 ms")
    xml, metadata = build_arena(robot=False, tape_mode="rigid", profile={"timestep_s": .001})
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    original = {o["name"]: world.find(f"body[@name='{o['name']}']") for o in metadata["objects"]}
    categories = ["Cylinder_Red", "Medical_Kit", "Biological_Sample", "Cylinder_Green", "Cylinder_Yellow"]
    names = []
    for category in categories:
        names.extend(name for name in original if name.startswith(category))
    # Interleave task types, so even a three-object evaluation covers all three
    # grasp geometries. Additional objects reuse the same physical definitions.
    first = [next(n for n in original if n.startswith(prefix)) for prefix in categories[:3]]
    names = first + [name for name in names if name not in first]
    for body in original.values():
        world.remove(body)
    for geom in list(world.findall("geom")):
        if (geom.get("name") or "").endswith("_mark"):
            world.remove(geom)
    objects = []
    cols = min(4, max(1, math.ceil(math.sqrt(num_objects))))
    rows = math.ceil(num_objects / cols)
    for index in range(num_objects):
        source = names[index % len(names)]
        body = copy.deepcopy(original[source])
        new_name = f"payload_{index:02d}"
        for element in body.iter():
            if element.get("name"):
                element.set("name", new_name + "_" + element.get("name"))
        body.set("name", new_name)
        joint = body.find("freejoint")
        joint.set("name", new_name + "_free")
        shape = next(g for g in body.findall("geom") if g.get("contype", "1") != "0")
        shape.set("name", new_name + "_geom")
        shape.set("priority", "0")
        size = np.fromstring(shape.get("size"), sep=" ")
        is_box = shape.get("type") == "box"
        radius, half_height = (size[1], size[2]) if is_box else (size[0], size[1])
        x = .28 + (index % cols) * min(.23, .62 / max(1, cols - 1))
        y = .28 + (index // cols) * min(.31, .65 / max(1, rows - 1))
        body.set("pos", numbers([x, y, half_height + .0001]))
        world.append(body)
        kind = "sample" if source.startswith("Biological") else ("kit" if is_box else "cylinder")
        objects.append({"name": new_name, "source_name": source, "kind": kind,
                        "radius": float(radius), "half_height": float(half_height),
                        "initial_position_m": [x, y, float(half_height + .0001)]})
    # Eight columns and five rows fit entirely in the 280 x 480 mm start zone.
    packing = [(.884 + (i % 8) * .031, .744 + (i // 8) * .084, .0072) for i in range(num_robots)]
    robots = add_swarm_robots(root, num_robots, packing)
    # Remove material-pair entries that referenced a removed original payload.
    geom_names = {g.get("name") for g in root.iter("geom")}
    contact = root.find("contact")
    if contact is not None:
        for pair in list(contact):
            if pair.tag == "pair" and (pair.get("geom1") not in geom_names or pair.get("geom2") not in geom_names):
                contact.remove(pair)
    option = root.find("option")
    option.set("timestep", str(timestep))
    option.set("solver", "Newton")
    option.set("iterations", "50")
    option.set("ls_iterations", "50")
    option.set("tolerance", "1e-8")
    option.set("cone", "elliptic")
    option.set("impratio", "10")
    ET.SubElement(world, "camera", name="swarm_side", pos="1.75 -.75 .65", xyaxes=".78 .62 0 -.20 .25 .95")
    metadata.update(robots=robots, objects=objects, task="cooperative_transport_benchmark",
                    rule_score=False, robot_start_packing_m=packing)
    return ET.tostring(root, encoding="unicode"), metadata


class SwarmVectorEnv:
    """Torch observations/rewards with native or batched MuJoCo-Warp dynamics."""

    def __init__(self, num_envs=1, num_robots=40, num_objects=4, device="cpu",
                 backend="native", seed=0, timestep=.002, control_dt=.02,
                 episode_seconds=40., start_mode="grasp", **kwargs):
        if kwargs:
            raise TypeError(f"Unknown environment arguments: {sorted(kwargs)}")
        if backend not in {"native", "warp"}:
            raise ValueError("backend must be native or warp")
        self.device = torch.device(device)
        self.backend, self.num_envs = backend, int(num_envs)
        self.num_robots, self.num_objects = int(num_robots), int(num_objects)
        self.dt, self.timestep = float(control_dt), float(timestep)
        self.substeps = round(self.dt / self.timestep)
        self.max_steps = round(episode_seconds / self.dt)
        self.start_mode = start_mode
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.seed = seed
        self.xml, self.metadata = build_swarm_scene(num_robots, num_objects, timestep)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self._bind()
        self.curriculum = {"num_active_robots": num_robots, "num_active_objects": num_objects, "difficulty": 1.}
        self.command = torch.zeros((num_envs, num_robots, 3), device=self.device)
        self.lift_target = torch.zeros((num_envs, num_robots), device=self.device)
        self._init_physics()
        self.reset()

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _bind(self):
        m = self.model
        self.robot_qadr_np = np.array([m.joint(r["freejoint"]).qposadr[0] for r in self.metadata["robots"]])
        self.robot_dadr_np = np.array([m.joint(r["freejoint"]).dofadr[0] for r in self.metadata["robots"]])
        self.wheel_dofs_np = np.array([[m.joint(j).dofadr[0] for j in r["wheel_joints"]] for r in self.metadata["robots"]])
        self.lift_qadr_np = np.array([m.joint(r["lift_joint"]).qposadr[0] for r in self.metadata["robots"]])
        self.actuator_ids_np = np.array([[m.actuator(n).id for n in r["motor_names"] + [r["lift_motor"]]] for r in self.metadata["robots"]])
        self.object_qadr_np = np.array([m.joint(o["name"] + "_free").qposadr[0] for o in self.metadata["objects"]])
        self.object_dadr_np = np.array([m.joint(o["name"] + "_free").dofadr[0] for o in self.metadata["objects"]])
        self.robot_qadr = self._tensor(self.robot_qadr_np, torch.long)
        self.robot_dadr = self._tensor(self.robot_dadr_np, torch.long)
        self.wheel_dofs = self._tensor(self.wheel_dofs_np, torch.long)
        self.lift_qadr = self._tensor(self.lift_qadr_np, torch.long)
        self.object_qadr = self._tensor(self.object_qadr_np, torch.long)
        self.object_dadr = self._tensor(self.object_dadr_np, torch.long)
        self.object_radius = self._tensor([o["radius"] for o in self.metadata["objects"]])
        self.object_half_height = self._tensor([o["half_height"] for o in self.metadata["objects"]])
        self.geom_robot = np.full(m.ngeom, -1, dtype=np.int32)
        self.geom_pad = np.full(m.ngeom, -1, dtype=np.int32)
        self.geom_object = np.full(m.ngeom, -1, dtype=np.int32)
        for i, robot in enumerate(self.metadata["robots"]):
            for collider in robot["colliders"]:
                self.geom_robot[m.geom(collider["name"]).id] = i
            for name in robot["pad_geoms"]:
                self.geom_pad[m.geom(name).id] = i
        for i, obj in enumerate(self.metadata["objects"]):
            self.geom_object[m.geom(obj["name"] + "_geom").id] = i
        self.geom_pad_t = self._tensor(self.geom_pad, torch.long)
        self.geom_robot_t = self._tensor(self.geom_robot, torch.long)
        self.geom_object_t = self._tensor(self.geom_object, torch.long)
        self.robot_indices = torch.arange(self.num_robots, device=self.device)
        self.object_indices = torch.arange(self.num_objects, device=self.device)
        self.partner = self.robot_indices ^ 1
        self.assignment = (self.robot_indices // 2).clamp(max=self.num_objects - 1)
        self.pair_sign = torch.where(self.robot_indices % 2 == 0, 1., -1.)

    def _init_physics(self):
        initial = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, initial)
        if self.backend == "native":
            if self.device.type != "cpu":
                raise ValueError("Native debug physics uses CPU tensors; use warp for GPU training")
            self.native_data = [mujoco.MjData(self.model) for _ in range(self.num_envs)]
            self.qpos = self._tensor(np.tile(initial.qpos, (self.num_envs, 1)))
            self.qvel = torch.zeros((self.num_envs, self.model.nv), device=self.device)
            self.overflow = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        else:
            if self.device.type != "cuda":
                raise ValueError("Warp training requires an NVIDIA CUDA device")
            import warp as wp
            import mujoco_warp as mjw
            from .swarm_gpu import motor_control
            self.wp, self.mjw, self.motor_kernel = wp, mjw, motor_control
            self.wp_device = wp.get_device(str(self.device) if self.device.index is not None else "cuda:0")
            with wp.ScopedDevice(self.wp_device):
                self.gpu_model = mjw.put_model(self.model)
                nconmax = self.num_robots * 28 + self.num_objects * 24 + 64
                self.gpu_data = mjw.put_data(self.model, initial, nworld=self.num_envs,
                                             nconmax=nconmax, njmax=nconmax * 5)
                self.qpos = wp.to_torch(self.gpu_data.qpos)
                self.qvel = wp.to_torch(self.gpu_data.qvel)
                self.overflow = wp.to_torch(self.gpu_data.overflow)
                self.wp_commands = wp.from_torch(self.command, dtype=wp.float32)
                self.wp_lift_target = wp.from_torch(self.lift_target, dtype=wp.float32)
                self.wp_wheel_dofs = wp.array(self.wheel_dofs_np.astype(np.int32), dtype=int)
                self.wp_actuator_ids = wp.array(self.actuator_ids_np.astype(np.int32), dtype=int)
                self._physics_graph = None

    def set_curriculum(self, stage):
        if isinstance(stage, int):
            robots = [2, 4, 8, self.num_robots][min(stage, 3)]
            stage = {"num_active_robots": robots, "num_active_objects": min(max(1, robots // 2), self.num_objects), "difficulty": min(stage / 3, 1)}
        self.curriculum.update(stage)
        self.curriculum["num_active_robots"] = min(self.num_robots, max(2, int(self.curriculum["num_active_robots"])))
        self.curriculum["num_active_objects"] = min(self.num_objects, self.curriculum["num_active_robots"] // 2,
                                                       max(1, int(self.curriculum["num_active_objects"])))

    def reset(self):
        E, N, O = self.num_envs, self.num_robots, self.num_objects
        self.steps = torch.zeros(E, dtype=torch.long, device=self.device)
        self.phase = torch.zeros((E, O), dtype=torch.long, device=self.device)
        self.phase_steps = torch.zeros((E, O), dtype=torch.long, device=self.device)
        self.ever_lifted = torch.zeros((E, O), dtype=torch.bool, device=self.device)
        self.delivered = torch.zeros((E, O), dtype=torch.bool, device=self.device)
        self.supported_previous = torch.zeros((E, O), dtype=torch.bool, device=self.device)
        self.contact_hold = torch.zeros((E, O), dtype=torch.long, device=self.device)
        self.pad_contact = torch.zeros((E, N), device=self.device)
        self.robot_collisions = torch.zeros((E, N), device=self.device)
        self.object_floor_contact = torch.zeros((E, O), device=self.device)
        self.previous_action = torch.zeros_like(self.command)
        self.active = (self.robot_indices < self.curriculum["num_active_robots"]).expand(E, -1).float()
        self.object_active = (self.object_indices < self.curriculum["num_active_objects"]).expand(E, -1)
        self.carrier = ((self.robot_indices // 2 < self.curriculum["num_active_objects"]) & (self.robot_indices < 2 * self.num_objects)).expand(E, -1)
        self.axis = torch.zeros((E, O, 2), device=self.device)
        self.origins = torch.zeros((E, O, 2), device=self.device)
        self.goals = torch.zeros_like(self.origins)
        self.queue_targets = torch.zeros((E, N, 2), device=self.device)
        self._reset_indices(torch.arange(E, device=self.device))
        return self._observation()

    def reset_done(self, done_mask):
        indices = torch.nonzero(done_mask, as_tuple=False).flatten()
        if len(indices):
            self._reset_indices(indices)
        return self._observation()

    def _reset_indices(self, ids):
        size = len(ids)
        difficulty = float(self.curriculum["difficulty"])
        qpos = self._tensor(np.tile(self.model.qpos0, (size, 1)))
        qvel = torch.zeros((size, self.model.nv), device=self.device)
        base = self._tensor([o["initial_position_m"][:2] for o in self.metadata["objects"]])
        origin = base.expand(size, -1, -1).clone()
        origin += (torch.rand((size, self.num_objects, 2), generator=self.rng, device=self.device) - .5) * (.006 + .014 * difficulty)
        angles = (torch.rand((size, self.num_objects), generator=self.rng, device=self.device) - .5) * .5 * difficulty
        axis = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        distance = .05 + .15 * difficulty
        self.origins[ids], self.axis[ids] = origin, axis
        self.goals[ids] = origin + axis * distance
        for o in range(self.num_objects):
            adr = self.object_qadr_np[o]
            qpos[:, adr:adr + 2] = origin[:, o]
            qpos[:, adr + 2] = self.object_half_height[o] + .0001
        packing = self._tensor(self.metadata["robot_start_packing_m"])
        queue = packing[:, :2].expand(size, -1, -1).clone()
        # Queue agents learn a short, staggered repositioning inside their start
        # area while carriers manipulate the objects elsewhere on the field.
        queue[..., 1] += .025
        self.queue_targets[ids] = queue
        for r in range(self.num_robots):
            adr = self.robot_qadr_np[r]
            if r < 2 * self.curriculum["num_active_objects"] and self.start_mode == "grasp":
                o, sign = r // 2, (1. if r % 2 == 0 else -1.)
                offset = self.object_radius[o] + .030 + .004 * difficulty
                qpos[:, adr:adr + 2] = origin[:, o] - sign * axis[:, o] * offset
                yaw = -angles[:, o] + (0. if sign > 0 else math.pi)
                yaw += (torch.rand(size, generator=self.rng, device=self.device) - .5) * .08 * difficulty
                qpos[:, adr + 3] = torch.cos(yaw / 2)
                qpos[:, adr + 4:adr + 6] = 0
                qpos[:, adr + 6] = torch.sin(yaw / 2)
            else:
                qpos[:, adr:adr + 3] = packing[r]
        self.qpos[ids], self.qvel[ids] = qpos, qvel
        self.command[ids] = 0
        self.command[ids, :, 2] = -1
        self.lift_target[ids] = 0
        self.previous_action[ids] = self.command[ids]
        for name in ("steps", "phase", "phase_steps", "ever_lifted", "delivered", "supported_previous", "contact_hold", "pad_contact", "robot_collisions", "object_floor_contact"):
            getattr(self, name)[ids] = 0
        if self.backend == "native":
            for local, e in enumerate(ids.tolist()):
                mujoco.mj_resetData(self.model, self.native_data[e])
                self.native_data[e].qpos[:] = qpos[local].numpy()
                mujoco.mj_forward(self.model, self.native_data[e])
        else:
            # Restore reset state through the documented reset API, then install
            # episode initial conditions. There are no state writes in step().
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[ids] = True
            reset_qpos = self.qpos.clone()
            with self.wp.ScopedDevice(self.wp_device):
                self.mjw.reset_data(self.gpu_model, self.gpu_data, self.wp.from_torch(mask, dtype=self.wp.bool))
                self.qpos[ids] = reset_qpos[ids]
                self.qvel[ids] = 0
                self.mjw.forward(self.gpu_model, self.gpu_data)
        self._read_contacts()

    def _state(self):
        qr = self.qpos[:, self.robot_qadr[:, None] + torch.arange(7, device=self.device)]
        qo = self.qpos[:, self.object_qadr[:, None] + torch.arange(7, device=self.device)]
        quat = qr[..., 3:7]
        yaw = torch.atan2(2 * (quat[..., 0] * quat[..., 3] + quat[..., 1] * quat[..., 2]),
                          1 - 2 * (quat[..., 2].square() + quat[..., 3].square()))
        forward = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=-1)
        right = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        velocity = self.qvel[:, self.robot_dadr[:, None] + torch.arange(2, device=self.device)]
        object_velocity = self.qvel[:, self.object_dadr[:, None] + torch.arange(2, device=self.device)]
        lift = self.qpos[:, self.lift_qadr]
        return qr, qo, yaw, forward, right, velocity, object_velocity, lift

    def _targets(self, state=None):
        qr, qo, yaw, forward, right, velocity, object_velocity, lift = self._state() if state is None else state
        phase = self.phase[:, self.assignment]
        axis = self.axis[:, self.assignment]
        obj = qo[:, self.assignment, :2]
        goal = self.goals[:, self.assignment]
        radius = self.object_radius[self.assignment]
        sign = self.pair_sign[None, :, None]
        gap = torch.where(phase == 0, .028 + .0005, .028 - .0004)
        offset = radius[None, :] + gap
        target = obj - sign * axis * offset[..., None]
        # Once released, back away from the object in the original grasp axis.
        retreat = obj - sign * axis * (radius[None, :] + .060)[..., None]
        target = torch.where((phase >= 4)[..., None], retreat, target)
        target = torch.where(self.carrier[..., None], target, self.queue_targets)
        wanted_yaw = torch.atan2(-axis[..., 0], axis[..., 1]) + (self.pair_sign < 0).float()[None, :] * math.pi
        wanted_yaw = torch.where(self.carrier, wanted_yaw, torch.zeros_like(wanted_yaw))
        return target, wanted_yaw, obj, goal, phase, axis

    def _observation(self):
        state = self._state()
        qr, qo, yaw, forward, right, velocity, object_velocity, lift = state
        target, wanted_yaw, obj, goal, phase, axis = self._targets(state)
        def local(vector):
            return torch.stack([(vector * right).sum(-1), (vector * forward).sum(-1)], dim=-1)
        heading_error = wanted_yaw - yaw
        clearance = (qo[..., 2] - self.object_half_height[None, :])[:, self.assignment]
        phi = torch.nn.functional.one_hot(phase, 6).float()
        fraction = (self.delivered & self.object_active).sum(-1) / self.object_active.sum(-1).clamp(min=1)
        local_obs = torch.cat([
            local(target - qr[..., :2]) / .15,
            torch.stack([torch.sin(heading_error), torch.cos(heading_error)], -1),
            local(obj - qr[..., :2]) / .1, local(goal - obj) / .4,
            local(velocity) / .14, self.qvel[:, self.robot_dadr + 5, None] / 4,
            lift[..., None] / .008, self.pad_contact[..., None], self.pad_contact[:, self.partner, None],
            clearance[..., None] / .008, self.carrier.float()[..., None], phi,
            self.previous_action, local(axis), self.object_radius[self.assignment][None, :, None].expand(self.num_envs, -1, -1) / .03,
            local(object_velocity[:, self.assignment]) / .14,
            fraction[:, None, None].expand(-1, self.num_robots, 1),
            (self.steps / self.max_steps)[:, None, None].expand(-1, self.num_robots, 1),
        ], dim=-1).clamp(-10, 10)
        delta = qr[:, None, :, :2] - qr[:, :, None, :2]
        distance = delta.square().sum(-1)
        distance[:, self.robot_indices, self.robot_indices] = float("inf")
        # Always include the opposing gripper in the communication neighborhood.
        distance[:, self.robot_indices, self.partner] = -1
        k = min(6, self.num_robots - 1)
        nearest = distance.topk(k, largest=False).indices
        ei = torch.arange(self.num_envs, device=self.device)[:, None, None]
        neighbor_delta = qr[ei, nearest, :2] - qr[:, :, None, :2]
        relative_velocity = velocity[ei, nearest] - velocity[:, :, None]
        nright, nforward = right[:, :, None], forward[:, :, None]
        same_task = (self.assignment[nearest] == self.assignment[None, :, None]) & self.carrier[:, :, None] & self.carrier[ei, nearest]
        neighbors = torch.cat([
            torch.stack([(neighbor_delta * nright).sum(-1), (neighbor_delta * nforward).sum(-1)], -1) / .15,
            torch.stack([(relative_velocity * nright).sum(-1), (relative_velocity * nforward).sum(-1)], -1) / .14,
            torch.stack([(forward[ei, nearest] * nright).sum(-1), (forward[ei, nearest] * nforward).sum(-1)], -1),
            same_task.float()[..., None], (nearest == self.partner[None, :, None]).float()[..., None],
        ], dim=-1).clamp(-10, 10)
        mask = ((neighbor_delta.square().sum(-1) < .35 ** 2) | same_task) & self.active[ei, nearest].bool()
        denominator = self.active.sum(-1, keepdim=True).clamp(min=1)
        mean = (local_obs * self.active[..., None]).sum(1) / denominator
        maximum = torch.where(self.active[..., None].bool(), local_obs, -10.).max(1).values
        phase_fraction = (torch.nn.functional.one_hot(self.phase, 6).float() * self.object_active[..., None]).sum(1) / self.object_active.sum(-1, keepdim=True).clamp(min=1)
        global_obs = torch.cat([mean, maximum, phase_fraction, fraction[:, None], (self.active.mean(-1))[:, None]], -1)
        return {"local": local_obs, "neighbors": neighbors, "neighbor_mask": mask,
                "global": global_obs, "active": self.active,
                "learning_weight": torch.where(self.carrier, 4., 1.) * self.active,
                "phase": torch.where(self.carrier, phase, torch.zeros_like(phase))}

    def teacher_action(self):
        """Demonstration controller; called only by dataset generation/baselines."""
        state = self._state()
        qr, qo, yaw, forward, right, velocity, object_velocity, lift = state
        target, wanted_yaw, obj, goal, phase, axis = self._targets(state)
        error = target - qr[..., :2]
        along = (error * forward).sum(-1)
        lateral = (error * right).sum(-1)
        heading_error = torch.atan2(torch.sin(wanted_yaw - yaw), torch.cos(wanted_yaw - yaw))
        turn = (8 * heading_error - 30 * lateral).clamp(-2., 2.)
        nav_speed = (3 * along).clamp(-.045, .045)
        pressure_speed = .021
        remaining = ((goal - obj) * axis).sum(-1)
        carry_speed = (.7 * remaining).clamp(0, .014)
        tangent = carry_speed * self.pair_sign[None, :]
        velocity_target = torch.where(phase == 0, nav_speed + pressure_speed,
                                     torch.where(phase == 2, pressure_speed + tangent, torch.full_like(nav_speed, pressure_speed)))
        velocity_target = torch.where(phase >= 4, nav_speed.clamp(-.021, 0.), velocity_target)
        velocity_target = torch.where(phase == 5, torch.zeros_like(nav_speed), velocity_target)
        velocity_target = torch.where(self.carrier, velocity_target, nav_speed)
        # Soft avoidance is restricted to navigation. A grasping pair maintains
        # contact instead of treating its partner as an obstacle.
        delta = qr[:, :, None, :2] - qr[:, None, :, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp(min=.001)
        other = self.robot_indices[:, None] != self.robot_indices[None, :]
        partner = self.robot_indices[None, :] == self.partner[:, None]
        avoidance = ((delta / distance[..., None]) * ((.055 - distance).clamp(min=0) * other * ~partner)[..., None]).sum(2)
        nav = (~self.carrier) | (phase == 0)
        turn += torch.where(nav, -20 * (avoidance * right).sum(-1), 0.)
        wheels = torch.stack([velocity_target - turn * .021 / 2, velocity_target + turn * .021 / 2], -1) / .14
        lift_goal = torch.where((phase == 1) | (phase == 2), 1., -1.)
        lift_goal = torch.where(self.carrier, lift_goal, -1.)
        action = torch.cat([wheels.clamp(-1, 1), lift_goal[..., None]], -1)
        idle = torch.zeros_like(action)
        idle[..., 2] = -1
        return torch.where(self.active[..., None].bool(), action, idle)

    def _step_physics(self):
        if self.backend == "native":
            actions = self.command.numpy()
            lt = self.lift_target.numpy()
            for e, data in enumerate(self.native_data):
                for _ in range(self.substeps):
                    wheel_velocity = data.qvel[self.wheel_dofs_np]
                    desired = -20 * actions[e, :, :2]
                    request = .00015 * (desired - wheel_velocity)
                    limit = np.where(request * wheel_velocity > 0, .002 * np.maximum(0, 1 - np.abs(wheel_velocity) / 26.1799388), .002)
                    data.ctrl[self.actuator_ids_np[:, :2]] = np.clip(request, -limit, limit)
                    wanted = .004 * (actions[e, :, 2] + 1)
                    lt[e] += np.clip(wanted - lt[e], -.01 * self.timestep, .01 * self.timestep)
                    data.ctrl[self.actuator_ids_np[:, 2]] = lt[e]
                    mujoco.mj_step(self.model, data)
                mujoco.mj_forward(self.model, data)
                self.qpos[e] = self._tensor(data.qpos)
                self.qvel[e] = self._tensor(data.qvel)
                if any(w.number for w in data.warning):
                    raise FloatingPointError("Native MuJoCo warning in swarm simulation")
        else:
            wp, mjw = self.wp, self.mjw
            with wp.ScopedDevice(self.wp_device):
                if self._physics_graph is None:
                    with wp.ScopedCapture() as capture:
                        for _ in range(self.substeps):
                            wp.launch(self.motor_kernel, (self.num_envs, self.num_robots), inputs=[
                                self.gpu_data.qvel, self.wp_commands, self.wp_wheel_dofs,
                                self.wp_actuator_ids, self.wp_lift_target, self.gpu_data.ctrl, self.timestep])
                            mjw.step(self.gpu_model, self.gpu_data)
                        mjw.forward(self.gpu_model, self.gpu_data)
                    self._physics_graph = capture.graph
                wp.capture_launch(self._physics_graph)

    def _read_contacts(self):
        self.pad_contact.zero_()
        self.robot_collisions.zero_()
        self.object_floor_contact.zero_()
        if self.backend == "native":
            for e, data in enumerate(self.native_data):
                for contact in data.contact:
                    if contact.dist > .0003:
                        continue
                    a, b = contact.geom
                    ra, rb = self.geom_robot[a], self.geom_robot[b]
                    pa, pb = self.geom_pad[a], self.geom_pad[b]
                    oa, ob = self.geom_object[a], self.geom_object[b]
                    if pa >= 0 and ob == int(self.assignment[pa]): self.pad_contact[e, pa] = 1
                    if pb >= 0 and oa == int(self.assignment[pb]): self.pad_contact[e, pb] = 1
                    if ra >= 0 and rb >= 0 and ra != rb:
                        self.robot_collisions[e, ra] = self.robot_collisions[e, rb] = 1
                    if oa >= 0 and rb < 0 and ob < 0: self.object_floor_contact[e, oa] = 1
                    if ob >= 0 and ra < 0 and oa < 0: self.object_floor_contact[e, ob] = 1
        else:
            d, wp = self.gpu_data, self.wp
            geom = wp.to_torch(d.contact.geom).long()
            world = wp.to_torch(d.contact.worldid).long()
            count = wp.to_torch(d.nacon)[0]
            valid = (torch.arange(len(world), device=self.device) < count) & (world >= 0) & (world < self.num_envs) & (wp.to_torch(d.contact.dist) < .0003)
            a, b = geom[:, 0].clamp(0, self.model.ngeom - 1), geom[:, 1].clamp(0, self.model.ngeom - 1)
            ra, rb = self.geom_robot_t[a], self.geom_robot_t[b]
            pa, pb = self.geom_pad_t[a], self.geom_pad_t[b]
            oa, ob = self.geom_object_t[a], self.geom_object_t[b]
            world = world.clamp(0, self.num_envs - 1)
            for pad, obj in ((pa, ob), (pb, oa)):
                safe = pad.clamp(min=0)
                match = valid & (pad >= 0) & (obj == self.assignment[safe])
                self.pad_contact.view(-1).scatter_reduce_(0, world * self.num_robots + safe, match.float(), reduce="amax", include_self=True)
            collision = valid & (ra >= 0) & (rb >= 0) & (ra != rb)
            for r in (ra, rb):
                self.robot_collisions.view(-1).scatter_reduce_(0, world * self.num_robots + r.clamp(min=0), collision.float(), reduce="amax", include_self=True)
            for obj, other_robot, other_object in ((oa, rb, ob), (ob, ra, oa)):
                match = valid & (obj >= 0) & (other_robot < 0) & (other_object < 0)
                self.object_floor_contact.view(-1).scatter_reduce_(0, world * self.num_objects + obj.clamp(min=0), match.float(), reduce="amax", include_self=True)

    def step(self, actions):
        if actions.shape != self.command.shape:
            raise ValueError(f"Expected actions {tuple(self.command.shape)}, got {tuple(actions.shape)}")
        before = self._state()
        old_distance = torch.linalg.vector_norm(self.goals - before[1][..., :2], dim=-1)
        target_before = self._targets(before)[0]
        old_reach = torch.linalg.vector_norm(target_before - before[0][..., :2], dim=-1)
        old_phase = self.phase.clone()
        self.command.copy_(actions.detach().clamp(-1, 1))
        self.command[..., :2] *= self.active[..., None]
        self.command[..., 2] = torch.where(self.active.bool(), self.command[..., 2], -1.)
        self._step_physics()
        self._read_contacts()
        self.steps += 1
        self.phase_steps += 1
        qr, qo, yaw, forward, right, velocity, object_velocity, lift = self._state()
        clearance = qo[..., 2] - self.object_half_height[None, :]
        pair_contact = self.pad_contact[:, :2 * self.num_objects].reshape(self.num_envs, self.num_objects, 2).amin(-1).bool()
        supported = pair_contact & (clearance > .004) & (self.object_floor_contact == 0)
        self.contact_hold = torch.where(pair_contact, self.contact_hold + 1, torch.zeros_like(self.contact_hold))
        distance = torch.linalg.vector_norm(self.goals - qo[..., :2], dim=-1)
        transition = ((self.phase == 0) & (self.contact_hold >= 8)) | ((self.phase == 1) & supported & (self.phase_steps > 50))
        transition |= (self.phase == 2) & (distance < .008) & supported
        pair_lift = lift[:, :2 * self.num_objects].reshape(self.num_envs, self.num_objects, 2).amax(-1)
        transition |= (self.phase == 3) & (pair_lift < .001) & (clearance < .0015) & (self.phase_steps > 55)
        transition |= (self.phase == 4) & (~pair_contact) & (self.phase_steps > 100)
        transition &= self.object_active
        self.phase = torch.where(transition, (self.phase + 1).clamp(max=5), self.phase)
        self.phase_steps = torch.where(transition, 0, self.phase_steps)
        pickup = supported & ~self.ever_lifted & self.object_active
        self.ever_lifted |= supported & self.object_active
        settled = (self.phase == 5) & (distance < .012) & (clearance.abs() < .002) & (torch.linalg.vector_norm(object_velocity, dim=-1) < .01) & ~pair_contact
        newly_delivered = settled & ~self.delivered & self.ever_lifted & self.object_active
        self.delivered |= newly_delivered
        dropped = self.supported_previous & ~supported & (old_phase == 2)
        self.supported_previous = supported.clone()
        obj_progress = 30 * (old_distance - distance) * supported * (old_phase == 2)
        obj_reward = obj_progress + 2 * pickup.float() + 10 * newly_delivered.float() - 2 * dropped.float()
        target_after = self._targets()[0]
        reach = torch.linalg.vector_norm(target_after - qr[..., :2], dim=-1)
        navigation = ~self.carrier | (old_phase[:, self.assignment] == 0)
        reach_reward = 4 * (old_reach - reach) * navigation
        team = obj_reward[:, self.assignment] * self.carrier
        symmetry = (self.pad_contact - self.pad_contact[:, self.partner]).abs() * self.carrier * ((old_phase[:, self.assignment] == 1) | (old_phase[:, self.assignment] == 2))
        outside = ((qr[..., 0] < .012) | (qr[..., 0] > 1.131) | (qr[..., 1] < .028) | (qr[..., 1] > 1.153))
        tilt = 1 - 2 * (qr[..., 4].square() + qr[..., 5].square())
        energy = .0005 * self.command[..., :2].square().sum(-1)
        smoothness = .002 * (self.command - self.previous_action).square().sum(-1)
        components = {"transport_progress": obj_progress[:, self.assignment] * self.carrier,
                      "pickup": 2 * pickup[:, self.assignment].float() * self.carrier,
                      "delivery": 10 * newly_delivered[:, self.assignment].float() * self.carrier,
                      "drop": -2 * dropped[:, self.assignment].float() * self.carrier,
                      "approach_progress": reach_reward,
                      "grip_asymmetry": -.002 * symmetry, "robot_collision": -.03 * self.robot_collisions,
                      "outside": -.2 * outside.float(), "energy": -energy, "action_change": -smoothness}
        reward = sum(components.values()) * self.active
        valid_delivered = self.delivered & settled
        success = ((valid_delivered | ~self.object_active).all(-1) & (self.steps > 100))
        failure = ((outside & self.active.bool()).any(-1) | ((tilt < .6) & self.active.bool()).any(-1) |
                   ~torch.isfinite(self.qpos).all(-1) | (self.overflow != 0))
        terminated = success | failure
        truncated = (self.steps >= self.max_steps) & ~terminated
        self.previous_action.copy_(self.command)
        info = {"success": success, "failure": failure, "delivered": valid_delivered.sum(-1),
                "lifted": self.ever_lifted.sum(-1), "supported": supported.sum(-1),
                "collisions": self.robot_collisions.sum(-1), "reward_components": components,
                "elapsed": self.steps * self.dt, "phase": self.phase.clone(),
                "overflow": self.overflow.clone(), "clearance": clearance,
                "object_goal_error": distance, "active_objects": self.object_active.sum(-1)}
        return self._observation(), reward, terminated, truncated, info

    def native_snapshot(self, world=0):
        if self.backend == "native":
            return self.native_data[world]
        result = mujoco.MjData(self.model)
        self.mjw.get_data_into(result, self.model, self.gpu_data, world_id=world)
        return result

    def close(self):
        pass
