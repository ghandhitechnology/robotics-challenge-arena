"""A mobile, deformable magnetic body made from independently driven modules.

Every module starts in the start zone. Short-range shoulder magnets transmit
loads between colliding bodies; the fourth policy output requests their release.
Payloads are lifted by contact pads and receive no applied magnetic wrench.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from .builder import numbers
from .swarm_env import FEATURES as TRANSPORT_FEATURES, SwarmVectorEnv, build_swarm_scene
from .swarm_flow import DockingState, FlowConfig, compact_packing, flow_actions, body_telemetry, velocity_actions
from .swarm_magnets import MagneticCoupling, add_magnetic_docks


FEATURES = TRANSPORT_FEATURES + [
    "body_centroid_right", "body_centroid_forward", "body_goal_right", "body_goal_forward",
    "magnets_enabled", "magnetic_degree", "magnetic_component_fraction", "body_phase",
    "body_approach_stage", "docking_stage",
]
APPROACH_FLOW = FlowConfig(steering_full_speed=.001, max_yaw_rate=2.5, heading_gain=6.)
DEPLOY_FLOW = FlowConfig(consensus_steps=0)
PAYLOAD_BODY_STANDOFF = .24


class SwarmBodyEnv(SwarmVectorEnv):
    action_dim = 4

    def __init__(self, *args, backend="native", episode_seconds=70., **kwargs):
        if backend != "native":
            raise ValueError("Magnetic body physics currently requires native MuJoCo")
        self.body_ready = False
        kwargs["start_mode"] = "packing"
        super().__init__(*args, backend=backend, episode_seconds=episode_seconds, **kwargs)
        self.magnets = [MagneticCoupling(self.model, self.metadata["robots"])
                        for _ in range(self.num_envs)]
        E, N = self.num_envs, self.num_robots
        self.body_docking = [DockingState(N) for _ in range(E)]
        self.object_is_cylinder = np.array([obj["kind"] == "cylinder" for obj in self.metadata["objects"]])
        self.magnet_command = torch.ones((E, N), device=self.device)
        self.body_motor_gain = np.full((E, N), .00045)
        self.body_links = np.zeros((E, N, N), dtype=bool)
        self.body_initial_links = np.zeros((E, N, N), dtype=bool)
        self.body_new_links = np.zeros((E, N, N), dtype=bool)
        self.body_release_started = torch.zeros(E, dtype=torch.bool)
        self.body_initial_connected_run = torch.zeros(E, dtype=torch.long)
        self.body_initial_connected_best = torch.zeros(E, dtype=torch.long)
        self.body_middle_steps = torch.zeros(E, dtype=torch.long)
        self.body_middle_connected_steps = torch.zeros(E, dtype=torch.long)
        self.body_phase = torch.zeros(E, dtype=torch.long)
        self.body_approach_stage = torch.zeros((E, N), dtype=torch.long)
        self.body_initial = torch.zeros((E, N, 2))
        self.body_initial_centroid = torch.zeros((E, 2))
        self.body_waypoint = torch.zeros((E, 2))
        self.body_connected_steps = torch.zeros(E, dtype=torch.long)
        self.body_link_formations = torch.zeros(E, dtype=torch.long)
        self.body_link_releases = torch.zeros(E, dtype=torch.long)
        self.body_hold = torch.zeros(E, dtype=torch.long)
        self.body_ready = True
        self.reset()

    def _build_scene(self, num_robots, num_objects, timestep):
        if num_objects > 2:
            raise ValueError("The connected-body collection layout has one or two payloads")
        xml, metadata = build_swarm_scene(num_robots, num_objects, timestep)
        root = ET.fromstring(xml)
        poses = compact_packing(num_robots)
        # Put the grasp specialists on the leading boundary of the same body.
        if num_robots == 40:
            leading = [4, 0, 32, 36][:2*num_objects]
            order = leading + [i for i in range(40) if i not in leading]
            poses = poses[order]
        metadata["robot_start_packing_m"] = poses[:, :3].tolist()
        metadata["robot_start_yaw_rad"] = poses[:, 3].tolist()
        for r, pose in zip(metadata["robots"], poses):
            body = root.find(f".//body[@name='{r['name']}']")
            body.set("pos", numbers(pose[:3]))
            body.set("quat", numbers([math.cos(pose[3]/2), 0, 0, math.sin(pose[3]/2)]))
            r["initial_position_m"], r["initial_yaw_rad"] = pose[:3].tolist(), float(pose[3])
        centers = [[.60, .78], [.60, .99]] if num_objects == 2 else [[.60, .885]]
        for obj, xy in zip(metadata["objects"], centers):
            obj["initial_position_m"][:2] = xy
            root.find(f".//body[@name='{obj['name']}']").set("pos", numbers(obj["initial_position_m"]))
        add_magnetic_docks(root, metadata["robots"])
        metadata.update(task="magnetic_swarm_body_transport", rule_score=False,
                        policy_action_names=["left_wheel", "right_wheel", "lift", "magnet_enable"],
                        magnetic_force_targets="robot_root_bodies_only",
                        start_mode="all_modules_inside_start_zone")
        return ET.tostring(root, encoding="unicode"), metadata

    def _reset_indices(self, ids):
        super()._reset_indices(ids)
        if not self.body_ready:
            return
        packing = self._tensor(self.metadata["robot_start_packing_m"])
        yaws = self._tensor(self.metadata["robot_start_yaw_rad"])
        self.axis[ids] = self._tensor([-1., 0.])
        self.goals[ids] = self.origins[ids] + self.axis[ids] * (.10 + .10*float(self.curriculum["difficulty"]))
        for local, world in enumerate(ids.tolist()):
            data = self.native_data[world]
            for robot in range(self.num_robots):
                adr = self.robot_qadr_np[robot]
                data.qpos[adr:adr+3] = packing[robot].numpy()
                data.qpos[adr+3:adr+7] = [math.cos(float(yaws[robot])/2), 0, 0,
                                         math.sin(float(yaws[robot])/2)]
            for obj, adr in enumerate(self.object_qadr_np):
                data.qpos[adr+3:adr+7] = [1., 0, 0, 0]
            data.qvel[:] = 0
            data.xfrc_applied[:] = 0
            mujoco.mj_forward(self.model, data)
            self.qpos[world] = self._tensor(data.qpos)
            self.qvel[world] = self._tensor(data.qvel)
            self.magnets[world].reset()
            self.body_docking[world].reset()
        self.magnet_command[ids] = 1
        self.body_links[ids.numpy()] = False
        self.body_initial_links[ids.numpy()] = False
        self.body_new_links[ids.numpy()] = False
        self.body_release_started[ids] = False
        self.body_phase[ids] = 0
        self.body_approach_stage[ids] = torch.where(self.robot_indices % 2 == 0, -1, -2)
        self.body_initial[ids] = packing[:, :2]
        self.body_initial_centroid[ids] = packing[:, :2].mean(0)
        self.body_waypoint[ids] = self._tensor([.80, .885])
        for name in ("body_connected_steps", "body_link_formations", "body_link_releases", "body_hold",
                     "body_initial_connected_run", "body_initial_connected_best", "body_middle_steps",
                     "body_middle_connected_steps"):
            getattr(self, name)[ids] = 0

    def _flow_members(self, world):
        if self.body_phase[world] in (1, 2):
            return np.flatnonzero((~self.carrier[world] |
                                  (self.phase[world, self.assignment] == 5)).numpy())
        return np.arange(self.num_robots)

    def _flow_obstacles(self, qo, world):
        obstacles = [[1.012, .335, 1.124, .646]]
        for pos, radius in zip(qo[world, :, :2].numpy(), self.object_radius.numpy()):
            margin = float(radius) + .012
            obstacles.append([pos[0]-margin, pos[1]-margin, pos[0]+margin, pos[1]+margin])
        return obstacles

    def _update_docking(self, state):
        qr, qo, yaw, *_ = state
        for world, docking in enumerate(self.body_docking):
            members = self._flow_members(world)
            docking.update(qr[world, members, :2].numpy(), yaw[world, members].numpy(),
                           self.body_links[world][np.ix_(members, members)], module_ids=members,
                           obstacles=self._flow_obstacles(qo, world))

    def _flow(self, state):
        qr, qo, yaw, _, _, velocity, _, _ = state
        actions, fields, guidance_positions, guidance_yaws, guidance_active = [], [], [], [], []
        for world in range(self.num_envs):
            # Real payloads and the raised laboratory are navigation obstacles.
            obstacles = self._flow_obstacles(qo, world)
            members = self._flow_members(world)
            action = np.zeros((self.num_robots, 4))
            action[:, 2:] = [-1., 1.]
            field = np.zeros((self.num_robots, 2))
            positions = qr[world, :, :2].numpy().copy()
            headings = yaw[world].numpy().copy()
            active = np.zeros(self.num_robots, dtype=bool)
            if len(members):
                subset_action, subset_field, guidance = flow_actions(qr[world, members, :2].numpy(), velocity[world, members].numpy(),
                                            yaw[world, members].numpy(), self.body_waypoint[world].numpy(),
                                            links=self.body_links[world][np.ix_(members, members)], shape_radii=(.15, .145),
                                            obstacles=obstacles, docking=True, docking_state=self.body_docking[world],
                                            module_ids=members, return_guidance=True,
                                            config=DEPLOY_FLOW if self.body_phase[world] < 2 else None)
                action[members], field[members] = subset_action, subset_field
                positions[members], headings[members], active[members] = guidance['positions'], guidance['yaw'], guidance['active']
            actions.append(action)
            fields.append(field)
            guidance_positions.append(positions)
            guidance_yaws.append(headings)
            guidance_active.append(active)
        guidance = dict(positions=self._tensor(np.stack(guidance_positions)),
                        yaw=self._tensor(np.stack(guidance_yaws)),
                        active=torch.as_tensor(np.stack(guidance_active)))
        return self._tensor(np.stack(actions)), self._tensor(np.stack(fields)), guidance

    def _targets(self, state=None):
        state = self._state() if state is None else state
        result = super()._targets(state)
        if not self.body_ready:
            return result
        target, wanted_yaw, obj, goal, phase, axis = result
        qr, _, yaw, _, _, _, _, _ = state
        stage = self.body_approach_stage
        sign = self.pair_sign[None, :]
        side = torch.where(self.assignment == 0, -1., 1.)[None, :]
        radius = self.object_radius[self.assignment][None, :]
        outboard = obj + torch.stack([torch.full_like(sign.expand_as(phase), .060),
                                       (side*.075).expand_as(phase)], -1)
        far_side = obj + torch.stack([-(radius+.055).expand_as(phase),
                                      (side*.075).expand_as(phase)], -1)
        pregrasp = obj + torch.stack([(sign*(radius+.042)).expand_as(phase), torch.zeros_like(phase)], -1)
        approach = torch.where((stage == 0)[..., None], outboard,
                              torch.where((stage == 1)[..., None], far_side, pregrasp))
        # Boundary partners leave in sequence. The outside module first rolls
        # straight clear of the packed shoulders before turning around its load.
        peel = torch.stack([obj[..., 0]+.060, self.body_initial[..., 1]], -1)
        approach = torch.where((stage == -2)[..., None], peel, approach)
        approach = torch.where((stage == -1)[..., None], qr[..., :2], approach)
        approaching = self.carrier & (phase == 0) & (stage < 5)
        target = torch.where(approaching[..., None], approach, target)
        delta = target-qr[..., :2]
        navigation_yaw = torch.atan2(-delta[..., 0], delta[..., 1])
        nav_error = torch.atan2(torch.sin(navigation_yaw-yaw), torch.cos(navigation_yaw-yaw))
        navigation_yaw = torch.where(nav_error.abs() > math.pi/2, navigation_yaw+math.pi, navigation_yaw)
        wanted_yaw = torch.where(approaching & (stage < 3), navigation_yaw, wanted_yaw)
        wanted_yaw = torch.where(approaching & (stage == -1), yaw, wanted_yaw)
        _, field, guidance = self._flow(state)
        flow_target = qr[..., :2] + field
        flow_yaw = torch.atan2(-field[..., 0], field[..., 1])
        speed = torch.linalg.vector_norm(field, dim=-1)
        flow_yaw = torch.where(speed > .001, flow_yaw, yaw)
        flow_target = torch.where(guidance['active'][..., None], guidance['positions'], flow_target)
        flow_yaw = torch.where(guidance['active'], guidance['yaw'], flow_yaw)
        # The whole body first clears the start area before boundary modules peel.
        deploy = self.body_phase[:, None] == 0
        use_flow = ~self.carrier | deploy | (phase == 5)
        target = torch.where(use_flow[..., None], flow_target, target)
        wanted_yaw = torch.where(use_flow, flow_yaw, wanted_yaw)
        return target, wanted_yaw, obj, goal, phase, axis

    def teacher_action(self):
        if not self.body_ready:
            return super().teacher_action()
        state = self._state()
        qr, qo, yaw, _, _, _, _, _ = state
        flow, _, _ = self._flow(state)
        contact_action = super().teacher_action()
        targets, headings, *_ = self._targets(state)
        action = flow.clone()
        for world in range(self.num_envs):
            if self.body_phase[world] == 0:
                continue
            for robot in torch.where(self.carrier[world])[0].tolist():
                obj = robot // 2
                phase = int(self.phase[world, obj])
                if phase == 5:
                    continue
                stage = int(self.body_approach_stage[world, robot])
                heading_error = math.atan2(math.sin(float(headings[world, robot]-yaw[world, robot])),
                                           math.cos(float(headings[world, robot]-yaw[world, robot])))
                if phase == 0 and stage < 3:
                    vector = (targets[world, robot]-qr[world, robot, :2]).numpy()
                    norm = np.linalg.norm(vector)
                    desired = vector / max(norm, 1e-6) * min(.035, 2*norm)
                    command = velocity_actions(desired[None], np.array([float(yaw[world, robot])]),
                                               config=APPROACH_FLOW)
                    action[world, robot] = self._tensor(command[0])
                    action[world, robot, 2:] = self._tensor([-1., -1.])
                elif phase == 0 and stage == 3:
                    yaw_rate = float(self.qvel[world, self.robot_dadr[robot]+5])
                    turn = np.clip(6*heading_error + .65*np.sign(heading_error)*(abs(heading_error) > .006)
                                   - .7*yaw_rate, -2., 2.)
                    action[world, robot] = self._tensor([-turn*.0105/.14, turn*.0105/.14, -1., -1.])
                elif phase == 0 and stage == 4:
                    delta = targets[world, robot]-qr[world, robot, :2]
                    forward = state[3][world, robot]
                    along = float(delta@forward)
                    speed = np.clip(2*along, -.012, .012)
                    yaw_rate = float(self.qvel[world, self.robot_dadr[robot]+5])
                    turn = np.clip(6*heading_error + .65*np.sign(heading_error)*(abs(heading_error) > .006)
                                   - .7*yaw_rate, -2., 2.)
                    if abs(heading_error) > .01:
                        speed = 0.
                    action[world, robot] = self._tensor([(speed-turn*.0105)/.14,
                                                        (speed+turn*.0105)/.14, -1., -1.])
                else:
                    action[world, robot, :3] = contact_action[world, robot]
                    if phase == 2 and self.body_phase[world] < 2:
                        remaining = float((self.goals[world, obj]-qo[world, obj, :2])@self.axis[world, obj])
                        carry = min(.014, max(0., .7*remaining)) * float(self.pair_sign[robot])
                        action[world, robot, :2] -= carry/.14
                    action[world, robot, 3] = 1 if phase in (1, 2, 3, 5) else -1
        return action

    def _step_native_world(self, world):
        if not self.body_ready:
            return super()._step_native_world(world)
        data = self.native_data[world]
        actions = self.command.numpy()[world]
        lift_target = self.lift_target.numpy()[world]
        enabled = (self.magnet_command[world].numpy() > 0) & self.active[world].numpy().astype(bool)
        phase = self.phase[world, self.assignment].numpy()
        cylinder = self.carrier[world].numpy() & self.object_is_cylinder[self.assignment.numpy()]
        differential_gain = np.where(cylinder & (phase == 2), .00045,
                                     np.where(cylinder & (phase == 0) & (self.body_approach_stage[world].numpy() >= 2),
                                              .00015, self.body_motor_gain[world]))
        for _ in range(self.substeps):
            wheel_velocity = data.qvel[self.wheel_dofs_np]
            error = -20*actions[:, :2]-wheel_velocity
            common = self.body_motor_gain[world]*error.mean(-1)
            differential = differential_gain*(error[:, 1]-error[:, 0])/2
            request = np.column_stack([common-differential, common+differential])
            limit = np.where(request*wheel_velocity > 0,
                             .002*np.maximum(0, 1-np.abs(wheel_velocity)/26.1799388), .002)
            data.ctrl[self.actuator_ids_np[:, :2]] = np.clip(request, -limit, limit)
            wanted = .004*(actions[:, 2]+1)
            lift_target += np.clip(wanted-lift_target, -.01*self.timestep, .01*self.timestep)
            data.ctrl[self.actuator_ids_np[:, 2]] = lift_target
            mujoco.mj_step1(self.model, data)
            self.magnets[world].apply(data, enabled)
            mujoco.mj_step2(self.model, data)
        mujoco.mj_forward(self.model, data)
        if any(w.number for w in data.warning):
            raise FloatingPointError(f"Native MuJoCo warning in magnetic world {world}")

    def _observation(self):
        base = super()._observation()
        if not self.body_ready:
            return base
        qr, _, _, forward, right, _, _, _ = self._state()
        positions = qr[..., :2]
        centroid = (positions*self.active[..., None]).sum(1)/self.active.sum(-1, keepdim=True).clamp(min=1)
        def local(vector):
            return torch.stack([(vector*right).sum(-1), (vector*forward).sum(-1)], -1)
        graph = self._tensor(self.body_links)
        degree = graph.sum(-1)/4
        components = torch.zeros_like(degree)
        for world in range(self.num_envs):
            seen = set()
            for robot in range(self.num_robots):
                if robot in seen:
                    continue
                group, pending = [], [robot]
                while pending:
                    item = pending.pop()
                    if item in seen:
                        continue
                    seen.add(item)
                    group.append(item)
                    pending.extend(np.flatnonzero(self.body_links[world, item]).tolist())
                components[world, group] = len(group)/self.num_robots
        docking_stage = self._tensor(np.stack([
            np.where(plan.release_steps > 0, -3, np.where(plan.waiting, -2, plan.stage))
            for plan in self.body_docking]))
        approach_stage = torch.where(self.carrier, self.body_approach_stage, 0).float()
        extra = torch.cat([local(centroid[:, None]-positions)/.2,
                           local(self.body_waypoint[:, None]-positions)/.4,
                           (self.magnet_command > 0).float()[..., None], degree[..., None],
                           components[..., None], self.body_phase[:, None, None].expand(-1, self.num_robots, 1)/3,
                           approach_stage[..., None]/5, docking_stage[..., None]/3], -1)
        base["local"] = torch.cat([base["local"], extra], -1).clamp(-10, 10)
        # Recompute exactly the base neighborhood indices to add docking state.
        delta = positions[:, None]-positions[:, :, None]
        distance = delta.square().sum(-1)
        distance[:, self.robot_indices, self.robot_indices] = float("inf")
        distance[:, self.robot_indices, self.partner] = -1
        nearest = distance.topk(min(6, self.num_robots-1), largest=False).indices
        ei = torch.arange(self.num_envs)[:, None, None]
        ri = self.robot_indices[None, :, None]
        extra_neighbors = torch.stack([graph[ei, ri, nearest],
                                       (self.magnet_command[ei, nearest] > 0).float(),
                                       degree[ei, nearest], components[ei, nearest]], -1)
        base["neighbors"] = torch.cat([base["neighbors"], extra_neighbors], -1)
        count = self.active.sum(-1, keepdim=True).clamp(min=1)
        mean = (base["local"]*self.active[..., None]).sum(1)/count
        maximum = torch.where(self.active[..., None].bool(), base["local"], -10.).max(1).values
        base["global"] = torch.cat([mean, maximum, base["global"][:, -8:]], -1)
        return base

    def step(self, actions):
        actions = torch.as_tensor(actions, device=self.device)
        if actions.shape != (self.num_envs, self.num_robots, 4) or not torch.isfinite(actions).all():
            raise ValueError("Magnetic policy must provide four finite actions per module")
        old_centroid = self._state()[0][..., :2].mean(1)
        old_distance = torch.linalg.vector_norm(self.body_waypoint-old_centroid, dim=-1)
        previous_links = self.body_links.copy()
        previous_enabled = self.magnet_command > 0
        self.magnet_command.copy_(actions[..., 3].detach().clamp(-1, 1))
        self.body_release_started |= (self.magnet_command <= 0).any(-1)
        state = self._state()
        qr, qo, yaw, _, _, _, _, _ = state
        target, wanted_yaw, _, _, _, _ = super()._targets(state)
        heading_error = torch.atan2(torch.sin(wanted_yaw-yaw), torch.cos(wanted_yaw-yaw))
        aligned = (torch.linalg.vector_norm(target-qr[..., :2], dim=-1) < .015) & (heading_error.abs() < .2)
        phase = self.phase[:, self.assignment]
        pinch = self.carrier & (phase < 5) & ((phase > 0) | (aligned & (self.body_approach_stage >= 5)))
        self.body_motor_gain[:] = np.where(pinch.numpy(), .00015, .00045)
        _, _, _, truncated, info = super().step(actions[..., :3])
        for world, magnets in enumerate(self.magnets):
            self.body_links[world] = magnets.load_bearing_graph()
        qr, qo, _, _, _, velocity, _, _ = self._state()
        centroid = qr[..., :2].mean(1)
        distance = torch.linalg.vector_norm(self.body_waypoint-centroid, dim=-1)
        self.body_phase = torch.where((self.body_phase == 0) & (centroid[:, 0] < .81), 1, self.body_phase)
        target, wanted_yaw, _, _, phase, _ = self._targets()
        yaw = self._state()[2]
        stage = self.body_approach_stage
        error = torch.linalg.vector_norm(target-qr[..., :2], dim=-1)
        heading_error = torch.atan2(torch.sin(wanted_yaw-yaw), torch.cos(wanted_yaw-yaw))
        tolerance = torch.where(stage == 2, .0015, .010)
        arrived = ((stage >= 0) & (stage < 3) & (error < tolerance)) | ((stage == 3) & (heading_error.abs() < .006) & (error < .003))
        arrived &= self.carrier & (phase == 0) & (self.body_phase[:, None] > 0)
        self.body_approach_stage = torch.where(arrived, (stage+1).clamp(max=4), stage)
        peeled = (stage == -2) & (error < .003) & (self.body_phase[:, None] > 0)
        self.body_approach_stage = torch.where(peeled, 0, self.body_approach_stage)
        partner_outside = self.body_approach_stage[:, self.partner] >= 1
        waiting = (stage == -1) & partner_outside & (self.body_phase[:, None] > 0)
        self.body_approach_stage = torch.where(waiting, 2, self.body_approach_stage)
        reposition = self.carrier & (phase == 0) & (stage >= 3) & (stage < 5) & (error > .003)
        self.body_approach_stage = torch.where(reposition, 2, self.body_approach_stage)
        aligned_ready = (self.body_approach_stage >= 4) & ((stage >= 5) | ((error < .003) & (heading_error.abs() < .008)))
        aligned_ready &= self.qvel[:, self.robot_dadr+5].abs() < .04
        pair_ready = aligned_ready[:, :2*self.num_objects].reshape(self.num_envs, self.num_objects, 2).all(-1)
        self.body_approach_stage = torch.where(self.carrier & pair_ready[:, self.assignment], 5, self.body_approach_stage)
        # Steer the supported grasp gradually toward the measured remaining
        # goal vector, so lateral drift cannot strand a payload beside its goal.
        remaining = self.goals-qo[..., :2]
        wanted_angle = torch.atan2(remaining[..., 1], remaining[..., 0])
        current_angle = torch.atan2(self.axis[..., 1], self.axis[..., 0])
        correction = torch.atan2(torch.sin(wanted_angle-current_angle), torch.cos(wanted_angle-current_angle))
        corrected_angle = current_angle + correction.clamp(-.005, .005)
        direction = torch.stack([torch.cos(corrected_angle), torch.sin(corrected_angle)], -1)
        steer = (self.phase == 2) & (torch.linalg.vector_norm(remaining, dim=-1) > .009)
        self.axis = torch.where(steer[..., None], direction, self.axis)
        ready_to_carry = ((self.phase >= 2) | ~self.object_active).all(-1)
        self.body_phase = torch.where(ready_to_carry & (self.body_phase < 2), 2, self.body_phase)
        released = info["delivered"] == self.object_active.sum(-1)
        self.body_phase = torch.where(released, 3, self.body_phase)
        mean_payload = (qo[..., :2]*self.object_active[..., None]).sum(1)/self.object_active.sum(-1, keepdim=True).clamp(min=1)
        rearmost_payload = torch.where(self.object_active, qo[..., 0], -torch.inf).amax(-1)
        pickup_center = torch.stack([rearmost_payload+PAYLOAD_BODY_STANDOFF, mean_payload[:, 1]], -1)
        self.body_waypoint = torch.where((self.body_phase == 1)[:, None], self._tensor([.80, .885]), self.body_waypoint)
        self.body_waypoint = torch.where((self.body_phase == 2)[:, None], pickup_center, self.body_waypoint)
        regroup_center = mean_payload + self._tensor([PAYLOAD_BODY_STANDOFF, 0.])
        self.body_waypoint = torch.where((self.body_phase == 3)[:, None], regroup_center, self.body_waypoint)
        self._update_docking(self._state())
        metrics = []
        for world in range(self.num_envs):
            metrics.append(body_telemetry(qr[world, :, :2].numpy(), velocity[world].numpy(),
                                          self.body_links[world], initial_positions=self.body_initial[world].numpy(),
                                          previous_links=previous_links[world],
                                          link_forces=self.magnets[world].pair_force_magnitudes_n))
            metrics[-1]["close_dock_edges"] = int(np.triu(self.magnets[world].contact_graph(), 1).sum())
        formations = np.triu(self.body_links & ~previous_links, 1).sum((1, 2))
        releases = np.triu(previous_links & ~self.body_links, 1).sum((1, 2))
        self.body_link_formations += torch.as_tensor(formations)
        self.body_link_releases += torch.as_tensor(releases)
        component = torch.tensor([max(self._component_sizes(g)) for g in self.body_links])
        connected = component >= max(2, math.ceil(.8*self.num_robots))
        self.body_connected_steps += connected
        before_release = ~self.body_release_started
        self.body_initial_links |= self.body_links & before_release.numpy()[:, None, None]
        self.body_new_links |= (self.body_links & ~previous_links & ~self.body_initial_links &
                               self.body_release_started.numpy()[:, None, None])
        new_neighbors = torch.as_tensor(np.triu(self.body_new_links, 1).sum((1, 2)))
        self.body_initial_connected_run = torch.where(before_release & (component == self.num_robots),
                                                      self.body_initial_connected_run+1, 0)
        self.body_initial_connected_best = torch.maximum(self.body_initial_connected_best,
                                                         self.body_initial_connected_run)
        initial_connected_seconds = (self.body_initial_connected_best-1).clamp(min=0)*self.dt
        self.body_middle_steps += self.body_release_started
        self.body_middle_connected_steps += self.body_release_started & connected
        middle_connected_fraction = self.body_middle_connected_steps/self.body_middle_steps.clamp(min=1)
        # Contact itself is expected. Penalize excessive penetration separately.
        severe = torch.zeros((self.num_envs, self.num_robots))
        payload_robot_contacts = torch.zeros(self.num_envs, dtype=torch.long)
        for world, data in enumerate(self.native_data):
            for contact in data.contact[:data.ncon]:
                a, b = self.geom_robot[contact.geom1], self.geom_robot[contact.geom2]
                if a >= 0 and b >= 0 and a != b and contact.dist < -.0005:
                    severe[world, a] = severe[world, b] = 1
                if (a >= 0 and self.geom_object[contact.geom2] >= 0) or (b >= 0 and self.geom_object[contact.geom1] >= 0):
                    payload_robot_contacts[world] += 1
        original_components = info["reward_components"]
        def shared(value):
            return value.expand(-1, self.num_robots)
        carrier_count = self.carrier.sum(-1, keepdim=True).clamp(min=1)
        components = {key: shared(original_components[key].sum(-1, keepdim=True)/carrier_count)
                      for key in ("transport_progress", "pickup", "delivery", "drop")}
        components.update(body_progress=shared(8*(old_distance-distance)[:, None]),
                          magnetic_link_progress=shared(.002*self._tensor(formations-releases)[:, None]/self.num_robots),
                          disconnected=shared(-.002*(1-component[:, None]/self.num_robots)),
                          severe_robot_contact=-.02*severe,
                          magnet_switch=-.0001*(previous_enabled != (self.magnet_command > 0)).float(),
                          energy=original_components["energy"],
                          action_change=original_components["action_change"],
                          outside=original_components["outside"])
        reward = sum(components.values())
        travel = torch.linalg.vector_norm(qr[..., :2]-self.body_initial, dim=-1)
        gathered = torch.linalg.vector_norm(qr[..., :2]-centroid[:, None], dim=-1).amax(-1) < .34
        settled = torch.linalg.vector_norm(velocity, dim=-1).amax(-1) < .015
        finished = (released & (component == self.num_robots) & gathered & settled &
                    (payload_robot_contacts == 0) & (travel.amin(-1) > .20) &
                    (initial_connected_seconds >= 1.) &
                    (middle_connected_fraction >= .95) & (new_neighbors >= 1) &
                    (self.body_link_releases >= 2))
        self.body_hold = torch.where(finished, self.body_hold+1, 0)
        success = self.body_hold >= 50
        failure = info["failure"]
        info.update(success=success, reward_components=components,
                    body_metrics=metrics, body_phase=self.body_phase.clone(),
                    largest_magnetic_component=component, magnetic_edges=self._tensor(np.triu(self.body_links, 1).sum((1, 2))),
                    magnetic_link_formations=self.body_link_formations.clone(), magnetic_link_releases=self.body_link_releases.clone(),
                    severe_robot_contacts=severe.sum(-1), all_robot_min_travel=travel.amin(-1),
                    payload_robot_contacts=payload_robot_contacts,
                    initial_connected_seconds=initial_connected_seconds,
                    middle_connected_control_fraction=middle_connected_fraction,
                    new_magnetic_neighbor_pairs=new_neighbors,
                    connected_control_steps=self.body_connected_steps.clone())
        truncated = (self.steps >= self.max_steps) & ~success & ~failure
        return self._observation(), reward*self.active, success | failure, truncated, info

    @staticmethod
    def _component_sizes(graph):
        seen, sizes = set(), []
        for robot in range(len(graph)):
            if robot in seen:
                continue
            pending, count = [robot], 0
            while pending:
                item = pending.pop()
                if item in seen:
                    continue
                seen.add(item)
                count += 1
                pending.extend(np.flatnonzero(graph[item]).tolist())
            sizes.append(count)
        return sizes
