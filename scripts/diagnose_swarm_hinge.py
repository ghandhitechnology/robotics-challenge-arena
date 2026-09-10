#!/usr/bin/env python3
"""Measure hinge-only motion from joint torques and passive shell contact."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.swarm_hinge_robot import HINGE_DESIGN, add_hinge_robot, hinge_motor_torques


@dataclass(frozen=True)
class Gait:
    frequency_hz: float = .7
    center_rad: float = -.35
    amplitude_rad: float = .25
    phase_rad: float = math.pi/2
    direction: float = 1.
    steering_rad: float = 0.
    steering_phase_rad: float = 0.

    def __post_init__(self):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("gait parameters must be finite")
        if self.frequency_hz <= 0 or self.amplitude_rad < 0 or self.direction not in (-1., 1.):
            raise ValueError("gait needs positive frequency, nonnegative amplitude, and direction -1 or 1")
        if abs(self.center_rad)+self.amplitude_rad > HINGE_DESIGN['joint_limits_rad'][1][1]:
            raise ValueError("pitch targets exceed the hinge travel")
        if abs(self.steering_rad) > HINGE_DESIGN['joint_limits_rad'][0][1]:
            raise ValueError("steering target exceeds the yaw hinge travel")

    def target(self, time_s):
        phase = 2*math.pi*self.frequency_hz*time_s
        ramp = min(max(time_s*self.frequency_hz, 0.), 1.)
        return np.array([self.steering_rad*ramp*(.5+.5*math.sin(phase+self.steering_phase_rad)),
                         self.center_rad+ramp*self.amplitude_rad*math.sin(phase),
                         self.center_rad+ramp*self.amplitude_rad*math.sin(phase+self.direction*self.phase_rad)])


def scene(friction=.7, timestep=.001, initial_yaw=0.):
    root = ET.Element("mujoco", model="hinge_only_module")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=str(timestep), gravity="0 0 -9.81", integrator="implicitfast",
                  cone="elliptic", iterations="50", tolerance="1e-10")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="board", type="plane", size="1 1 .01", rgba=".85 .84 .80 1",
                  friction=f"{friction} .00001 .000001", condim="3")
    ET.SubElement(world, "light", pos=".2 -.1 .5", diffuse=".9 .9 .9")
    ET.SubElement(world, "camera", name="side", pos=".13 -.10 .07", xyaxes=".61 .79 0 -.27 .21 .94")
    spec = add_hinge_robot(root, friction=friction, yaw=initial_yaw)
    return ET.tostring(root, encoding="unicode"), spec


def heading(quaternion):
    w, x, y, z = quaternion
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def run(gait, *, seconds=10., friction=.7, timestep=.001, initial_yaw=0., save=None):
    if seconds <= 0 or timestep <= 0 or not all(math.isfinite(value) for value in (seconds, timestep, initial_yaw)):
        raise ValueError("duration and timestep must be positive and finite")
    xml, spec = scene(friction, timestep, initial_yaw)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    joints = [model.joint(name) for name in spec["joint_names"]]
    qadr = np.array([joint.qposadr[0] for joint in joints])
    dadr = np.array([joint.dofadr[0] for joint in joints])
    motors = np.array([model.actuator(name).id for name in spec["motor_names"]])
    robot = model.body(spec["name"]).id
    colliders = np.array([model.geom(name).id for name in spec["collider_names"]])
    board = model.geom("board").id
    warmup = 1.5
    duration = warmup+seconds
    samples = dict(time=[], qpos=[], qvel=[], actions=[], target_angles_rad=[], motor_torques_nm=[],
                   contact_normal_n=[], contact_count=[], contact_slip_speed_m_s=[], com=[])
    peak_torque = peak_speed = peak_penetration = external_force = peak_roll_pitch = 0.
    contact_steps = np.zeros(3, int)
    contact_impulse = np.zeros(3)
    slip_impulse = np.zeros(3)
    sticking_steps = np.zeros(3, int)
    airborne_steps = 0
    energy = 0.
    min_height, max_height = np.inf, -np.inf
    shell_top = 0.
    start = start_com = start_heading = None
    contact_detail = []
    t0 = time.monotonic()
    sample_every = max(1, round(.01/timestep))
    for step in range(round(duration/timestep)):
        gait_time = max(0., data.time-warmup)
        target = gait.target(gait_time)
        torque = hinge_motor_torques(data.qpos[qadr], data.qvel[dadr], target)
        data.ctrl[motors] = torque
        mujoco.mj_step(model, data)
        if data.time >= warmup and start is None:
            start, start_com, start_heading = data.qpos[:3].copy(), data.subtree_com[robot].copy(), heading(data.qpos[3:7])
        if data.time < warmup:
            continue
        peak_torque = max(peak_torque, float(np.max(np.abs(torque))))
        peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel[dadr]))))
        external_force = max(external_force, float(np.abs(data.xfrc_applied).max()), float(np.abs(data.qfrc_applied).max()))
        energy += float(np.maximum(torque*data.qvel[dadr], 0).sum())*timestep
        min_height, max_height = min(min_height, data.qpos[2]), max(max_height, data.qpos[2])
        up = data.xmat[robot].reshape(3, 3)[:, 2]
        peak_roll_pitch = max(peak_roll_pitch, math.acos(np.clip(up[2], -1., 1.)))
        normal = np.zeros(3)
        weighted_slip = np.zeros(3)
        count = np.zeros(3, int)
        detail = []
        for index, contact in enumerate(data.contact[:data.ncon]):
            if board not in (contact.geom1, contact.geom2):
                continue
            body_geom = contact.geom2 if contact.geom1 == board else contact.geom1
            which = np.flatnonzero(colliders == body_geom)
            if not len(which):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            normal_force = max(force[0], 0.)
            normal[which[0]] += normal_force
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, contact.pos, model.geom_bodyid[body_geom])
            velocity = jacobian@data.qvel
            direction = contact.frame[:3]
            slip = float(np.linalg.norm(velocity-direction*(velocity@direction)))
            weighted_slip[which[0]] += normal_force*slip
            count[which[0]] += 1
            peak_penetration = max(peak_penetration, max(0., -contact.dist))
            if save is not None and step % sample_every == 0:
                detail.append(dict(geom=model.geom(body_geom).name, position=contact.pos.tolist(),
                                   distance_m=float(contact.dist), force_contact_frame=force.tolist(),
                                   tangent_speed_m_s=slip))
        mean_slip = weighted_slip/np.maximum(normal, 1e-12)
        contact_steps += normal > .001
        contact_impulse += normal*timestep
        slip_impulse += weighted_slip*timestep
        sticking_steps += (normal > .001) & (mean_slip < .001)
        airborne_steps += normal.sum() <= .001
        if save is not None and step % sample_every == 0:
            values = dict(time=data.time-warmup, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                          actions=target/np.asarray(spec['joint_limits_rad'])[:, 1], target_angles_rad=target.copy(),
                          motor_torques_nm=torque.copy(), contact_normal_n=normal, contact_count=count,
                          contact_slip_speed_m_s=mean_slip,
                          com=data.subtree_com[robot].copy())
            for key, value in values.items():
                samples[key].append(value)
            contact_detail.append(detail)
        if step % sample_every == 0:
            shell_top = max(shell_top, data.geom_xpos[colliders[0], 2]+.008)
            for geom in colliders[1:]:
                mesh = model.geom_dataid[geom]
                start_vertex, count_vertex = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start_vertex:start_vertex+count_vertex]
                world_vertices = vertices@data.geom_xmat[geom].reshape(3, 3).T+data.geom_xpos[geom]
                shell_top = max(shell_top, world_vertices[:, 2].max())
    mujoco.mj_forward(model, data)
    delta = data.qpos[:3]-start
    com_delta = data.subtree_com[robot]-start_com
    yaw_delta = heading(data.qpos[3:7])-start_heading
    yaw_delta = math.atan2(math.sin(yaw_delta), math.cos(yaw_delta))
    result = dict(gait=asdict(gait), seconds=seconds, timestep=timestep, friction=friction,
                  root_displacement_m=delta.tolist(), com_displacement_m=com_delta.tolist(),
                  forward_mm=float(delta[:2]@np.array([-math.sin(initial_yaw), math.cos(initial_yaw)])*1000),
                  lateral_mm=float(delta[:2]@np.array([math.cos(initial_yaw), math.sin(initial_yaw)])*1000),
                  speed_mm_s=float(delta[:2]@np.array([-math.sin(initial_yaw), math.cos(initial_yaw)])*1000/seconds),
                  yaw_change_deg=math.degrees(yaw_delta), initial_yaw_deg=math.degrees(initial_yaw),
                  settled_root_height_mm=float(start[2]*1000), maximum_shell_height_mm=float(shell_top*1000),
                  root_height_range_mm=[min_height*1000, max_height*1000],
                  maximum_tilt_deg=math.degrees(peak_roll_pitch), peak_motor_torque_nm=peak_torque,
                  peak_joint_speed_rad_s=peak_speed, positive_motor_work_j=energy,
                  contact_fraction=(contact_steps/(seconds/timestep)).tolist(),
                  sticking_fraction=(sticking_steps/(seconds/timestep)).tolist(),
                  force_weighted_slip_mm_s=(1000*slip_impulse/np.maximum(contact_impulse, 1e-12)).tolist(),
                  airborne_fraction=airborne_steps/(seconds/timestep),
                  peak_penetration_mm=peak_penetration*1000, external_applied_force_max=external_force,
                  warnings=int(data.warning.number.sum()), wall_seconds=time.monotonic()-t0,
                  total_mass_kg=float(model.body_mass.sum()))
    if save is not None:
        output = Path(save)
        output.mkdir(parents=True, exist_ok=True)
        (output/"scene.xml").write_text(xml)
        (output/"metadata.json").write_text(json.dumps(spec, indent=2))
        (output/"metrics.json").write_text(json.dumps(result, indent=2))
        (output/"contacts.json").write_text(json.dumps(contact_detail))
        np.savez_compressed(output/"trajectory.npz", **{key: np.asarray(value) for key, value in samples.items()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--frequency", type=float, default=.7)
    parser.add_argument("--center", type=float, default=-.35)
    parser.add_argument("--amplitude", type=float, default=.25)
    parser.add_argument("--phase", type=float, default=90.)
    parser.add_argument("--direction", type=float, default=-1., choices=(-1., 1.),
                        help="rear pitch phase order: -1 crawls forward; +1 crawls backward")
    parser.add_argument("--steering", type=float, default=0.)
    parser.add_argument("--steering-phase", type=float, default=0.)
    parser.add_argument("--friction", type=float, default=.7)
    parser.add_argument("--timestep", type=float, default=.001)
    parser.add_argument("--output", type=Path, default=Path("tmp/swarm_hinge_single"))
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    if args.search:
        args.output.mkdir(parents=True, exist_ok=True)
        results = []
        for frequency, center, amplitude, phase in itertools.product((.4, .7, 1., 1.3), (-.2, -.35, -.45), (.15, .25, .35), (60., 90., 120.)):
            if abs(center)+amplitude > math.radians(40):
                continue
            gait = Gait(frequency, center, amplitude, math.radians(phase))
            result = run(gait, seconds=args.seconds, friction=args.friction, timestep=args.timestep)
            results.append(result)
            print(json.dumps(result), flush=True)
        results.sort(key=lambda row: abs(row['speed_mm_s']), reverse=True)
        (args.output/"search.json").write_text(json.dumps(results, indent=2))
    else:
        gait = Gait(args.frequency, args.center, args.amplitude, math.radians(args.phase), args.direction,
                    math.radians(args.steering), math.radians(args.steering_phase))
        print(json.dumps(run(gait, seconds=args.seconds, friction=args.friction, timestep=args.timestep, save=args.output)), flush=True)


if __name__ == "__main__":
    main()
