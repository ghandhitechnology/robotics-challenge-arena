"""Hinge-driven prototype using the existing tapered shells and central housing.

Only motor torques and passive shell contacts move the free module. Local +Y
is forward. This builder is separate from the preserved wheel-based baseline.
"""
from __future__ import annotations

import copy
import math
from xml.etree.ElementTree import Element, SubElement

import numpy as np

from .swarm_robot import _numbers, _section


HINGE_DESIGN = {
    "name": "swarm_hinge_module", "schema_version": 1,
    "mass_kg": .040, "neutral_shell_bounds_m": [[-.010, -.027, -.005], [.010, .027, .011]],
    "joint_names": ["front_yaw", "front_pitch", "rear_pitch"],
    "joint_limits_rad": [[-.5235987756, .5235987756], [-.6981317008, .6981317008], [-.6981317008, .6981317008]],
    "max_motor_torque_nm": .008, "motor_no_load_speed_rad_s": 45*2*math.pi/60,
    "position_gain_nm_rad": .04, "velocity_gain_nm_s_rad": .0014,
    "joint_armature_kg_m2": .000005, "shell_friction": .7,
    "physical_motor_count": 3,
    "hardware": "Three Pololu 2359-class 700:1 geared motors, joint encoders and custom hinge transmissions",
    "sources": ["https://www.pololu.com/product/2359"],
    "assumptions": ["40 g total assembly: 12 g central housing and 14 g per lobe",
                    "8 mN m operating torque cap and estimated 5e-6 kg m2 reflected inertia",
                    "Uniform isotropic shell friction; no separately actuated contact surfaces",
                    "Motor packaging, shell coating, and thermal duty cycle require prototype measurements"],
}


def add_hinge_robot(root: Element, index=0, position=(0., 0., .025), yaw=0., *, friction=.7):
    """Append a free three-body module with two pitch joints and one yaw joint."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("index must be a nonnegative integer")
    if len(position) != 3 or not all(math.isfinite(float(v)) for v in (*position, yaw, friction)) or friction < 0:
        raise ValueError("pose and friction must be finite; friction must be nonnegative")
    prefix = f"hinge_{index:02d}"
    if root.find(f".//body[@name='{prefix}']") is not None:
        raise ValueError(f"Duplicate hinge module index: {index}")
    asset, world, actuator, sensor = [_section(root, tag) for tag in ("asset", "worldbody", "actuator", "sensor")]
    compiler = root.find("compiler")
    radians = compiler is not None and compiler.get("angle", "degree") == "radian"
    # These are the same broad-shouldered, chamfered shell vertices used by the
    # wheel module. The shell itself supplies every distal ground contact.
    if asset.find("mesh[@name='swarm_hinge_lobe']") is None:
        vertices = []
        for y, half_width, zlow, zhigh in ((.007, .010, -.004, .010), (.027, .0045, -.004, .007)):
            bevel = .0015
            for x, z in ((-half_width+bevel, zlow), (half_width-bevel, zlow),
                         (half_width, zlow+bevel), (half_width, zhigh-bevel),
                         (half_width-bevel, zhigh), (-half_width+bevel, zhigh),
                         (-half_width, zhigh-bevel), (-half_width, zlow+bevel)):
                vertices.extend((x, y, z))
        SubElement(asset, "mesh", name="swarm_hinge_lobe", vertex=_numbers(vertices))
    base = SubElement(world, "body", name=prefix, pos=_numbers(position),
                      quat=_numbers([math.cos(yaw/2), 0, 0, math.sin(yaw/2)]))
    SubElement(base, "freejoint", name=f"{prefix}_free")
    contact = dict(contype="4", conaffinity="7", condim="3", friction=f"{friction} .00001 .000001",
                   solref=".004 1", solimp=".9 .99 .0001", priority="2")
    SubElement(base, "geom", name=f"{prefix}_ball_housing", type="sphere", pos="0 0 .003",
               size=".008", mass=".012", rgba=".055 .06 .065 1", **contact)
    joints, motors = [], []
    colliders = [dict(name=f"{prefix}_ball_housing", body=prefix, material="coated_shell",
                      grip_pad=False, robot_part=True)]
    for side, sign in (("front", 1), ("rear", -1)):
        lobe = SubElement(base, "body", name=f"{prefix}_{side}", pos=f"0 {sign*.006} .003",
                          quat="1 0 0 0" if sign == 1 else "0 0 0 1")
        axes = (("front_yaw", "0 0 1", HINGE_DESIGN["joint_limits_rad"][0]),) if sign == 1 else ()
        axes += ((f"{side}_pitch", "1 0 0", HINGE_DESIGN["joint_limits_rad"][1]),)
        for short, axis, limits in axes:
            joint, motor = f"{prefix}_{short}", f"{prefix}_{short}_motor"
            SubElement(lobe, "joint", name=joint, type="hinge", axis=axis, limited="true",
                       range=_numbers(limits if radians else [math.degrees(value) for value in limits]),
                       damping=".00002", frictionloss=".00008",
                       armature=str(HINGE_DESIGN["joint_armature_kg_m2"]),
                       solreflimit=".004 1", solimplimit=".95 .99 .0001")
            SubElement(actuator, "motor", name=motor, joint=joint, gear="1",
                       ctrllimited="true", ctrlrange="-.008 .008", forcelimited="true", forcerange="-.008 .008")
            SubElement(sensor, "jointpos", name=f"{joint}_position", joint=joint)
            SubElement(sensor, "jointvel", name=f"{joint}_speed", joint=joint)
            joints.append(joint)
            motors.append(motor)
        shell = f"{prefix}_shell_{side}"
        SubElement(lobe, "geom", name=shell, type="mesh", mesh="swarm_hinge_lobe", pos="0 -.006 -.003",
                   mass=".014", rgba=".43 .47 .45 1", **contact)
        colliders.append(dict(name=shell, body=lobe.get("name"), material="coated_shell",
                              grip_pad=False, robot_part=True))
    spec = copy.deepcopy(HINGE_DESIGN)
    spec.update(index=index, name=prefix, freejoint=f"{prefix}_free", joint_names=joints,
                motor_names=motors, colliders=colliders, collider_names=[item['name'] for item in colliders],
                initial_position_m=list(position),
                initial_yaw_rad=float(yaw), shell_friction=float(friction))
    return spec


def hinge_motor_torques(angles, velocities, target_angles):
    """Bounded joint servo, including the geared motor's torque-speed envelope."""
    angles, velocities, target = np.broadcast_arrays(np.asarray(angles, float),
                                                     np.asarray(velocities, float),
                                                     np.asarray(target_angles, float))
    if angles.ndim < 1 or angles.shape[-1] != 3 or not all(np.isfinite(x).all() for x in (angles, velocities, target)):
        raise ValueError("hinge motor inputs need three finite joint values")
    limits = np.asarray(HINGE_DESIGN["joint_limits_rad"])
    target = np.clip(target, limits[:, 0], limits[:, 1])
    requested = (HINGE_DESIGN["position_gain_nm_rad"]*(target-angles)
                 - HINGE_DESIGN["velocity_gain_nm_s_rad"]*velocities)
    propulsive = requested*velocities > 0
    limit = HINGE_DESIGN["max_motor_torque_nm"]*np.where(
        propulsive, np.maximum(0., 1-np.abs(velocities)/HINGE_DESIGN["motor_no_load_speed_rad_s"]), 1.)
    return np.clip(requested, -limit, limit)
