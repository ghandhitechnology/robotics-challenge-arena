"""A parameterized differential-drive robot for exercising arena contacts."""
from __future__ import annotations

import math
from xml.etree.ElementTree import Element, SubElement


def add_robot(worldbody: Element, actuator: Element, sensor: Element,
              profile: dict) -> dict:
    """Append a 1 kg reference robot; return names needed by the controller.

    The chassis faces +Y. Both wheel hinges point +X, so negative wheel
    angular velocity drives the robot forward. Dimensions are meters.
    """
    settings = profile.get("robot", profile)
    torque = float(settings.get("max_motor_torque_nm", 0.12))
    torque *= float(settings.get("motor_strength_scale", 1.0))
    speed = float(settings.get("max_wheel_speed_rad_s", 20.0))
    gain = float(settings.get("velocity_gain", 0.04))
    if not all(math.isfinite(v) and v > 0 for v in (torque, speed, gain)):
        raise ValueError("Robot torque, speed, and velocity gain must be positive and finite")
    root = SubElement(worldbody, "body", name="reference_robot", pos="1.02 0.78 0.045")
    SubElement(root, "freejoint", name="robot_free")
    SubElement(root, "geom", name="robot_chassis", type="box", size="0.06 0.07 0.018",
               mass="0.70", rgba="0.16 0.22 0.30 1", friction="0.35 0.0002 0.00001")
    # The low front lip can push the arena's 20 mm high cylinders.
    SubElement(root, "geom", name="robot_bumper", type="box", size="0.056 0.005 0.012",
               pos="0 0.065 -0.027", mass="0.05", rgba="0.25 0.31 0.38 1",
               friction="0.35 0.0002 0.00001")
    colliders = [{"name": name, "material": "plastic", "body": "reference_robot"}
                 for name in ("robot_chassis", "robot_bumper")]
    for side, x in (("left", -0.067), ("right", 0.067)):
        wheel = SubElement(root, "body", name=f"wheel_{side}_body", pos=f"{x} 0.018 -0.020")
        SubElement(wheel, "joint", name=f"wheel_{side}", type="hinge", axis="1 0 0",
                   damping="0.001", frictionloss="0.002", armature="0.00005")
        SubElement(wheel, "geom", name=f"wheel_{side}_geom", type="cylinder",
                   size="0.025 0.006", quat="0.7071067811865476 0 0.7071067811865476 0",
                   mass="0.10", rgba="0.065 0.065 0.07 1", condim="6",
                   friction="0.95 0.0002 0.00003")
        SubElement(actuator, "motor", name=f"motor_{side}", joint=f"wheel_{side}",
                   gear="1", ctrllimited="true", ctrlrange=f"{-torque} {torque}",
                   forcelimited="true", forcerange=f"{-torque} {torque}")
        SubElement(sensor, "jointpos", name=f"encoder_{side}", joint=f"wheel_{side}")
        SubElement(sensor, "jointvel", name=f"wheel_speed_{side}", joint=f"wheel_{side}")
        colliders.append({"name": f"wheel_{side}_geom", "material": "rubber",
                          "body": f"wheel_{side}_body"})
    caster = SubElement(root, "body", name="robot_caster", pos="0 -0.055 -0.033")
    SubElement(caster, "joint", name="caster_ball", type="ball", damping="0.00001")
    SubElement(caster, "geom", name="caster_geom", type="sphere", size="0.012",
               mass="0.05", rgba="0.5 0.52 0.54 1", condim="6",
               friction="0.35 0.0001 0.000005")
    colliders.append({"name": "caster_geom", "material": "plastic", "body": "robot_caster"})
    SubElement(root, "site", name="robot_imu", pos="0 0 0", size="0.002", rgba="0 0 0 0")
    SubElement(sensor, "framepos", name="robot_position", objtype="body", objname="reference_robot")
    SubElement(sensor, "framequat", name="robot_orientation", objtype="body", objname="reference_robot")
    SubElement(sensor, "gyro", name="robot_gyro", site="robot_imu")
    SubElement(sensor, "accelerometer", name="robot_accelerometer", site="robot_imu")
    return {
        "name": "reference_robot", "freejoint": "robot_free", "mass_kg": 1.0,
        "chassis_size_m": [0.12, 0.14, 0.036], "wheel_radius_m": 0.025,
        "wheel_track_m": 0.134, "wheel_joints": ["wheel_left", "wheel_right"],
        "motor_names": ["motor_left", "motor_right"], "wheel_direction_sign": -1.0,
        "max_motor_torque_nm": torque, "max_wheel_speed_rad_s": speed,
        "velocity_gain": gain, "colliders": colliders,
        "description": "Reference differential-drive robot; replace with measured competition hardware.",
    }
