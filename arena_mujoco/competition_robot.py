"""Compact four-motor preliminary robot with contact-only interchangeable jaws.

Coordinates match the reference robot: +Y forward, +X right, +Z up. The one
jaw motor drives two opposite rack slides through a native joint equality.
Objects remain free bodies; grasping uses only the rubber/wood contact cone.
"""
from __future__ import annotations

import copy
import math
from xml.etree.ElementTree import Element, SubElement

DESIGN = {
    "name": "competition_robot",
    "scope": "senior_preliminary",
    "footprint_m": [0.180, 0.200],
    "footprint_bounds_local_m": [[-0.090, -0.075], [0.090, 0.125]],
    "mass_kg": 0.800,
    "wheel_radius_m": 0.020,
    "wheel_track_m": 0.170,
    "wheel_axle_y_m": 0.005,
    "wheel_direction_sign": -1.0,
    "max_motor_torque_nm": 0.025,
    "max_wheel_speed_rad_s": 20.0,
    "motor_no_load_speed_rad_s": 430 * 2 * math.pi / 60,
    "velocity_gain": 0.008,
    "lift_range_m": [0.0, 0.080],
    "max_lift_speed_m_s": 0.080,
    "max_lift_force_n": 4.0,
    "jaw_joint_range_m": [0.0, 0.025],
    "jaw_aperture_range_m": [0.018, 0.068],
    "max_jaw_speed_m_s": 0.050,
    "max_grip_generalized_force_n": 2.0,
    "nominal_normal_force_per_jaw_n": 1.0,
    "grasp_center_body_m": [0.0, 0.105, -0.0159],
    "upper_grasp_center_body_m": [0.0, 0.105, -0.009],
    "sample_pad_floor_heights_m": [0.0034, 0.0048],
    "upper_pad_floor_heights_m": [0.006, 0.016],
    "physical_motor_count": 4,
    "hardware": {
        "drive": "2 × Pololu #5188, 75:1 HPCB 6V, 12 CPR encoder",
        "wheel": "Pololu #1452, 40 × 7 mm silicone tire, 3 mm D shaft",
        "lift_and_grip": "2 × ROBOTIS XL330-M288-T, regulated 5 V",
    },
}


def _numbers(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def _section(root: Element, tag: str) -> Element:
    found = root.find(tag)
    return found if found is not None else SubElement(root, tag)


def add_competition_robot(root: Element, profile: dict | None = None) -> dict:
    """Append the robot to an MJCF root and return controller/build metadata.

    Position targets must be slew-limited by the runtime using the returned
    speed limits. Actuator force bounds apply inside the native solver.
    """
    settings = dict((profile or {}).get("robot", profile or {}))
    spec = copy.deepcopy(DESIGN)
    for key in ("max_motor_torque_nm", "max_wheel_speed_rad_s",
                "motor_no_load_speed_rad_s", "velocity_gain", "max_lift_speed_m_s",
                "max_lift_force_n", "max_jaw_speed_m_s", "max_grip_generalized_force_n"):
        spec[key] = float(settings.get(key, spec[key]))
        if not math.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    spec["max_motor_torque_nm"] *= float(settings.get("motor_strength_scale", 1))
    if not math.isfinite(spec["max_motor_torque_nm"]) or spec["max_motor_torque_nm"] <= 0:
        raise ValueError("motor_strength_scale must be finite and positive")
    if spec["max_wheel_speed_rad_s"] > spec["motor_no_load_speed_rad_s"]:
        raise ValueError("Wheel command speed exceeds motor no-load speed")
    world, actuators, sensors, equality = [_section(root, tag) for tag in
                                         ("worldbody", "actuator", "sensor", "equality")]
    base = SubElement(world, "body", name="competition_robot", pos="1.020 .825 .020")
    SubElement(base, "freejoint", name="competition_robot_free")
    colliders = []

    def geom(parent, short_name, material="plastic", **attributes):
        name = f"robot_comp_{short_name}"
        rubber = material == "rubber"
        values = {"name": name, "contype": "4", "conaffinity": "3", "condim": "3",
                  "friction": ".7 .00005 .000005" if rubber else ".3 .0001 .00001",
                  "solref": ".003 1" if rubber else ".001 1",
                  "solimp": ".90 .99 .0003" if rubber else ".98 .999 .0001"}
        values.update({key: str(value) for key, value in attributes.items()})
        element = SubElement(parent, "geom", **values)
        colliders.append({"name": name, "material": material, "body": parent.get("name"),
                          "robot_part": True, "grip_pad": short_name.startswith("pad_")})
        return element

    geom(base, "deck", type="box", size=".075 .060 .013", pos="0 -.015 .001",
         mass=".400", rgba=".12 .19 .25 1")
    geom(base, "battery", type="box", size=".031 .023 .010", pos="0 -.047 .024",
         mass=".165", rgba=".08 .10 .12 1")
    geom(base, "controller", type="box", size=".024 .020 .003", pos="0 -.030 .037",
         mass=".025", rgba=".12 .42 .32 1")
    geom(base, "mast", type="box", size=".032 .008 .055", pos="0 .047 .070",
         mass=".070", rgba=".62 .67 .69 1")
    # Visual envelope guards sit above the tires; internal collisions are disabled
    # by the robot's dedicated bit, while all arena contacts remain enabled.
    for sign in (-1, 1):
        geom(base, f"guard_{sign}", type="box", size=".002 .028 .003",
             pos=_numbers([sign * .088, .005, .025]), mass="0", rgba=".12 .19 .25 1")
    motor_names, wheel_joints = [], []
    for side, x in (("left", -.085), ("right", .085)):
        joint = f"comp_wheel_{side}"
        motor = f"comp_motor_{side}"
        wheel = SubElement(base, "body", name=f"comp_wheel_{side}_body", pos=_numbers([x, .005, 0]))
        SubElement(wheel, "joint", name=joint, type="hinge", axis="1 0 0",
                   damping=".00005", frictionloss=".00015", armature=".000001")
        geom(wheel, f"wheel_{side}", "rubber", type="cylinder", size=".020 .0035",
             quat=".707106781187 0 .707106781187 0", mass=".012", rgba=".06 .07 .08 1",
             condim="6", friction=".85 .0001 .00001")
        torque = spec["max_motor_torque_nm"]
        SubElement(actuators, "motor", name=motor, joint=joint, gear="1",
                   ctrllimited="true", ctrlrange=_numbers([-torque, torque]),
                   forcelimited="true", forcerange=_numbers([-torque, torque]))
        SubElement(sensors, "jointpos", name=f"comp_encoder_{side}", joint=joint)
        SubElement(sensors, "jointvel", name=f"comp_speed_{side}", joint=joint)
        motor_names.append(motor)
        wheel_joints.append(joint)
    caster = SubElement(base, "body", name="comp_caster_body", pos="0 -.055 -.014")
    SubElement(caster, "joint", name="comp_caster_ball", type="ball", damping=".000001")
    geom(caster, "caster", type="sphere", size=".006", mass=".012", rgba=".65 .67 .70 1",
         condim="6", friction=".2 .00001 .000001")

    carriage = SubElement(base, "body", name="comp_lift_body", pos="0 .105 0")
    SubElement(carriage, "joint", name="comp_lift", type="slide", axis="0 0 1",
               limited="true", range="0 .080", damping=".8", frictionloss=".08", armature=".002",
               solreflimit=".001 1", solimplimit=".99 .999 .0001")
    geom(carriage, "crossbar", type="box", size=".044 .010 .010", pos="0 -.031 .012",
         mass=".040", rgba=".86 .56 .17 1")
    geom(carriage, "lift_servo", type="box", size=".010 .013 .017", pos="0 -.031 .039",
         mass=".020", rgba=".18 .22 .25 1")
    geom(carriage, "guide_carriage", type="box", size=".020 .014 .008", pos="0 -.054 .023",
         mass="0", rgba=".62 .67 .69 1")
    SubElement(carriage, "site", name="gripper_center", pos="0 0 -.0159", size=".0015",
               rgba=".95 .25 .15 1")
    SubElement(carriage, "site", name="gripper_upper_center", pos="0 0 -.009", size=".0015",
               rgba=".95 .65 .15 1")
    for side, sign in (("left", -1), ("right", 1)):
        finger = SubElement(carriage, "body", name=f"comp_jaw_{side}_body", pos=_numbers([sign*.012, 0, 0]))
        SubElement(finger, "joint", name=f"comp_jaw_{side}", type="slide", axis=_numbers([sign, 0, 0]),
                   limited="true", range="0 .025", damping=".15", armature=".0001")
        SubElement(finger, "inertial", pos="0 0 -.004", mass=".022",
                   diaginertia=".000003 .000002 .000003")
        geom(finger, f"finger_{side}", type="box", size=".002 .018 .012",
             pos=_numbers([sign*.002, 0, -.002]), mass="0", rgba=".86 .56 .17 1")
        geom(finger, f"slider_link_{side}", type="box", size=".003 .017 .003",
             pos=_numbers([sign*.002, -.020, .009]), mass="0", rgba=".86 .56 .17 1")
        # Low side pads grip the upper 1.4 mm of the 5 mm sample. Their underside
        # stays above the 3 mm laboratory plate even when the sample is seated.
        geom(finger, f"pad_{side}_sample", "rubber", type="box", size=".0005 .016 .0007",
             pos=_numbers([-sign*.0025, 0, -.0159]), mass="0", rgba=".18 .26 .29 1")
        geom(finger, f"sample_pad_support_{side}", type="box", size=".0015 .015 .0016",
             pos=_numbers([-sign*.001, 0, -.0136]), mass="0", rgba=".62 .67 .69 1")
        # Broad parallel upper faces avoid pushing a kit forward as the jaws
        # close. The separate low band still handles the thin sample.
        geom(finger, f"upper_pad_support_{side}", type="box", size=".002 .016 .005",
             pos=_numbers([-sign*.0005, 0, -.009]), mass="0", rgba=".86 .56 .17 1")
        geom(finger, f"pad_{side}_upper", "rubber", type="box", size=".0007 .016 .005",
             pos=_numbers([-sign*.0023, 0, -.009]), mass="0", rgba=".18 .26 .29 1")
    SubElement(equality, "joint", name="comp_jaw_coupling", joint1="comp_jaw_left",
               joint2="comp_jaw_right", polycoef="0 1 0 0 0", solref=".001 1",
               solimp=".99 .999 .00001")
    lift_force = spec["max_lift_force_n"]
    grip_force = spec["max_grip_generalized_force_n"]
    SubElement(actuators, "position", name="comp_lift_motor", joint="comp_lift", kp="800", kv="8",
               ctrllimited="true", ctrlrange="0 .080", forcelimited="true",
               forcerange=_numbers([-lift_force, lift_force]))
    SubElement(actuators, "position", name="comp_grip_motor", joint="comp_jaw_left", kp="2000", kv="6",
               ctrllimited="true", ctrlrange="0 .025", forcelimited="true",
               forcerange=_numbers([-grip_force, grip_force]))
    for joint in ("comp_lift", "comp_jaw_left", "comp_jaw_right"):
        SubElement(sensors, "jointpos", name=f"{joint}_position", joint=joint)
    SubElement(base, "site", name="comp_imu", pos="0 0 0", size=".002", rgba="0 0 0 0")
    SubElement(sensors, "framepos", name="comp_position", objtype="body", objname="competition_robot")
    SubElement(sensors, "framequat", name="comp_orientation", objtype="body", objname="competition_robot")
    SubElement(sensors, "gyro", name="comp_gyro", site="comp_imu")
    spec.update({
        "freejoint": "competition_robot_free", "wheel_joints": wheel_joints,
        "motor_names": motor_names, "lift_joint": "comp_lift", "lift_motor": "comp_lift_motor",
        "jaw_joints": ["comp_jaw_left", "comp_jaw_right"], "grip_motor": "comp_grip_motor",
        "jaw_coupling": "comp_jaw_coupling", "gripper_site": "gripper_center",
        "upper_gripper_site": "gripper_upper_center", "forward_axis": [0, 1, 0],
        "initial_position_m": [1.020, .825, .020], "colliders": colliders,
        "chassis_size_m": [.150, .120, .026],
        "description": "Four-motor preliminary robot with symmetric stepped contact jaws.",
    })
    return spec
