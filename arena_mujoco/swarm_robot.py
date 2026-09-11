"""Photo-inspired miniature modules with wheel drive and contact-only lift pads.

Local +Y is forward. Every module has a free root; no object attachment is
created. Hardware packaging and friction coefficients are design assumptions.
"""
from __future__ import annotations

import copy
import math
from xml.etree.ElementTree import Element, SubElement

DESIGN = {
    "name": "swarm_module", "schema_version": 1,
    "footprint_m": [.024, .055],
    "footprint_bounds_local_m": [[-.012, -.027], [.012, .028]],
    "height_m": .023, "mass_kg": .040,
    "wheel_radius_m": .007, "wheel_track_m": .021,
    "wheel_direction_sign": -1., "forward_axis": [0, 1, 0],
    "max_motor_torque_nm": .002,
    "motor_no_load_speed_rad_s": 250 * 2 * math.pi / 60,
    "max_wheel_speed_rad_s": 20., "velocity_gain": .00015,
    "lift_range_m": [0., .008], "max_lift_force_n": .5,
    "max_lift_speed_m_s": .010,
    "pad_center_body_m": [0., .027, -.0029],
    "pad_front_body_m": [0., .028, -.0029],
    "pad_floor_heights_m": [.0034, .0048],
    "upper_pad_floor_heights_m": [.005, .013],
    "physical_motor_count": 3,
    "hardware": {
        "drive": "2 x Pololu 2358, 136:1, regulated 3 V, custom bevel transfer",
        "lift": "1 x Pololu 2359, 700:1, regulated 3 V, 3 mm radius pinion and rack",
        "wheels": "Custom 14 x 3 mm silicone tires on independent half axles",
        "electronics": "Custom motor-driver/radio PCB, 1S LiPo, wheel encoders and lift position sensor",
    },
    "sources": ["https://www.pololu.com/product/2358",
                "https://www.pololu.com/product/2359",
                "https://www.pololu.com/file/0J1831/sub-micro-plastic-planetary-gearmotor-dimension-diagram.pdf"],
    "assumptions": ["40 g complete mass budget; no prototype measured",
                    "Custom longitudinal motor packaging and bevel shafts require detailed CAD",
                    "Tire friction 0.85 and pad friction 0.7 require measurement",
                    "The photograph establishes shell appearance, not internal actuation"],
}


def _numbers(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def _section(root, tag):
    element = root.find(tag)
    return element if element is not None else SubElement(root, tag)


def add_swarm_robot(root: Element, index: int, position=(0., 0., .0072), yaw=0.) -> dict:
    """Append one module, returning actuator/joint/site names and physical limits.

    Wheels accept torque in N m. Runtime converts desired speed through a
    saturated motor model. Lift accepts a position in meters; runtime must slew
    that target at max_lift_speed_m_s. Names use a stable zero-based index.
    """
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("index must be a nonnegative integer")
    position = tuple(float(v) for v in position)
    if len(position) != 3 or not all(math.isfinite(v) for v in (*position, yaw)):
        raise ValueError("position must have three finite coordinates and yaw must be finite")
    prefix = f"swarm_{index:02d}"
    if root.find(f".//body[@name='{prefix}']") is not None:
        raise ValueError(f"Duplicate module index: {index}")
    world, actuators, sensors, asset = [_section(root, tag) for tag in
                                       ("worldbody", "actuator", "sensor", "asset")]
    spec = copy.deepcopy(DESIGN)
    base = SubElement(world, "body", name=prefix, pos=_numbers(position),
                      quat=_numbers([math.cos(yaw/2), 0, 0, math.sin(yaw/2)]))
    SubElement(base, "freejoint", name=f"{prefix}_free")
    # Complete base assembly mass includes motors, battery, PCB, and ballast.
    SubElement(base, "inertial", pos="0 -.002 .002", mass=".035",
               diaginertia=".0000085 .0000025 .000009")
    colliders = []

    def geom(parent, short, material="plastic", **attributes):
        name = f"{prefix}_{short}"
        values = dict(name=name, contype="4", conaffinity="7", condim="3",
                      friction=".7 .00001 .000001" if material == "rubber" else ".3 .00001 .000001",
                      solref=".004 1", solimp=".9 .99 .0001", mass="0", priority="2")
        values.update({k: str(v) for k, v in attributes.items()})
        SubElement(parent, "geom", **values)
        colliders.append({"name": name, "body": parent.get("name"), "material": material,
                          "grip_pad": short in {"pad", "pad_upper"}, "robot_part": True})
        return name

    # Shared convex shell asset preserves the reference's broad shoulders and
    # tapered ends. Eight vertices at each end produce chamfered side faces.
    if asset.find("mesh[@name='swarm_lobe']") is None:
        vertices = []
        for y, half_width, zlow, zhigh in ((.007, .010, -.004, .010),
                                          (.027, .0045, -.004, .007)):
            bevel = .0015
            for x, z in ((-half_width+bevel, zlow), (half_width-bevel, zlow),
                         (half_width, zlow+bevel), (half_width, zhigh-bevel),
                         (half_width-bevel, zhigh), (-half_width+bevel, zhigh),
                         (-half_width, zhigh-bevel), (-half_width, zlow+bevel)):
                vertices.extend((x, y, z))
        SubElement(asset, "mesh", name="swarm_lobe", vertex=_numbers(vertices))
    for side, quat in (("front", "1 0 0 0"), ("rear", "0 0 0 1")):
        geom(base, f"shell_{side}", type="mesh", mesh="swarm_lobe", quat=quat,
             rgba=".43 .47 .45 1")
    geom(base, "ball_housing", type="sphere", size=".008", pos="0 0 .003",
         rgba=".055 .06 .065 1")
    # Passive rounded skids represent low-friction wear pads. They keep the
    # long lobes level without additional control or nonphysical planar joints.
    for side, y in (("front", .020), ("rear", -.022)):
        geom(base, f"skid_{side}", type="sphere", size=".0015",
             pos=_numbers([0, y, -.0055]), friction=".04 .000001 .0000001",
             rgba=".22 .23 .24 1")
    wheel_joints, motor_names = [], []
    for side, x in (("left", -.0105), ("right", .0105)):
        joint, motor = f"{prefix}_wheel_{side}", f"{prefix}_motor_{side}"
        wheel = SubElement(base, "body", name=f"{joint}_body", pos=_numbers([x, 0, 0]))
        SubElement(wheel, "joint", name=joint, type="hinge", axis="1 0 0",
                   damping=".000001", frictionloss=".000002", armature=".00000001")
        geom(wheel, f"tire_{side}", "rubber", type="cylinder", size=".007 .0015",
             quat=".707106781187 0 .707106781187 0", mass=".001",
             friction=".85 .00001 .000001", rgba=".065 .07 .075 1")
        SubElement(actuators, "motor", name=motor, joint=joint, gear="1",
                   ctrllimited="true", ctrlrange="-.002 .002",
                   forcelimited="true", forcerange="-.002 .002")
        SubElement(sensors, "jointpos", name=f"{joint}_position", joint=joint)
        SubElement(sensors, "jointvel", name=f"{joint}_speed", joint=joint)
        wheel_joints.append(joint)
        motor_names.append(motor)
    lift_joint, lift_motor = f"{prefix}_lift", f"{prefix}_lift_motor"
    carriage = SubElement(base, "body", name=f"{prefix}_carriage", pos="0 .027 0")
    SubElement(carriage, "joint", name=lift_joint, type="slide", axis="0 0 1",
               limited="true", range="0 .008", damping=".1", armature=".00005",
               solreflimit=".004 1", solimplimit=".95 .99 .0001")
    SubElement(carriage, "inertial", pos="0 0 0", mass=".003",
               diaginertia=".00000005 .00000008 .00000005")
    pad = geom(carriage, "pad", "rubber", type="box", size=".005 .001 .0007",
               pos="0 0 -.0029", rgba=".13 .19 .18 1")
    upper_pad = geom(carriage, "pad_upper", "rubber", type="box", size=".005 .001 .004",
                     pos="0 0 .002", rgba=".13 .19 .18 1")
    SubElement(carriage, "site", name=f"{prefix}_pad_site", pos="0 .001 -.0029",
               size=".001", rgba=".8 .42 .13 1")
    SubElement(actuators, "position", name=lift_motor, joint=lift_joint, kp="250", kv="1",
               ctrllimited="true", ctrlrange="0 .008", forcelimited="true", forcerange="-.5 .5")
    SubElement(sensors, "jointpos", name=f"{lift_joint}_position", joint=lift_joint)
    SubElement(sensors, "touch", name=f"{prefix}_pad_touch", site=f"{prefix}_pad_touch_site")
    SubElement(carriage, "site", name=f"{prefix}_pad_touch_site", type="box", size=".0051 .0011 .0061",
               rgba="0 0 0 0")
    spec.update(index=index, name=prefix, freejoint=f"{prefix}_free",
                initial_position_m=list(position), initial_yaw_rad=float(yaw),
                wheel_joints=wheel_joints, motor_names=motor_names,
                lift_joint=lift_joint, lift_motor=lift_motor,
                gripper_site=f"{prefix}_pad_site", pad_geom=pad, pad_geoms=[pad, upper_pad],
                colliders=colliders)
    return spec


def add_swarm_robots(root: Element, count=40, positions=None) -> list[dict]:
    """Add a fleet at supplied XYZ or XYZYaw poses; default is a display grid.

    Arena reset owns obstacle-aware packing. The default grid is centered at the
    origin for standalone inspection and must not be used as arena placement.
    """
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 40:
        raise ValueError("count must be between 1 and 40")
    if positions is None:
        positions = [((i % 8 - 3.5)*.030, (i//8-2)*.060, .0072) for i in range(count)]
    poses = list(positions)
    if len(poses) != count or any(len(p) not in (3, 4) for p in poses):
        raise ValueError("positions must contain count XYZ or XYZYaw poses")
    return [add_swarm_robot(root, i, p[:3], p[3] if len(p) == 4 else 0.)
            for i, p in enumerate(poses)]
