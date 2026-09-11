"""Arena and moving magnetic ports for the hinge-driven swarm."""
from __future__ import annotations

import copy
from dataclasses import asdict
import math
import xml.etree.ElementTree as ET

import numpy as np

from .builder import build_arena, numbers
from .swarm_hinge_robot import add_hinge_robot
from .swarm_magnets import PARAMETERS


HINGE_ACTION_NAMES = ["front_yaw", "front_pitch", "rear_pitch", "magnet_enable"]
HINGE_PORT_FRAMES = [(side, sign*.012, .008, .001)
                     for side in ("front", "rear") for sign in (-1, 1)]


def add_hinge_magnetic_docks(root, specs):
    """Place two lateral EPM faces on each moving lobe.

    Connector forces act on these child bodies. Housings are included in each
    lobe's 14 g mass budget and do not add actuated floor contacts.
    """
    for spec in specs:
        if spec.get("magnetic_sites"):
            raise ValueError(f"Magnetic docks already present: {spec['name']}")
        spec["magnetic_sites"] = []
        frames = []
        for port, (side, x, y, z) in enumerate(HINGE_PORT_FRAMES):
            body_name = f"{spec['name']}_{side}"
            body = root.find(f".//body[@name='{body_name}']")
            if body is None:
                raise ValueError(f"Moving lobe missing: {body_name}")
            sign = -1 if x < 0 else 1
            name = f"{spec['name']}_magnet_{port}"
            housing = name + "_housing"
            ET.SubElement(body, "geom", name=housing, type="box",
                          pos=numbers([sign*.0095, y, z]), size=".0025 .0015 .0015",
                          mass="0", contype="4", conaffinity="7", condim="3",
                          friction=".4 .00001 .000001", solref=".004 1",
                          solimp=".9 .99 .0001", priority="2", rgba=".22 .31 .32 1")
            ET.SubElement(body, "site", name=name, type="box", size=".0014 .0014 .0001",
                          pos=numbers([x, y, z]),
                          quat=numbers([math.sqrt(.5), 0, sign*math.sqrt(.5), 0]),
                          rgba=".23 .65 .65 1")
            spec["magnetic_sites"].append(name)
            spec["colliders"].append(dict(name=housing, body=body_name,
                                          material="magnet_housing", robot_part=True,
                                          grip_pad=False))
            spec["collider_names"].append(housing)
            frames.append(dict(body=body_name, position_m=[x, y, z], normal=[sign, 0, 0]))
        spec["magnetic_docking"] = {
            **asdict(PARAMETERS), "ports": 4, "releasable": True,
            "port_frames": frames, "max_partners_per_port": 1,
            "switching": "coordinated EPM handshake",
            "assumption": "Unbuilt shoulder EPMs included in 40 g assembly mass",
        }
    return specs


def hinge_packing(count=40):
    """Stagger 24 mm-wide modules inside the competition start zone."""
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 40:
        raise ValueError("count must be an integer from 1 to 40")
    columns = 4 if count > 8 else (2 if count > 2 else 1)
    row, column = np.divmod(np.arange(count), columns)
    return np.column_stack((.900+.056*column+.028*(row % 2),
                            .770+.0248*row, np.full(count, .0052),
                            np.full(count, math.pi/2)))


def hinge_packing_phases(poses):
    """Neighboring front/rear ports receive the same bending phase.

    One quarter-cycle separates lobes. Each 28 mm offset along the initial
    forward axis contributes the same quarter-cycle to a module's oscillator.
    These are initial oscillator phases, not moving root-position commands.
    """
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 4 or not np.isfinite(poses).all():
        raise ValueError("poses must contain finite XYZYaw rows")
    heading = poses[0, 3]
    forward = np.array([-math.sin(heading), math.cos(heading)])
    longitudinal = (poses[:, :2]-poses[0, :2]) @ forward
    return longitudinal/.028*(-math.pi/2)


def build_hinge_swarm_scene(num_robots=40, num_objects=0, timestep=.001,
                            solver="CG", friction=.7):
    """Use the competition field and exact payload geometries with hinge robots."""
    if not isinstance(num_objects, int) or isinstance(num_objects, bool) or not 0 <= num_objects <= 2:
        raise ValueError("The hinge scene supports zero, one, or two payloads")
    if timestep not in (.0005, .001, .002):
        raise ValueError("Hinge timestep must be 0.5, 1, or 2 ms")
    if solver not in ("Newton", "CG"):
        raise ValueError("Use Newton or CG contact dynamics")
    poses = hinge_packing(num_robots)
    xml, metadata = build_arena(robot=False, tape_mode="rigid", profile={"timestep_s": timestep})
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    original = {obj["name"]: world.find(f"body[@name='{obj['name']}']")
                for obj in metadata["objects"]}
    categories = ["Cylinder_Red", "Medical_Kit"][:num_objects]
    selected = [next(name for name in original if name.startswith(category))
                for category in categories]
    for body in original.values():
        world.remove(body)
    for geom in list(world.findall("geom")):
        if (geom.get("name") or "").endswith("_mark"):
            world.remove(geom)
    objects = []
    for index, source in enumerate(selected):
        body = copy.deepcopy(original[source])
        name = f"hinge_payload_{index:02d}"
        for element in body.iter():
            if element.get("name"):
                element.set("name", name+"_"+element.get("name"))
        body.set("name", name)
        body.find("freejoint").set("name", name+"_free")
        shape = next(geom for geom in body.findall("geom") if geom.get("contype", "1") != "0")
        shape.set("name", name+"_geom")
        shape.set("priority", "0")
        size = np.fromstring(shape.get("size"), sep=" ")
        is_box = shape.get("type") == "box"
        radius, half_height = (size[1], size[2]) if is_box else (size[0], size[1])
        xy = [.78, .885+(.10 if index else -.10)] if num_objects == 2 else [.78, .885]
        position = [*xy, float(half_height+.0001)]
        body.set("pos", numbers(position))
        world.append(body)
        objects.append(dict(name=name, freejoint=name+"_free", geom=name+"_geom",
                            source_name=source, kind="kit" if is_box else "cylinder",
                            radius=float(radius), half_height=float(half_height),
                            initial_position_m=position))
    specs = [add_hinge_robot(root, index, pose[:3], pose[3], friction=friction)
             for index, pose in enumerate(poses)]
    add_hinge_magnetic_docks(root, specs)
    names = {geom.get("name") for geom in root.iter("geom")}
    contacts = root.find("contact")
    if contacts is not None:
        for pair in list(contacts):
            if pair.tag == "pair" and (pair.get("geom1") not in names or pair.get("geom2") not in names):
                contacts.remove(pair)
    option = root.find("option")
    for key, value in dict(timestep=timestep, solver=solver, iterations=200,
                           ls_iterations=50, tolerance=1e-12, cone="elliptic", impratio=1).items():
        option.set(key, str(value))
    ET.SubElement(world, "camera", name="hinge_side", pos="1.6 -.3 .5", xyaxes=".78 .62 0 -.20 .25 .95")
    metadata.update(task="hinge_magnetic_swarm_body", policy_contract="hinge_body_v1",
                    policy_action_names=HINGE_ACTION_NAMES, robots=specs, objects=objects,
                    rule_score=False, robot_start_packing_m=poses[:, :3].tolist(),
                    robot_start_yaw_rad=poses[:, 3].tolist(),
                    robot_gait_phase_rad=hinge_packing_phases(poses).tolist(),
                    magnetic_force_targets="module_owned_lobe_bodies",
                    start_mode="all_modules_inside_start_zone")
    return ET.tostring(root, encoding="unicode"), metadata
