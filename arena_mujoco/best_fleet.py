"""Five compact contact-gripper robots in the senior preliminary arena.

The finite arena, wooden objects and laboratory retain the audited geometry.
Only motor torques and force-limited jaw/lift actuators move the robots.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .builder import build_arena, numbers
from .competition_robot import add_competition_robot
from .competition_events import initialize_competition_events, update_competition_events
from .competition_scoring import score_competition
from .materials import load_profile

ROOT = Path(__file__).resolve().parents[1]


def design():
    return json.loads((ROOT / "best_design.json").read_text())


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _rename(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_rename(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _rename(v, mapping) for k, v in value.items()}
    return value


def add_compact_robot(root, entry, shared):
    """Reuse the tested stepped jaw geometry with a smaller drive platform."""
    temporary = ET.Element("mujoco")
    meta = add_competition_robot(temporary)
    base = temporary.find("worldbody/body")
    base.set("pos", numbers([*entry["start_xy_m"], shared["wheel_radius_m"]]))
    yaw = entry.get("start_yaw_rad", 0.0)
    base.set("quat", numbers([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))

    def item(name):
        return next(e for e in temporary.iter() if e.get("name") == name)

    item("robot_comp_deck").set("size", ".0525 .045 .002")
    item("robot_comp_deck").set("pos", "0 -.020 .002")
    item("robot_comp_deck").set("rgba", numbers(entry["rgba"]))
    item("robot_comp_battery").set("size", ".027 .016 .0075")
    item("robot_comp_battery").set("pos", "0 -.040 .0125")
    item("robot_comp_controller").set("size", ".0325 .015 .006")
    item("robot_comp_controller").set("pos", "0 -.030 .026")
    item("robot_comp_mast").set("size", ".032 .006 .033")
    item("robot_comp_mast").set("pos", "0 .016 .036")
    for side, sign in (("left", -1), ("right", 1)):
        item(f"comp_wheel_{side}_body").set("pos", numbers([sign * .055, 0, 0]))
        item(f"robot_comp_wheel_{side}").set("size", ".025 .004")
        item(f"robot_comp_guard_{sign}").set("pos", numbers([sign * .0605, 0, .030]))
    item("comp_caster_body").set("pos", "0 -.055 -.019")
    item("comp_lift_body").set("pos", "0 .065 -.005")
    item("comp_lift").set("range", "0 .045")
    item("comp_lift_motor").set("ctrlrange", "0 .045")
    # The inherited servo depiction would exceed the new 110 mm height budget.
    item("robot_comp_lift_servo").set("pos", "0 -.031 .027")
    item("robot_comp_lift_servo").set("size", ".010 .013 .012")
    if entry["id"] == "kit":
        base.remove(item("comp_lift_body"))
        base.remove(item("robot_comp_mast"))
        temporary.remove(temporary.find("equality"))
        for element in list(temporary.find("actuator")):
            if element.get("name") not in meta["motor_names"]:
                temporary.find("actuator").remove(element)
        for element in list(temporary.find("sensor")):
            if element.get("joint") not in meta["wheel_joints"]:
                temporary.find("sensor").remove(element)
        gates = []
        for index, (x, width) in enumerate(((-.0295, .056), (.0135, .030), (.0445, .030))):
            for side, sign in (("left", -1), ("right", 1)):
                ET.SubElement(base, "geom", name=f"kit_chute_{index}_{side}", type="box",
                              pos=numbers([x + sign * (width / 2 - .00075), .051, .030]),
                              size=".00075 .0175 .013", mass=".003", contype="4",
                              rgba=".58 .66 .72 1", friction=".25 .0001 .00001")
            for end, y in (("back", .0335), ("front", .0685)):
                ET.SubElement(base, "geom", name=f"kit_chute_{index}_{end}", type="box",
                              pos=numbers([x, y, .030]), size=numbers([width / 2, .0015, .013]),
                              mass=".003", contype="4", rgba=".58 .66 .72 1")
            body = ET.SubElement(base, "body", name=f"kit_gate_{index}_body",
                                 pos=numbers([x, .035, .017]))
            joint = f"kit_gate_{index}"
            ET.SubElement(body, "joint", name=joint, type="hinge", axis="-1 0 0",
                          range="0 1.57079632679", damping=".002", armature=".000002")
            ET.SubElement(body, "geom", name=f"kit_gate_{index}_floor", type="box",
                          pos="0 .016 -.0005", size=numbers([width / 2 - .0015, .016, .0005]),
                          mass=".006", contype="4", rgba=".84 .60 .20 1")
            actuator = f"kit_gate_{index}_motor"
            ET.SubElement(temporary.find("actuator"), "position", name=actuator, joint=joint,
                          kp=".6", kv=".02", ctrlrange="0 1.57079632679", forcerange="-.06 .06")
            gates.append(actuator)
        meta.update(gate_motors=gates, gate_joints=[f"kit_gate_{i}" for i in range(3)],
                    lift_motor=None, grip_motor=None, jaw_joints=[], physical_motor_count=5,
                    preload_local_xy_m=[[-.043, .051], [-.016, .051], [.0135, .051], [.0445, .051]])
    if entry["id"] == "lab":
        carriage = item("comp_lift_body")
        base.remove(carriage)
        extension = ET.SubElement(base, "body", name="comp_extension_body")
        ET.SubElement(extension, "joint", name="comp_extension", type="slide", axis="0 1 0",
                      range="0 .040", damping="1", armature=".002")
        ET.SubElement(extension, "geom", name="robot_comp_extension_rail", type="box",
                      size=".019 .026 .002", pos="0 .025 .028", mass=".022",
                      rgba=".65 .68 .7 1", contype="4", conaffinity="7")
        extension.append(carriage)
        ET.SubElement(temporary.find("actuator"), "position", name="comp_extension_motor",
                      joint="comp_extension", kp="1000", kv="10", ctrlrange="0 .04",
                      forcerange="-4 4")
        meta.update(extension_joint="comp_extension", extension_motor="comp_extension_motor",
                    extension_range_m=[0., .04], physical_motor_count=5)
    mass_elements = [e for e in temporary.iter() if float(e.get("mass", 0)) > 0]
    scale = shared["mass_budget_kg"] / sum(float(e.get("mass")) for e in mass_elements)
    for element in mass_elements:
        element.set("mass", str(float(element.get("mass")) * scale))
        if element.get("diaginertia"):
            element.set("diaginertia", numbers(np.fromstring(element.get("diaginertia"), sep=" ") * scale))
    for geom in temporary.iter("geom"):
        geom.set("conaffinity", "7")
        geom.set("priority", "2")
    # Robot-to-robot contact stays enabled; only components of the same assembly
    # are excluded. The shared jaw equality remains a joint coupling, not a grasp.
    bodies = [b.get("name") for b in base.iter("body")]
    contact = ET.SubElement(temporary, "contact")
    for a, b in itertools.combinations(bodies, 2):
        ET.SubElement(contact, "exclude", body1=a, body2=b)
    prefix = f"fleet_{entry['id']}_"
    mapping = {e.get("name"): prefix + e.get("name") for e in temporary.iter() if e.get("name")}
    for element in temporary.iter():
        for key, value in list(element.attrib.items()):
            if value in mapping:
                element.set(key, mapping[value])
    for section in temporary:
        target = root.find(section.tag)
        if target is None:
            target = ET.SubElement(root, section.tag)
        target.extend(section)
    meta = _rename(meta, mapping)
    meta.update(id=entry["id"], footprint_m=shared["footprint_m"],
                footprint_bounds_local_m=shared["footprint_bounds_local_m"],
                mass_kg=shared["mass_budget_kg"], wheel_radius_m=.025,
                wheel_track_m=.110, wheel_axle_y_m=0., lift_range_m=[0., .045],
                initial_position_m=[*entry["start_xy_m"], .025],
                grasp_center_body_m=[0., .065, -.0209])
    return meta


def build_fleet(*, seed=0, randomize=False, only=None, timestep=.001):
    spec = design()
    profile = load_profile({"timestep_s": timestep}, seed=seed, randomize=randomize)
    xml, metadata = build_arena(robot=False, tape_mode="rigid", profile=profile)
    root = ET.fromstring(xml)
    option = root.find("option")
    option.set("solver", "Newton")
    option.set("cone", "elliptic")
    option.set("iterations", "50")
    option.set("ls_iterations", "50")
    option.set("impratio", "10")
    selected = [r for r in spec["robots"] if only is None or r["id"] in only]
    metadata["robots"] = [add_compact_robot(root, r, spec["shared_robot"]) for r in selected]
    kit_robot = next((r for r in metadata["robots"] if r["id"] == "kit"), None)
    if kit_robot:
        for index, xy in enumerate(kit_robot["preload_local_xy_m"], 1):
            body = root.find(f".//body[@name='Medical_Kit_{index:02d}']")
            body.set("pos", numbers([kit_robot["initial_position_m"][0] + xy[0],
                                      kit_robot["initial_position_m"][1] + xy[1], .052]))
    metadata["design"] = spec["name"]
    metadata["randomize"] = randomize
    metadata["seed"] = seed
    metadata["physics_tier"] = "native rigid tape; deformable adhesive tape is excluded"
    return ET.tostring(root, encoding="unicode"), metadata


class FleetSimulation:
    """One physical world and one synchronized motor-control clock."""

    control_dt = .02

    def __init__(self, *, seed=0, randomize=False, only=None, timestep=.001):
        self.xml, self.metadata = build_fleet(seed=seed, randomize=randomize, only=only, timestep=timestep)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.robots = {r["id"]: r for r in self.metadata["robots"]}
        rng = np.random.default_rng(seed + 711)
        self.motor_strength = {key: float(rng.uniform(.8, 1.)) if randomize else 1.
                               for key in self.robots}
        self.wheel_friction_scale = float(rng.uniform(.75, 1.15)) if randomize else 1.
        for gid in range(self.model.ngeom):
            name = self.model.geom(gid).name or ""
            if "robot_comp_wheel" in name:
                self.model.geom_friction[gid, 0] *= self.wheel_friction_scale
        self.metadata["drive_domain"] = {"motor_strength": self.motor_strength,
                                         "wheel_friction_scale": self.wheel_friction_scale}
        self.targets = {key: np.array([0., 0., .0, .025]) for key in self.robots}
        self.integrals = {key: np.zeros(2) for key in self.robots}
        self.extension_targets = {key: 0. for key in self.robots}
        self.gate_targets = {key: np.zeros(len(robot.get("gate_motors", [])))
                             for key, robot in self.robots.items()}
        self.max_torque = 0.
        self.max_tilt = 0.
        self.trace = []
        for robot in self.robots.values():
            for joint in robot["jaw_joints"]:
                self.data.qpos[self.model.joint(joint).qposadr[0]] = .025
        mujoco.mj_forward(self.model, self.data)
        self.setup = initialize_competition_events(self.model, self.data, self.metadata)

    def pose(self, key):
        body = self.data.body(self.robots[key]["name"])
        rotation = body.xmat.reshape(3, 3)
        return body.xpos[:2].copy(), math.atan2(rotation[1, 1], rotation[0, 1])

    def joint(self, key, field):
        name = self.robots[key][field]
        if isinstance(name, list):
            name = name[0]
        return float(self.data.qpos[self.model.joint(name).qposadr[0]])

    def tool(self, key):
        return self.data.site(self.robots[key]["gripper_site"]).xpos.copy()

    def command(self, key, forward=0., yaw=0., lift=.035, jaw=.025):
        """Body velocity goals in SI units; native actuators enforce force caps."""
        values = np.asarray([forward, yaw, lift, jaw], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite fleet command")
        self.targets[key] = np.clip(values, [-.35, -2.5, 0., 0.], [.35, 2.5, .045, .025])

    def step(self, *, record=False):
        for _ in range(round(self.control_dt / self.model.opt.timestep)):
            for key, robot in self.robots.items():
                v, yaw, lift, jaw = self.targets[key]
                desired = -np.array([v - yaw * .055, v + yaw * .055]) / .025
                desired = np.clip(desired, -20, 20)
                for i, (joint, actuator) in enumerate(zip(robot["wheel_joints"], robot["motor_names"])):
                    speed = self.data.qvel[self.model.joint(joint).dofadr[0]]
                    error = desired[i] - speed
                    self.integrals[key][i] = np.clip(self.integrals[key][i] + .02 * error * self.model.opt.timestep, -.008, .008)
                    torque = .008 * error + self.integrals[key][i]
                    limit = .025 * self.motor_strength[key]
                    if torque * speed > 0:
                        limit *= max(0, 1 - abs(speed) / robot["motor_no_load_speed_rad_s"])
                    torque = np.clip(torque, -limit, limit)
                    self.data.ctrl[self.model.actuator(actuator).id] = torque
                    self.max_torque = max(self.max_torque, abs(float(torque)))
                for field, target, rate, force_error in (("lift_motor", lift, .08, .005), ("grip_motor", jaw, .05, .001)):
                    if not robot.get(field):
                        continue
                    actuator = self.model.actuator(robot[field]).id
                    joint_field = "lift_joint" if field == "lift_motor" else "jaw_joints"
                    position = self.joint(key, joint_field)
                    old = self.data.ctrl[actuator]
                    moved = old + np.clip(target - old, -rate * self.model.opt.timestep, rate * self.model.opt.timestep)
                    self.data.ctrl[actuator] = np.clip(moved, position - force_error, position + force_error)
                if "extension_motor" in robot:
                    actuator = self.model.actuator(robot["extension_motor"]).id
                    position = self.joint(key, "extension_joint")
                    old = self.data.ctrl[actuator]
                    moved = old + np.clip(self.extension_targets[key] - old,
                                          -.04 * self.model.opt.timestep, .04 * self.model.opt.timestep)
                    self.data.ctrl[actuator] = np.clip(moved, position - .004, position + .004)
                for index, name in enumerate(robot.get("gate_motors", [])):
                    actuator = self.model.actuator(name).id
                    old = self.data.ctrl[actuator]
                    self.data.ctrl[actuator] = old + np.clip(self.gate_targets[key][index] - old,
                                                           -2.5 * self.model.opt.timestep,
                                                           2.5 * self.model.opt.timestep)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning):
            raise FloatingPointError("Fleet simulation emitted a numerical warning")
        for key, robot in self.robots.items():
            tilt = math.acos(np.clip(self.data.body(robot["name"]).xmat[8], -1, 1))
            self.max_tilt = max(self.max_tilt, tilt)
        update_competition_events(self.model, self.data, self.metadata)
        if record:
            self.trace.append((float(self.data.time), self.data.qpos.copy(), self.data.qvel.copy(),
                               np.array(list(self.targets.values()))))

    def score(self):
        return score_competition(self.model, self.data, self.metadata, time_limit_s=120)

    def save(self, directory, report=None):
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "scene.xml").write_text(self.xml)
        (out / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        if self.trace:
            np.savez_compressed(out / "trajectory.npz", time=[t[0] for t in self.trace],
                                qpos=[t[1] for t in self.trace], qvel=[t[2] for t in self.trace],
                                command=[t[3] for t in self.trace])
        if report is not None:
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
