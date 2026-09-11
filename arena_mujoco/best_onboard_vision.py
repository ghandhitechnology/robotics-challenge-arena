"""Candidate LAB carriage camera for native RGB inspection coupons."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .best_fleet import FleetSimulation
from .best_vision_refiner import PinholeCamera, Plane, RefinerThresholds
from .competition_events import initialize_competition_events


CAMERA_NAME = "fleet_lab_sample_inspection"
CAMERA_LOCAL_POSITION_M = (0.0, 0.0, 0.0841)
CAMERA_FOVY_DEGREES = 58.0
CAMERA_FRAME_NAMES = tuple(
    f"fleet_lab_inspection_camera_frame_{name}"
    for name in ("left", "right", "rear", "front"))
CAMERA_BACKPLATE_NAME = "fleet_lab_inspection_camera_backplate"
ONBOARD_REFINER_THRESHOLDS = RefinerThresholds(
    minimum_cnn_confidence=0.995,
    minimum_arc_coverage=0.30,
    maximum_radial_residual_px=0.65,
    maximum_radius_relative_error=0.06,
    maximum_coarse_offset_px=10.0,
    minimum_edge_points=36,
)


def inject_lab_inspection_camera(xml: str) -> str:
    """Reuse the spec camera, or inject it into an older fleet snapshot."""
    root = ET.fromstring(xml)
    carriage = root.find(".//body[@name='fleet_lab_comp_lift_body']")
    if carriage is None:
        raise ValueError("LAB lift carriage is missing from the fleet model")
    existing_camera = carriage.find(f"camera[@name='{CAMERA_NAME}']")
    existing_parts = [carriage.find(f"geom[@name='{name}']")
                      for name in (*CAMERA_FRAME_NAMES, CAMERA_BACKPLATE_NAME)]
    if existing_camera is not None or any(part is not None for part in existing_parts):
        if existing_camera is None or any(part is None for part in existing_parts):
            raise ValueError("baseline fleet contains an incomplete inspection camera")
        position = np.fromstring(existing_camera.get("pos", ""), sep=" ")
        if (not np.allclose(position, CAMERA_LOCAL_POSITION_M, atol=1e-9) or
                abs(float(existing_camera.get("fovy")) - CAMERA_FOVY_DEGREES) > 1e-9):
            raise ValueError("baseline inspection camera differs from the coupon contract")
        return xml
    ET.SubElement(
        carriage, "camera", name=CAMERA_NAME,
        pos="0 0 .0841", quat="1 0 0 0", fovy=str(CAMERA_FOVY_DEGREES))
    outer = np.array((0.025, 0.024, 0.006))
    aperture = np.array((0.012, 0.012))
    center = np.array((0.0, 0.0, 0.080))
    side_width = (outer[0] - aperture[0]) / 2
    end_width = (outer[1] - aperture[1]) / 2
    pieces = []
    for sign, name in ((-1, "left"), (1, "right")):
        size = np.array((side_width, outer[1], outer[2]))
        position = center + (sign * (aperture[0] / 2 + side_width / 2), 0, 0)
        pieces.append((name, position, size))
    for sign, name in ((-1, "rear"), (1, "front")):
        size = np.array((aperture[0], end_width, outer[2]))
        position = center + (0, sign * (aperture[1] / 2 + end_width / 2), 0)
        pieces.append((name, position, size))
    volume = sum(float(np.prod(size)) for _, _, size in pieces)
    for name, position, size in pieces:
        ET.SubElement(
            carriage, "geom", name=f"fleet_lab_inspection_camera_frame_{name}",
            type="box", pos=" ".join(map(str, position)),
            size=" ".join(map(str, size / 2)),
            mass=str(0.004 * float(np.prod(size)) / volume),
            rgba=".12 .16 .18 1", contype="4", conaffinity="7")
    ET.SubElement(
        carriage, "geom", name=CAMERA_BACKPLATE_NAME,
        type="box", pos="0 0 .0853", size=".0125 .012 .0008",
        mass=".002", rgba=".05 .27 .18 1", contype="4", conaffinity="7")
    return ET.tostring(root, encoding="unicode")


class OnboardLabSimulation(FleetSimulation):
    """FleetSimulation variant whose only model change is the candidate camera."""

    def __init__(self, *, seed: int = 0, randomize: bool = False):
        super().__init__(seed=seed, randomize=randomize, only=["lab"])
        self.xml = inject_lab_inspection_camera(self.xml)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        if self.camera_id < 0:
            raise ValueError("candidate inspection camera was not compiled")
        if randomize:
            for geom_id in range(self.model.ngeom):
                name = self.model.geom(geom_id).name or ""
                if "robot_comp_wheel" in name:
                    self.model.geom_friction[geom_id, 0] *= self.wheel_friction_scale
        self.targets = {"lab": np.array((0.0, 0.0, 0.0, 0.025))}
        self.integrals = {"lab": np.zeros(2)}
        self.extension_targets = {"lab": 0.0}
        self.gate_targets = {"lab": np.zeros(0)}
        self.max_torque = 0.0
        self.max_tilt = 0.0
        self.trace = []
        for joint in self.robots["lab"]["jaw_joints"]:
            self.data.qpos[self.model.joint(joint).qposadr[0]] = 0.025
        mujoco.mj_forward(self.model, self.data)
        self.setup = initialize_competition_events(
            self.model, self.data, self.metadata)

    def camera_calibration(self, image_size: int) -> PinholeCamera:
        focal = 0.5 * image_size / math.tan(math.radians(CAMERA_FOVY_DEGREES) / 2)
        position = self.data.cam_xpos[self.camera_id]
        rotation = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        return PinholeCamera(
            focal, focal, image_size / 2, image_size / 2,
            tuple(position), tuple(map(tuple, rotation)))

    def carriage_plane(self, distance_from_camera_m: float) -> Plane:
        """Kinematic plane in the camera/carriage frame."""
        rotation = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        optical_forward = -rotation[:, 2]
        point = self.data.cam_xpos[self.camera_id] + distance_from_camera_m * optical_forward
        return Plane(tuple(point), tuple(optical_forward))

    def held_sample_top_plane(self) -> Plane:
        """Visible top rim: 0.9 mm above the sample-pad/gripper site."""
        return self.carriage_plane(0.0991)


def set_free_body_pose(simulation: OnboardLabSimulation, name: str,
                       xyz: np.ndarray, yaw: float = 0.0) -> None:
    joint = simulation.model.joint(f"{name}_free")
    address = int(joint.qposadr[0])
    simulation.data.qpos[address:address + 3] = xyz
    simulation.data.qpos[address + 3:address + 7] = (
        math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def reset_coupon(simulation: OnboardLabSimulation, rng: np.random.Generator,
                 mode: str) -> dict:
    """Reset a randomized floor, held-sample, or yellow-negative trial."""
    if mode not in {"floor", "held", "yellow_negative"}:
        raise ValueError("coupon mode must be floor, held, or yellow_negative")
    mujoco.mj_resetData(simulation.model, simulation.data)
    robot = simulation.robots["lab"]
    base_joint = simulation.model.joint(robot["freejoint"])
    base_address = int(base_joint.qposadr[0])
    base_xy = rng.uniform((0.42, 0.50), (0.70, 0.68))
    heading = rng.uniform(-math.pi, math.pi)
    model_yaw = heading - math.pi / 2
    simulation.data.qpos[base_address:base_address + 3] = (*base_xy, robot["wheel_radius_m"])
    simulation.data.qpos[base_address + 3:base_address + 7] = (
        math.cos(model_yaw / 2), 0.0, 0.0, math.sin(model_yaw / 2))
    for joint in robot["jaw_joints"]:
        simulation.data.qpos[simulation.model.joint(joint).qposadr[0]] = 0.025

    for index, item in enumerate(simulation.metadata["objects"]):
        if simulation.model.joint(f"{item['name']}_free").id >= 0:
            set_free_body_pose(simulation, item["name"],
                               np.array((-2.0 - 0.08 * index, -2.0, 0.04)))
    extension = float(rng.uniform(0.0, 0.040))
    direction = np.array((math.cos(heading), math.sin(heading)))
    right = np.array((math.sin(heading), -math.cos(heading)))
    if mode == "floor":
        local_offset = rng.uniform((-0.010, -0.010), (0.010, 0.010))
        target_xy = (base_xy + (0.065 + extension + local_offset[1]) * direction +
                     local_offset[0] * right)
        target_name = "Biological_Sample_01"
        target_z = 0.00252
    elif mode == "held":
        local_offset = rng.uniform((-0.00045, -0.00045), (0.00045, 0.00045))
        target_xy = (base_xy + (0.065 + extension + local_offset[1]) * direction +
                     local_offset[0] * right)
        target_name = "Biological_Sample_01"
        target_z = 0.00252
    else:
        local_offset = rng.uniform((-0.008, -0.008), (0.008, 0.008))
        target_xy = (base_xy + (0.065 + extension + local_offset[1]) * direction +
                     local_offset[0] * right)
        target_name = "Cylinder_Yellow_01"
        target_z = 0.01002
    set_free_body_pose(simulation, target_name,
                       np.array((*target_xy, target_z)), rng.uniform(-math.pi, math.pi))
    simulation.targets["lab"] = np.array((0.0, 0.0, 0.0, 0.025))
    simulation.integrals["lab"][:] = 0
    simulation.extension_targets["lab"] = extension
    simulation.data.ctrl[:] = 0
    mujoco.mj_forward(simulation.model, simulation.data)
    simulation.setup = initialize_competition_events(
        simulation.model, simulation.data, simulation.metadata)
    for _ in range(math.ceil(1.2 / simulation.control_dt)):
        simulation.command("lab", lift=0.0, jaw=0.025)
        simulation.step()
    if mode == "held":
        for _ in range(math.ceil(0.65 / simulation.control_dt)):
            simulation.command("lab", lift=0.0, jaw=0.0)
            simulation.step()
        lift = float(rng.uniform(0.018, 0.042))
        for _ in range(math.ceil(0.9 / simulation.control_dt)):
            simulation.command("lab", lift=lift, jaw=0.0)
            simulation.step()
    mujoco.mj_forward(simulation.model, simulation.data)
    return {
        "mode": mode,
        "target_name": target_name,
        "extension_m": extension,
        "camera_position_m": simulation.data.cam_xpos[simulation.camera_id].tolist(),
        "camera_module_mass_kg": 0.006,
    }


def randomize_render(simulation: OnboardLabSimulation, rng: np.random.Generator,
                     original_rgba: np.ndarray) -> None:
    simulation.model.geom_rgba[:] = original_rgba
    for light_id in range(simulation.model.nlight):
        simulation.model.light_diffuse[light_id] = rng.uniform(0.45, 1.0, 3)
        simulation.model.light_specular[light_id] = rng.uniform(0.04, 0.28, 3)
    simulation.model.vis.headlight.ambient[:] = rng.uniform(0.10, 0.28, 3)
    simulation.model.vis.headlight.diffuse[:] = rng.uniform(0.30, 0.68, 3)


def render_rgb(renderer: mujoco.Renderer, simulation: OnboardLabSimulation,
               rng: np.random.Generator) -> np.ndarray:
    renderer.update_scene(simulation.data, camera=simulation.camera_id)
    image = renderer.render().copy().astype(np.float32)
    image = image * rng.uniform(0.84, 1.16) + rng.uniform(-10, 10)
    image += rng.normal(0, rng.uniform(0.0, 2.5), image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)
