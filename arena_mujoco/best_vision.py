"""MuJoCo-rendered perception data and a compact multi-head vision model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .builder import build_arena, numbers


CLASS_NAMES = ("background", "red", "yellow", "green", "kit", "sample")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
DATASET_SCHEMA = 1
MODEL_SCHEMA = 1


@dataclass(frozen=True)
class VisionConfig:
    """Shared image, camera, and model contract."""

    image_size: int = 72
    channels: int = 32
    camera_height_m: tuple[float, float] = (0.10, 0.18)
    camera_fovy_degrees: tuple[float, float] = (48.0, 68.0)
    camera_tilt_degrees: float = 8.0
    target_center_limit: float = 0.22
    settle_steps: int = 8

    def validate(self) -> None:
        if self.image_size < 32 or self.image_size > 512:
            raise ValueError("image_size must be between 32 and 512")
        if self.channels < 8 or self.channels % 8:
            raise ValueError("channels must be at least 8 and divisible by 8")
        if not 0 < self.camera_height_m[0] <= self.camera_height_m[1]:
            raise ValueError("camera height range is invalid")
        if not 10 < self.camera_fovy_degrees[0] <= self.camera_fovy_degrees[1] < 120:
            raise ValueError("camera field-of-view range is invalid")
        if not 0 <= self.target_center_limit < 0.4:
            raise ValueError("target_center_limit must be in [0, 0.4)")


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.depthwise_norm = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pointwise_norm = nn.BatchNorm2d(out_channels)
        self.use_skip = stride == 1 and in_channels == out_channels

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = F.silu(self.depthwise_norm(self.depthwise(value)), inplace=True)
        result = self.pointwise_norm(self.pointwise(result))
        if self.use_skip:
            result = result + value
        return F.silu(result, inplace=True)


class BestVisionCNN(nn.Module):
    """Classify one proposed object crop and localize its projected center."""

    def __init__(self, channels: int = 32, class_count: int = len(CLASS_NAMES)):
        super().__init__()
        c = channels
        self.features = nn.Sequential(
            ConvNormAct(3, c, 2),
            DepthwiseBlock(c, c, 1),
            DepthwiseBlock(c, c * 2, 2),
            DepthwiseBlock(c * 2, c * 2, 1),
            DepthwiseBlock(c * 2, c * 3, 2),
            DepthwiseBlock(c * 3, c * 3, 1),
            DepthwiseBlock(c * 3, c * 4, 1),
        )
        self.embedding = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                       nn.Linear(c * 4, c * 4), nn.SiLU())
        self.classifier = nn.Linear(c * 4, class_count)
        self.center_heatmap = nn.Conv2d(c * 4, 1, 1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(image)
        logits = self.classifier(self.embedding(features))
        heatmap = self.center_heatmap(features).flatten(2).softmax(-1).reshape(
            features.shape[0], features.shape[2], features.shape[3])
        x_coordinates = torch.linspace(0.0, 1.0, features.shape[3], device=image.device)
        y_coordinates = torch.linspace(0.0, 1.0, features.shape[2], device=image.device)
        center_x = (heatmap * x_coordinates.view(1, 1, -1)).sum((1, 2))
        center_y = (heatmap * y_coordinates.view(1, -1, 1)).sum((1, 2))
        return logits, torch.stack((center_x, center_y), dim=1)


class CalibratedVision(nn.Module):
    """TorchScript export with validation-fitted logit temperature."""

    def __init__(self, model: BestVisionCNN, temperature: float):
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(max(float(temperature), 1e-4)))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, center = self.model(image)
        return logits / self.temperature, center


def preprocess_rgb(images: torch.Tensor) -> torch.Tensor:
    """Convert NHWC uint8 or float images to normalized NCHW floats."""
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError("images must have HWC or NHWC shape")
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    if images.shape[1] != 3:
        raise ValueError("images must have three RGB channels")
    images = images.float()
    if images.max().item() > 1.5:
        images = images / 255.0
    mean = images.new_tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
    scale = images.new_tensor((0.25, 0.25, 0.25)).view(1, 3, 1, 1)
    return (images - mean) / scale


class FrozenVisionDataset(Dataset):
    """Memory-mapped split generated by :class:`MujocoVisionGenerator`."""

    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = split
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No frozen vision dataset manifest at {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if split not in self.manifest["splits"]:
            raise ValueError(f"Dataset has no {split!r} split")
        self.images = np.load(self.root / f"{split}_images.npy", mmap_mode="r")
        self.labels = np.load(self.root / f"{split}_labels.npy", mmap_mode="r")
        self.centers = np.load(self.root / f"{split}_centers.npy", mmap_mode="r")
        if not (len(self.images) == len(self.labels) == len(self.centers)):
            raise ValueError(f"Frozen {split} arrays have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(np.array(self.images[index], copy=True))
        return image, torch.tensor(int(self.labels[index])), torch.from_numpy(
            np.array(self.centers[index], copy=True))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_provenance(root: str | Path) -> dict:
    root = Path(root)
    names = (
        "arena_spec.json",
        "arena_mujoco/assets/biohazard.json",
        "arena_mujoco/builder.py",
        "arena_mujoco/geometry.py",
        "arena_mujoco/materials.py",
        "arena_mujoco/best_vision.py",
        "scripts/train_best_vision.py",
        "scripts/test_best_vision.py",
        "requirements-best-design.txt",
    )
    return {name: sha256_file(root / name) for name in names if (root / name).is_file()}


def _add_vision_camera_and_occluders(xml: str) -> str:
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Arena MJCF has no worldbody")
    ET.SubElement(world, "camera", name="vision_inspection", pos=".57 .59 .14",
                  quat="1 0 0 0", fovy="58")
    for index in range(4):
        body = ET.SubElement(world, "body", name=f"vision_occluder_{index}", pos="-3 -3 .015")
        ET.SubElement(body, "freejoint", name=f"vision_occluder_{index}_free")
        ET.SubElement(body, "geom", name=f"vision_occluder_{index}_geom", type="box",
                      size=".006 .025 .015", rgba=".25 .25 .25 1", density="400",
                      contype="0", conaffinity="0")
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64).reshape(-1))
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def _camera_rotation(position: np.ndarray, look_at: np.ndarray, image_yaw: float) -> np.ndarray:
    forward = look_at - position
    forward /= np.linalg.norm(forward)
    image_up = np.array([math.cos(image_yaw), math.sin(image_yaw), 0.0])
    image_up -= forward * np.dot(image_up, forward)
    image_up /= np.linalg.norm(image_up)
    image_right = np.cross(forward, image_up)
    image_right /= np.linalg.norm(image_right)
    return np.column_stack((image_right, image_up, -forward))


def _ray_plane_point(camera_position: np.ndarray, rotation: np.ndarray, center: np.ndarray,
                     fovy_degrees: float, plane_z: float) -> np.ndarray:
    tangent = math.tan(math.radians(fovy_degrees) / 2)
    ray_camera = np.array(((2 * center[0] - 1) * tangent,
                           (1 - 2 * center[1]) * tangent, -1.0))
    ray_world = rotation @ ray_camera
    distance = (plane_z - camera_position[2]) / ray_world[2]
    return camera_position + distance * ray_world


def project_world_point(data: mujoco.MjData, camera_id: int, point: Iterable[float],
                        fovy_degrees: float, width: int, height: int) -> np.ndarray:
    """Project a world point to normalized image coordinates using a fixed camera."""
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    relative = np.asarray(point) - data.cam_xpos[camera_id]
    camera = rotation.T @ relative
    depth = -camera[2]
    if depth <= 0:
        return np.array((np.nan, np.nan), dtype=np.float32)
    tangent = math.tan(math.radians(fovy_degrees) / 2)
    aspect = width / height
    return np.array(((camera[0] / (depth * tangent * aspect) + 1) / 2,
                     (1 - camera[1] / (depth * tangent)) / 2), dtype=np.float32)


class MujocoVisionGenerator:
    """Render frozen object crops from the senior preliminary MuJoCo arena."""

    def __init__(self, config: VisionConfig):
        config.validate()
        self.config = config
        xml, metadata = build_arena("senior_preliminary", tape_mode="rigid", robot=False)
        self.model = mujoco.MjModel.from_xml_string(_add_vision_camera_and_occluders(xml))
        self.data = mujoco.MjData(self.model)
        self.metadata = metadata
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "vision_inspection")
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.object_records = self._find_objects()
        self.occluders = [self._free_joint(f"vision_occluder_{index}_free") for index in range(4)]
        self.occluder_geom_ids = [mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, f"vision_occluder_{index}_geom")
            for index in range(4)]
        self.original_rgba = self.model.geom_rgba.copy()
        self.renderer = mujoco.Renderer(
            self.model, height=config.image_size, width=config.image_size)

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> "MujocoVisionGenerator":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _free_joint(self, name: str) -> tuple[int, int]:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Missing free joint {name}")
        return joint_id, int(self.model.jnt_qposadr[joint_id])

    def _find_objects(self) -> list[dict]:
        records = []
        for item in self.metadata["objects"]:
            name = item["name"]
            if name.startswith("Cylinder_Red"):
                label = "red"
            elif name.startswith("Cylinder_Yellow"):
                label = "yellow"
            elif name.startswith("Cylinder_Green"):
                label = "green"
            elif name.startswith("Medical_Kit"):
                label = "kit"
            elif name.startswith("Biological_Sample"):
                label = "sample"
            else:
                continue
            joint_id, qpos_address = self._free_joint(f"{name}_free")
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id).tolist()
            records.append({"name": name, "class": label, "joint_id": joint_id,
                            "qpos_address": qpos_address, "body_id": body_id,
                            "geom_ids": geom_ids,
                            "z": float(self.data.qpos[qpos_address + 2])})
        missing = set(CLASS_NAMES[1:]) - {record["class"] for record in records}
        if missing:
            raise ValueError(f"Arena is missing vision classes: {sorted(missing)}")
        return records

    def _set_free_pose(self, record: dict | tuple[int, int], xyz: Iterable[float], yaw: float) -> None:
        address = record["qpos_address"] if isinstance(record, dict) else record[1]
        self.data.qpos[address:address + 3] = xyz
        self.data.qpos[address + 3:address + 7] = (
            math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def _hide_objects(self) -> None:
        for index, record in enumerate(self.object_records):
            self._set_free_pose(record, (-2.0 - 0.08 * index, -2.0, record["z"]), 0.0)
        for index, record in enumerate(self.occluders):
            self._set_free_pose(record, (-3.0 - 0.08 * index, -3.0, 0.015), 0.0)

    def _randomize_appearance(self, rng: np.random.Generator) -> None:
        self.model.geom_rgba[:] = self.original_rgba
        floor = rng.uniform(0.72, 1.0)
        warmth = rng.uniform(-0.035, 0.035)
        self.model.geom_rgba[self.floor_id, :3] = np.clip(
            (floor + warmth, floor + warmth * 0.45, floor - warmth), 0.55, 1.0)
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            rgb = self.model.geom_rgba[geom_id, :3]
            if name.startswith("Cylinder_") or name.startswith("Kit_") or name.startswith("Medical_") or name.startswith("Biological_"):
                rgb *= rng.uniform(0.72, 1.2)
                rgb += rng.normal(0, 0.018, 3)
                self.model.geom_rgba[geom_id, :3] = np.clip(rgb, 0.01, 1.0)
            elif name.startswith("tape_"):
                value = rng.uniform(0.005, 0.09)
                self.model.geom_rgba[geom_id, :3] = value
        for light_id in range(self.model.nlight):
            self.model.light_diffuse[light_id] = rng.uniform(0.35, 1.05, 3)
            self.model.light_specular[light_id] = rng.uniform(0.05, 0.35, 3)
            direction = self.model.light_dir[light_id] + rng.normal(0, 0.10, 3)
            direction[2] = -abs(direction[2])
            self.model.light_dir[light_id] = direction / np.linalg.norm(direction)
        self.model.vis.headlight.ambient[:] = rng.uniform(0.08, 0.32, 3)
        self.model.vis.headlight.diffuse[:] = rng.uniform(0.25, 0.72, 3)

    def render_sample(self, label_index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= label_index < len(CLASS_NAMES):
            raise ValueError("label_index is outside the class table")
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        self._hide_objects()
        self._randomize_appearance(rng)
        height = rng.uniform(*self.config.camera_height_m)
        fovy = rng.uniform(*self.config.camera_fovy_degrees)
        camera_xy = np.array((rng.uniform(0.20, 0.94), rng.uniform(0.20, 0.98)))
        camera_position = np.array((*camera_xy, height))
        tilt_radius = height * math.tan(math.radians(rng.uniform(0, self.config.camera_tilt_degrees)))
        tilt_yaw = rng.uniform(-math.pi, math.pi)
        look_at = np.array((camera_xy[0] + tilt_radius * math.cos(tilt_yaw),
                            camera_xy[1] + tilt_radius * math.sin(tilt_yaw), 0.0))
        image_yaw = rng.uniform(-math.pi, math.pi)
        rotation = _camera_rotation(camera_position, look_at, image_yaw)
        self.model.cam_pos[self.camera_id] = camera_position
        self.model.cam_quat[self.camera_id] = _matrix_to_quaternion(rotation)
        self.model.cam_fovy[self.camera_id] = fovy

        target = None
        requested_center = np.array((0.5, 0.5), dtype=np.float32)
        if label_index:
            class_name = CLASS_NAMES[label_index]
            options = [record for record in self.object_records if record["class"] == class_name]
            target = options[int(rng.integers(len(options)))]
            limit = self.config.target_center_limit
            requested_center = rng.uniform(0.5 - limit, 0.5 + limit, 2).astype(np.float32)
            point = _ray_plane_point(camera_position, rotation, requested_center, fovy, target["z"])
            self._set_free_pose(target, (*point[:2], target["z"]), rng.uniform(-math.pi, math.pi))

        available = [record for record in self.object_records if record is not target]
        rng.shuffle(available)
        for record in available[:int(rng.integers(0, 4))]:
            angle = rng.uniform(-math.pi, math.pi)
            radius = rng.uniform(0.48, 0.78)
            edge_center = np.array((0.5 + radius * math.cos(angle),
                                    0.5 + radius * math.sin(angle)))
            point = _ray_plane_point(camera_position, rotation, edge_center, fovy, record["z"])
            self._set_free_pose(record, (*point[:2], record["z"]), rng.uniform(-math.pi, math.pi))

        occluder_count = int(rng.integers(0, 3)) if target is not None else int(rng.integers(0, 2))
        for index in range(occluder_count):
            angle = rng.uniform(-math.pi, math.pi)
            distance = rng.uniform(0.07, 0.16)
            center = requested_center + distance * np.array((math.cos(angle), math.sin(angle)))
            center = np.clip(center, 0.16, 0.84)
            point = _ray_plane_point(camera_position, rotation, center, fovy, 0.015)
            self._set_free_pose(self.occluders[index], (*point[:2], 0.015), rng.uniform(-math.pi, math.pi))
            geom_id = self.occluder_geom_ids[index]
            self.model.geom_size[geom_id] = (rng.uniform(0.001, 0.0035),
                                             rng.uniform(0.008, 0.022),
                                             rng.uniform(0.005, 0.016))
            value = rng.uniform(0.08, 0.72)
            self.model.geom_rgba[geom_id, :3] = np.clip(
                value + rng.normal(0, 0.06, 3), 0.02, 0.9)

        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.config.settle_steps):
            mujoco.mj_step(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        image = self.renderer.render().copy()
        gain = rng.uniform(0.82, 1.18)
        offset = rng.uniform(-12, 12)
        image = np.clip(image.astype(np.float32) * gain + offset +
                        rng.normal(0, rng.uniform(0, 3.5), image.shape), 0, 255).astype(np.uint8)
        if target is None:
            center = np.array((0.5, 0.5), dtype=np.float32)
        else:
            center = project_world_point(
                self.data, self.camera_id, self.data.xpos[target["body_id"]], fovy,
                self.config.image_size, self.config.image_size)
            if not np.isfinite(center).all() or not ((center > 0.05) & (center < 0.95)).all():
                raise RuntimeError(f"Target projection escaped the crop: {center.tolist()}")
        return image, center


def generate_frozen_dataset(root: str | Path, config: VisionConfig, split_counts: dict[str, int],
                            split_seeds: dict[str, int], project_root: str | Path,
                            force: bool = False) -> dict:
    """Generate train/validation/test arrays with independent random streams."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    expected = {"schema_version": DATASET_SCHEMA, "classes": list(CLASS_NAMES),
                "vision_config": json.loads(json.dumps(asdict(config))), "split_counts": split_counts,
                "split_seeds": split_seeds}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text())
        if all(existing.get(key) == value for key, value in expected.items()):
            for split, record in existing["splits"].items():
                for kind, file_record in record["files"].items():
                    path = root / file_record["file"]
                    if not path.is_file() or sha256_file(path) != file_record["sha256"]:
                        raise ValueError(f"Frozen {split} {kind} data is missing or changed: {path}")
            return existing
        raise ValueError(f"Dataset at {root} has a different configuration; use --force-generate")
    if len(set(split_seeds.values())) != len(split_seeds):
        raise ValueError("Every dataset split needs a distinct seed")
    if any(count <= 0 for count in split_counts.values()):
        raise ValueError("Every requested split must contain at least one sample")
    split_records = {}
    generation_started = time.perf_counter()
    with MujocoVisionGenerator(config) as generator:
        for split, count in split_counts.items():
            started = time.perf_counter()
            image_tmp = root / f".{split}_images.tmp.npy"
            label_tmp = root / f".{split}_labels.tmp.npy"
            center_tmp = root / f".{split}_centers.tmp.npy"
            images = np.lib.format.open_memmap(
                image_tmp, mode="w+", dtype=np.uint8,
                shape=(count, config.image_size, config.image_size, 3))
            labels = np.lib.format.open_memmap(label_tmp, mode="w+", dtype=np.uint8, shape=(count,))
            centers = np.lib.format.open_memmap(center_tmp, mode="w+", dtype=np.float32, shape=(count, 2))
            rng = np.random.default_rng(split_seeds[split])
            balanced = np.resize(np.arange(len(CLASS_NAMES), dtype=np.uint8), count)
            rng.shuffle(balanced)
            for index, label in enumerate(balanced):
                images[index], centers[index] = generator.render_sample(int(label), rng)
                labels[index] = label
                if (index + 1) % 1000 == 0:
                    print(json.dumps({"phase": "render", "split": split,
                                      "complete": index + 1, "total": count}), flush=True)
            images.flush(); labels.flush(); centers.flush()
            del images, labels, centers
            paths = {}
            for kind, temporary in (("images", image_tmp), ("labels", label_tmp), ("centers", center_tmp)):
                final = root / f"{split}_{kind}.npy"
                os.replace(temporary, final)
                paths[kind] = {"file": final.name, "sha256": sha256_file(final),
                               "bytes": final.stat().st_size}
            elapsed = time.perf_counter() - started
            split_records[split] = {
                "count": count,
                "seed": split_seeds[split],
                "class_counts": {name: int(np.count_nonzero(balanced == index))
                                 for index, name in enumerate(CLASS_NAMES)},
                "files": paths,
                "generation_seconds": elapsed,
                "samples_per_second": count / max(elapsed, 1e-9),
            }
    manifest = {
        **expected,
        "generator": "native MuJoCo Renderer from build_arena('senior_preliminary')",
        "mujoco_version": mujoco.__version__,
        "render_backend": os.environ.get("MUJOCO_GL", "platform-default"),
        "camera_contract": {
            "name": "vision_inspection",
            "projection": "perspective",
            "resolution_px": [config.image_size, config.image_size],
            "height_above_board_m": list(config.camera_height_m),
            "vertical_fovy_degrees": list(config.camera_fovy_degrees),
            "maximum_tilt_degrees": config.camera_tilt_degrees,
            "crop_rule": "one proposed target in the central crop; zero to three edge distractors",
            "center_label": "projection of the target MuJoCo body origin, normalized left/top=0 and right/bottom=1",
        },
        "domain_randomization": [
            "camera height, field of view, tilt, yaw, and crop offset",
            "native light color, direction, headlight, paper tone, and material brightness",
            "object pose, edge distractors, physical box occluders, gain, offset, and sensor noise",
        ],
        "splits": split_records,
        "source_sha256": source_provenance(project_root),
        "generation_seconds": time.perf_counter() - generation_started,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray,
                     class_count: int = len(CLASS_NAMES)) -> np.ndarray:
    result = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(result, (labels.astype(int), predictions.astype(int)), 1)
    return result


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(1)
    matrix = confusion_matrix(labels, predictions, probabilities.shape[1])
    per_class = {}
    f1_values = []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": int(matrix[index].sum())}
    confidence = probabilities.max(1)
    correct = predictions == labels
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": float(np.mean(f1_values)),
        "negative_log_likelihood": float(-np.log(
            probabilities[np.arange(len(labels)), labels].clip(1e-9)).mean()),
        "expected_calibration_error_15_bin": expected_calibration_error(confidence, correct, 15),
        "confusion_true_rows_predicted_columns": matrix.tolist(),
        "per_class": per_class,
    }


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    result = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(result)


def localization_metrics(labels: np.ndarray, centers: np.ndarray,
                         predictions: np.ndarray, image_size: int) -> dict:
    foreground = labels != CLASS_TO_INDEX["background"]
    error = np.linalg.norm(predictions[foreground] - centers[foreground], axis=1)
    if not len(error):
        return {"foreground_count": 0}
    pixels = error * image_size
    return {
        "foreground_count": int(len(error)),
        "mean_normalized": float(error.mean()),
        "median_pixels": float(np.median(pixels)),
        "mean_pixels": float(pixels.mean()),
        "p95_pixels": float(np.quantile(pixels, 0.95)),
        "max_pixels": float(pixels.max()),
    }


class BestVisionPredictor:
    """Callable RGB crop inference API backed by the TorchScript export."""

    def __init__(self, artifact: str | Path, device: str = "cpu"):
        artifact = Path(artifact)
        if artifact.is_dir():
            artifact = artifact / "model.ts"
        self.artifact = artifact
        self.device = torch.device(device)
        self.model = torch.jit.load(str(artifact), map_location=self.device).eval()
        manifest_path = artifact.parent / "training.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        self.image_size = int(self.manifest.get("vision_config", {}).get("image_size", 72))

    @torch.inference_mode()
    def predict_batch(self, crops: np.ndarray | torch.Tensor) -> list[dict]:
        tensor = torch.as_tensor(crops, device=self.device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        original_height, original_width = int(tensor.shape[1]), int(tensor.shape[2])
        tensor = preprocess_rgb(tensor)
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(tensor, (self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False)
        logits, centers = self.model(tensor)
        probabilities = logits.softmax(1)
        confidence, labels = probabilities.max(1)
        results = []
        for index in range(len(tensor)):
            center = centers[index].cpu().tolist()
            results.append({
                "class_index": int(labels[index]),
                "class_name": CLASS_NAMES[int(labels[index])],
                "confidence": float(confidence[index]),
                "center_normalized_xy": center,
                "center_pixel_xy": [center[0] * original_width, center[1] * original_height],
                "probabilities": {name: float(probabilities[index, class_index])
                                  for class_index, name in enumerate(CLASS_NAMES)},
            })
        return results

    def __call__(self, crop: np.ndarray | torch.Tensor) -> dict:
        return self.predict_batch(crop)[0]
