"""RGB-only geometric refinement for biological-sample docking crops."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class Plane:
    """A world-space plane used for camera-ray intersection."""

    point: tuple[float, float, float]
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def normalized(self) -> tuple[np.ndarray, np.ndarray]:
        point = np.asarray(self.point, dtype=np.float64)
        normal = np.asarray(self.normal, dtype=np.float64)
        length = float(np.linalg.norm(normal))
        if point.shape != (3,) or normal.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(normal).all() or length < 1e-9:
            raise ValueError("plane point and normal must be finite three-vectors")
        return point, normal / length


@dataclass(frozen=True)
class PinholeCamera:
    """Full-image intrinsics and a MuJoCo-style camera-to-world transform."""

    fx: float
    fy: float
    cx: float
    cy: float
    position_world: tuple[float, float, float]
    rotation_camera_to_world: tuple[tuple[float, float, float], ...]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(self.position_world, dtype=np.float64)
        rotation = np.asarray(self.rotation_camera_to_world, dtype=np.float64)
        if (position.shape != (3,) or rotation.shape != (3, 3) or
                not np.isfinite(position).all() or not np.isfinite(rotation).all()):
            raise ValueError("camera extrinsics must contain finite 3D arrays")
        if min(self.fx, self.fy) <= 0 or not np.isfinite((self.fx, self.fy, self.cx, self.cy)).all():
            raise ValueError("camera intrinsics must be finite with positive focal lengths")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
            raise ValueError("camera rotation must be orthonormal")
        return position, rotation

    def ray(self, pixel_xy: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        position, rotation = self.arrays()
        u, v = np.asarray(pixel_xy, dtype=np.float64)
        camera_ray = np.array(((u - self.cx) / self.fx,
                               -(v - self.cy) / self.fy, -1.0))
        world_ray = rotation @ camera_ray
        return position, world_ray / np.linalg.norm(world_ray)

    def intersect(self, pixel_xy: Iterable[float], plane: Plane) -> np.ndarray:
        origin, ray = self.ray(pixel_xy)
        point, normal = plane.normalized()
        denominator = float(ray @ normal)
        if abs(denominator) < 1e-7:
            raise ValueError("camera ray is parallel to the docking plane")
        distance = float((point - origin) @ normal / denominator)
        if distance <= 0:
            raise ValueError("docking plane is behind the camera")
        return origin + distance * ray

    def project(self, point_world: Iterable[float]) -> np.ndarray:
        position, rotation = self.arrays()
        camera = rotation.T @ (np.asarray(point_world, dtype=np.float64) - position)
        depth = -float(camera[2])
        if depth <= 1e-9:
            raise ValueError("point is behind the camera")
        return np.array((self.cx + self.fx * camera[0] / depth,
                         self.cy - self.fy * camera[1] / depth))


@dataclass(frozen=True)
class RefinerThresholds:
    minimum_cnn_confidence: float = 0.995
    minimum_arc_coverage: float = 0.48
    maximum_radial_residual_px: float = 1.45
    maximum_radius_relative_error: float = 0.24
    maximum_coarse_offset_px: float = 18.0
    minimum_edge_points: int = 36


def _otsu(values: np.ndarray) -> float:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if len(values) < 8:
        return 0.12
    histogram, edges = np.histogram(values, bins=96, range=(0.0, 1.0))
    probability = histogram.astype(np.float64) / max(histogram.sum(), 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    cumulative = np.cumsum(probability)
    mean = np.cumsum(probability * centers)
    total = mean[-1]
    denominator = cumulative * (1.0 - cumulative)
    score = np.zeros_like(denominator)
    valid = denominator > 1e-12
    score[valid] = (total * cumulative[valid] - mean[valid]) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(score))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    result = []
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) >= 12:
            result.append(np.asarray(pixels, dtype=np.int32))
    return result


def _boundary(component: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[component[:, 0], component[:, 1]] = True
    interior = mask.copy()
    interior[1:-1, 1:-1] &= (mask[:-2, 1:-1] & mask[2:, 1:-1] &
                              mask[1:-1, :-2] & mask[1:-1, 2:])
    y, x = np.nonzero(mask & ~interior)
    return np.column_stack((x.astype(np.float64) + 0.5,
                            y.astype(np.float64) + 0.5))


def _circle_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    first, second, third = points
    matrix = 2.0 * np.vstack((second - first, third - first))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-5:
        return None
    rhs = np.array((second @ second - first @ first,
                    third @ third - first @ first))
    center = np.linalg.solve(matrix, rhs)
    return center, float(np.linalg.norm(first - center))


def _refine_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack((2.0 * points, np.ones(len(points))))
    rhs = np.einsum("ij,ij->i", points, points)
    cx, cy, constant = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center = np.array((cx, cy))
    radius = math.sqrt(max(float(constant + center @ center), 0.0))
    return center, radius


def _fit_visible_circle(points: np.ndarray, expected_radius: float,
                        coarse_center: np.ndarray) -> dict | None:
    if len(points) < 12:
        return None
    if len(points) > 1200:
        points = points[np.linspace(0, len(points) - 1, 1200).astype(int)]
    tolerance = max(1.15, expected_radius * 0.035)
    rng = np.random.default_rng(73021 + len(points))
    best = None
    for _ in range(min(420, max(120, len(points)))):
        circle = _circle_three(points[rng.choice(len(points), 3, replace=False)])
        if circle is None:
            continue
        center, radius = circle
        if not (0.52 * expected_radius <= radius <= 1.55 * expected_radius):
            continue
        if np.linalg.norm(center - coarse_center) > max(30.0, expected_radius * 0.8):
            continue
        residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        inliers = residual <= tolerance
        if inliers.sum() < 12:
            continue
        angles = np.arctan2(points[inliers, 1] - center[1], points[inliers, 0] - center[0])
        bins = np.unique(np.floor((angles + math.pi) * 48 / (2 * math.pi)).astype(int)).size
        rank = (int(inliers.sum()), bins, -float(np.median(residual[inliers])))
        if best is None or rank > best[0]:
            best = (rank, inliers)
    if best is None:
        return None
    center, radius = _refine_circle(points[best[1]])
    residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    inliers = residual <= tolerance
    if inliers.sum() >= 8:
        center, radius = _refine_circle(points[inliers])
        residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        inliers = residual <= tolerance
    angles = np.arctan2(points[inliers, 1] - center[1], points[inliers, 0] - center[0])
    occupied = np.unique(np.floor((angles + math.pi) * 48 / (2 * math.pi)).astype(int)).size
    return {
        "center": center,
        "radius_px": float(radius),
        "radial_residual_px": float(np.median(residual[inliers])) if inliers.any() else float("inf"),
        "arc_coverage": float(occupied / 48),
        "edge_points": int(len(points)),
        "inlier_points": int(inliers.sum()),
    }


def _fit_plane_circle(points_full_px: np.ndarray, camera: PinholeCamera,
                      plane: Plane, coarse_full_px: np.ndarray,
                      radius_m: float) -> dict | None:
    """Fit the physical circle after undoing perspective on the known plane."""
    point, normal = plane.normalized()
    first, second = _plane_basis(normal)
    world = np.asarray([camera.intersect(pixel, plane) for pixel in points_full_px])
    coordinates = np.column_stack(((world - point) @ first, (world - point) @ second))
    coarse_world = camera.intersect(coarse_full_px, plane)
    coarse = np.array(((coarse_world - point) @ first, (coarse_world - point) @ second))
    one_x = camera.intersect(coarse_full_px + (1.0, 0.0), plane)
    one_y = camera.intersect(coarse_full_px + (0.0, 1.0), plane)
    meters_per_pixel = math.sqrt(
        (np.linalg.norm(one_x - coarse_world) ** 2 +
         np.linalg.norm(one_y - coarse_world) ** 2) / 2)
    tolerance = max(0.00025, 1.15 * meters_per_pixel)
    if len(coordinates) > 1200:
        coordinates = coordinates[np.linspace(0, len(coordinates) - 1, 1200).astype(int)]
    rng = np.random.default_rng(91331 + len(coordinates))
    best = None
    for _ in range(min(500, max(140, len(coordinates)))):
        circle = _circle_three(coordinates[rng.choice(len(coordinates), 3, replace=False)])
        if circle is None:
            continue
        center, radius = circle
        if not (0.52 * radius_m <= radius <= 1.55 * radius_m):
            continue
        if np.linalg.norm(center - coarse) > max(0.025, radius_m * 0.8):
            continue
        residual = np.abs(np.linalg.norm(coordinates - center, axis=1) - radius)
        inliers = residual <= tolerance
        if inliers.sum() < 12:
            continue
        angles = np.arctan2(coordinates[inliers, 1] - center[1],
                            coordinates[inliers, 0] - center[0])
        bins = np.unique(np.floor((angles + math.pi) * 48 / (2 * math.pi)).astype(int)).size
        rank = (int(inliers.sum()), bins, -float(np.median(residual[inliers])))
        if best is None or rank > best[0]:
            best = (rank, inliers)
    if best is None:
        return None
    center, radius = _refine_circle(coordinates[best[1]])
    residual = np.abs(np.linalg.norm(coordinates - center, axis=1) - radius)
    inliers = residual <= tolerance
    if inliers.sum() >= 8:
        center, radius = _refine_circle(coordinates[inliers])
        residual = np.abs(np.linalg.norm(coordinates - center, axis=1) - radius)
        inliers = residual <= tolerance
    angles = np.arctan2(coordinates[inliers, 1] - center[1],
                        coordinates[inliers, 0] - center[0])
    occupied = np.unique(np.floor((angles + math.pi) * 48 / (2 * math.pi)).astype(int)).size
    center_world = point + center[0] * first + center[1] * second
    return {
        "center": camera.project(center_world),
        "world_point": center_world,
        "radius_m": float(radius),
        "radial_residual_px": float(np.median(residual[inliers]) / meters_per_pixel),
        "arc_coverage": float(occupied / 48),
        "edge_points": int(len(coordinates)),
        "inlier_points": int(inliers.sum()),
    }


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array((0.0, 0.0, 1.0))
    if abs(float(normal @ reference)) > 0.9:
        reference = np.array((1.0, 0.0, 0.0))
    first = np.cross(normal, reference)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def expected_disc_radius_px(camera: PinholeCamera, plane: Plane,
                            center_pixel_full: Iterable[float], radius_m: float) -> float:
    center_world = camera.intersect(center_pixel_full, plane)
    _, normal = plane.normalized()
    first, second = _plane_basis(normal)
    center_pixel = camera.project(center_world)
    radii = []
    for axis in (first, second):
        radii.extend((np.linalg.norm(camera.project(center_world + radius_m * axis) - center_pixel),
                      np.linalg.norm(camera.project(center_world - radius_m * axis) - center_pixel)))
    return float(np.mean(radii))


def rejection_reason(result: Mapping, thresholds: RefinerThresholds) -> str | None:
    if result.get("class_name") != "sample":
        return "cnn_not_sample"
    if float(result.get("cnn_confidence", 0.0)) < thresholds.minimum_cnn_confidence:
        return "cnn_confidence"
    if not result.get("fit_found", False):
        return "circle_fit"
    if result["edge_points"] < thresholds.minimum_edge_points:
        return "edge_support"
    if result["arc_coverage"] < thresholds.minimum_arc_coverage:
        return "arc_coverage"
    if result["radial_residual_px"] > thresholds.maximum_radial_residual_px:
        return "radial_residual"
    if result["radius_relative_error"] > thresholds.maximum_radius_relative_error:
        return "radius_consistency"
    if result["coarse_offset_px"] > thresholds.maximum_coarse_offset_px:
        return "coarse_disagreement"
    return None


class BestSampleGeometryRefiner:
    """Fit the 56 mm sample rim and map its RGB center to a world plane."""

    def __init__(self, thresholds: RefinerThresholds = RefinerThresholds(),
                 sample_diameter_m: float = 0.056):
        self.thresholds = thresholds
        self.radius_m = float(sample_diameter_m) * 0.5
        if self.radius_m <= 0:
            raise ValueError("sample diameter must be positive")

    def refine(self, rgb_crop: np.ndarray, coarse: Mapping,
               camera: PinholeCamera, plane: Plane,
               crop_origin_px: Iterable[float] = (0.0, 0.0)) -> dict:
        image = np.asarray(rgb_crop)
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
            raise ValueError("rgb_crop must be an H by W RGB image of at least 32 pixels")
        rgb = image.astype(np.float64)
        if rgb.max() > 1.5:
            rgb /= 255.0
        coarse_center = np.asarray(coarse["center_pixel_xy"], dtype=np.float64)
        crop_origin = np.asarray(crop_origin_px, dtype=np.float64)
        if coarse_center.shape != (2,) or crop_origin.shape != (2,):
            raise ValueError("coarse center and crop origin must be two-vectors")
        full_coarse = coarse_center + crop_origin
        expected_radius = expected_disc_radius_px(camera, plane, full_coarse, self.radius_m)

        red, green, blue = np.moveaxis(rgb, 2, 0)
        score = np.clip(np.minimum(red, green) - blue - 0.12 * np.abs(red - green), 0.0, 1.0)
        color_gate = ((red - blue > 0.055) & (green - blue > 0.035) &
                      (red > 0.16) & (green > 0.12))
        useful = score[color_gate]
        otsu = _otsu(useful)
        thresholds = sorted(set(max(0.035, float(value)) for value in
                                (otsu * 0.72, otsu * 0.9, otsu, otsu * 1.12,
                                 np.quantile(useful, 0.35) if len(useful) else 0.12)))
        candidates = []
        for color_threshold in thresholds:
            for component in _components(color_gate & (score >= color_threshold)):
                area = len(component)
                expected_area = math.pi * expected_radius ** 2
                if not (0.08 * expected_area <= area <= 1.8 * expected_area):
                    continue
                center_yx = component.mean(axis=0)
                component_center = center_yx[::-1] + 0.5
                if np.linalg.norm(component_center - coarse_center) > max(38.0, expected_radius):
                    continue
                boundary = _boundary(component, image.shape[:2])
                fit = _fit_plane_circle(boundary + crop_origin, camera, plane,
                                        full_coarse, self.radius_m)
                if fit is None:
                    continue
                fit["color_threshold"] = color_threshold
                fit["component_area_px"] = int(area)
                fit["rank"] = (fit["arc_coverage"], fit["inlier_points"],
                               -fit["radial_residual_px"])
                candidates.append(fit)

        result = {
            "class_name": coarse.get("class_name"),
            "cnn_confidence": float(coarse.get("confidence", 0.0)),
            "coarse_center_crop_px": coarse_center.tolist(),
            "expected_radius_px": expected_radius,
            "fit_found": bool(candidates),
            "thresholds": asdict(self.thresholds),
        }
        if candidates:
            fit = max(candidates, key=lambda item: item["rank"])
            center_full = fit.pop("center")
            center_crop = center_full - crop_origin
            world = fit.pop("world_point")
            camera_position, camera_rotation = camera.arrays()
            camera_point = camera_rotation.T @ (world - camera_position)
            fit.pop("rank")
            result.update(fit)
            result.update({
                "center_crop_px": center_crop.tolist(),
                "center_full_px": center_full.tolist(),
                "world_point_m": world.tolist(),
                "camera_point_m": camera_point.tolist(),
                "camera_lateral_xy_m": camera_point[:2].tolist(),
                "coarse_offset_px": float(np.linalg.norm(center_crop - coarse_center)),
                "radius_px": float(expected_radius * result["radius_m"] / self.radius_m),
                "radius_relative_error": float(abs(result["radius_m"] - self.radius_m) /
                                                self.radius_m),
            })
            one_x = camera.intersect(center_full + (1.0, 0.0), plane)
            one_y = camera.intersect(center_full + (0.0, 1.0), plane)
            mm_per_pixel = 1000.0 * math.sqrt(
                (np.linalg.norm(one_x - world) ** 2 + np.linalg.norm(one_y - world) ** 2) / 2)
            conservative_pixel_std = max(0.25, result["radial_residual_px"])
            result["estimated_position_std_mm"] = conservative_pixel_std * mm_per_pixel
        reason = rejection_reason(result, self.thresholds)
        result["accepted"] = reason is None
        result["rejection_reason"] = reason
        return result

    __call__ = refine
