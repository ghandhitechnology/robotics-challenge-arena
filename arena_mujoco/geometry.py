"""Geometry for the challenge arena, without a simulator dependency.

Specification values are millimetres. Returned geometry uses world coordinates
in metres, with the playing surface at Z=0. Only NumPy is required.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_MM = 0.001
_MAX_HOLE_ERROR_M = 0.00005


def _positive(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _dimensions(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    if spec.get("units", "millimeters") != "millimeters":
        raise ValueError("The arena specification must use millimeters")
    return spec["dimensions"]


def _bounds(values: Sequence[float]) -> list[float]:
    bounds = np.asarray(values, dtype=float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError("bounds_m must contain four finite coordinates")
    x0, y0, x1, y1 = bounds.tolist()
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Rectangle bounds must have positive width and height")
    return [x0, y0, x1, y1]


def tape_strips(spec: Mapping[str, Any], joins: str = "butt") -> list[dict[str, Any]]:
    """Return ten roll-width rectangles covering all four tape markings.

    ``butt`` partitions the original footprint without planar overlap. For
    ``overlap``, each transverse arm extends over the full supporting strip.
    ``layer=1`` identifies an upper arm; it is *not* a uniform Z translation.
    Only its ``overlap_regions`` have another strip beneath them. Its remaining
    ``floor_regions_m`` must stay floor-seated. The caller owns the height
    transition and any physical adhesion or inter-strip bonds.

    Each record contains ``name``, ``bounds_m``, ``layer``, ``support_names``,
    ``overlap_regions`` and ``floor_regions_m``. Overlap entries name their
    support and give an exact rectangular intersection in world metres.
    """
    if joins not in {"butt", "overlap"}:
        raise ValueError("joins must be 'butt' or 'overlap'")
    dim = _dimensions(spec)
    width = _positive(dim["surface_x"], "surface_x")
    height = _positive(dim["surface_y"], "surface_y")
    tape = _positive(dim["tape_width"], "tape_width")
    healthcare_depth = _positive(dim["healthcare_clear_depth"], "healthcare_clear_depth")
    middle = _positive(dim["healthcare_middle_clear_length"], "healthcare_middle_clear_length")
    upper = _positive(
        dim.get("healthcare_upper_clear_length", dim.get("healthcare_end_clear_length")),
        "healthcare_upper_clear_length",
    )
    lower = height - upper - middle - 2 * tape
    if lower <= 0:
        raise ValueError("The healthcare chain exceeds the field length")
    if "healthcare_lower_clear_length" in dim and not math.isclose(
        float(dim["healthcare_lower_clear_length"]), lower, abs_tol=1e-8,
    ):
        raise ValueError("healthcare_lower_clear_length disagrees with the dimension chain")
    start_x = width - _positive(dim["start_clear_depth"], "start_clear_depth")
    start_y = height - _positive(dim["start_clear_length"], "start_clear_length")
    isolation = _positive(dim["isolation_clear_side"], "isolation_clear_side")
    isolation_x = width - isolation
    left, right = (float(value) for value in dim["center_bar_x"])
    low, high = (float(value) for value in dim["center_bar_y_limits"])
    center = float(dim["center_crossbar_y"])
    half = tape / 2
    if not all(math.isfinite(value) for value in (left, right, low, high, center)):
        raise ValueError("Central H coordinates must be finite")
    if right - left <= tape or not low <= center - half < center + half <= high:
        raise ValueError("Central H bars and connector do not form a valid marking")

    healthcare = "Tape_Healthcare_Vertical"
    start = "Tape_Start_Vertical"
    isolation_name = "Tape_Isolation_Vertical"
    h_left = "Tape_Center_Left"
    h_right = "Tape_Center_Right"
    # name, butt bounds in mm, supporting strip names, extended overlap bounds.
    definitions = [
        (healthcare, [healthcare_depth, 0, healthcare_depth + tape, height], [], None),
        ("Tape_Healthcare_Lower_Arm", [0, lower, healthcare_depth, lower + tape],
         [healthcare], [0, lower, healthcare_depth + tape, lower + tape]),
        ("Tape_Healthcare_Upper_Arm",
         [0, lower + tape + middle, healthcare_depth, height - upper],
         [healthcare], [0, lower + tape + middle, healthcare_depth + tape, height - upper]),
        (start, [start_x - tape, start_y - tape, start_x, height], [], None),
        ("Tape_Start_Horizontal", [start_x, start_y - tape, width, start_y],
         [start], [start_x - tape, start_y - tape, width, start_y]),
        (isolation_name, [isolation_x - tape, 0, isolation_x, isolation + tape], [], None),
        ("Tape_Isolation_Horizontal", [isolation_x, isolation, width, isolation + tape],
         [isolation_name], [isolation_x - tape, isolation, width, isolation + tape]),
        (h_left, [left - half, low, left + half, high], [], None),
        (h_right, [right - half, low, right + half, high], [], None),
        ("Tape_Center_Crossbar", [left + half, center - half, right - half, center + half],
         [h_left, h_right], [left - half, center - half, right + half, center + half]),
    ]
    base_bounds = {name: _bounds([value * _MM for value in bounds])
                   for name, bounds, _, _ in definitions}
    strips = []
    for name, original, supports, extended in definitions:
        use_overlap = joins == "overlap" and bool(supports)
        bounds = _bounds([value * _MM for value in (extended if use_overlap else original)])
        if min(bounds[:2]) < -1e-12 or bounds[2] > width * _MM + 1e-12 or bounds[3] > height * _MM + 1e-12:
            raise ValueError(f"{name} extends outside the playing surface")
        regions = []
        for support in supports if use_overlap else []:
            target = base_bounds[support]
            intersection = _bounds([max(bounds[0], target[0]), max(bounds[1], target[1]),
                                    min(bounds[2], target[2]), min(bounds[3], target[3])])
            regions.append({"support_name": support, "bounds_m": intersection})
        strips.append({
            "name": name,
            "bounds_m": bounds,
            "layer": int(use_overlap),
            "support_names": list(supports) if use_overlap else [],
            "overlap_regions": regions,
            "floor_regions_m": [base_bounds[name]],
            "nominal_area_m2": (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]),
        })
    return strips


def rectangle_grid(
    bounds_m: Sequence[float], spacing_m: float = 0.01, radius_m: float = 0.000075,
) -> dict[str, Any]:
    """Triangulate an inset rectangle as one conforming, shared-vertex grid.

    The collision radius is removed from each planar edge before meshing, so
    adding a flex collision radius restores the requested outer AABB. Rounded
    collision corners remain rounded; they do not add material beyond it.
    Vertices are world XY at Z=0, ordered X-fast. Triangles face +Z. Every axis
    has at least three nodes, giving three across a 20 mm strip at 10 mm spacing.
    ``grid_shape`` is ``(ny, nx)`` and ``length_axis`` is 0 for X or 1 for Y.

    ``areas`` are nodal tributary *mesh* areas, one third of each incident
    triangle. Their sum equals the inset mesh area. Use ``nominal_area_m2`` if
    total physical strip mass must use its full roll width and cut length.
    """
    x0, y0, x1, y1 = _bounds(bounds_m)
    spacing = _positive(spacing_m, "spacing_m")
    radius = float(radius_m)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius_m must be finite and nonnegative")
    if 2 * radius >= min(x1 - x0, y1 - y0):
        raise ValueError("The collision radius consumes the rectangle width")
    xx0, yy0, xx1, yy1 = x0 + radius, y0 + radius, x1 - radius, y1 - radius
    nx = max(3, math.ceil((xx1 - xx0) / spacing) + 1)
    ny = max(3, math.ceil((yy1 - yy0) / spacing) + 1)
    xs, ys = np.linspace(xx0, xx1, nx), np.linspace(yy0, yy1, ny)
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(nx * ny)))
    triangles = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            a = row * nx + column
            b, c, d = a + 1, a + nx, a + nx + 1
            if (row + column) % 2:
                triangles.extend(((a, b, c), (b, d, c)))
            else:
                triangles.extend(((a, b, d), (a, d, c)))
    triangles = np.asarray(triangles, dtype=np.int32)
    edge_a = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edge_b = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    triangle_areas = np.cross(edge_a, edge_b)[:, 2] / 2
    areas = np.zeros(len(vertices))
    np.add.at(areas, triangles.ravel(), np.repeat(triangle_areas / 3, 3))
    # Counterclockwise perimeter, each boundary vertex exactly once.
    boundary = list(range(nx))
    boundary.extend(row * nx + nx - 1 for row in range(1, ny))
    boundary.extend((ny - 1) * nx + column for column in range(nx - 2, -1, -1))
    boundary.extend(row * nx for row in range(ny - 2, 0, -1))
    return {
        "vertices": vertices,
        "triangles": triangles,
        "areas": areas,
        "boundary_indices": np.asarray(boundary, dtype=np.int32),
        "length_axis": 0 if x1 - x0 >= y1 - y0 else 1,
        "grid_shape": (ny, nx),
        "bounds_m": [x0, y0, x1, y1],
        "centerline_bounds_m": [xx0, yy0, xx1, yy1],
        "collision_radius_m": radius,
        "nominal_area_m2": (x1 - x0) * (y1 - y0),
        "mesh_area_m2": float(triangle_areas.sum()),
    }


def _polygon_area(points: np.ndarray) -> float:
    following = np.roll(points, -1, axis=0)
    return float(np.sum(points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0]) / 2)


def _convex(points: np.ndarray) -> bool:
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    return bool(np.all(cross >= -1e-15) and _polygon_area(points) > 0)


def _prism(points: np.ndarray, thickness: float, center: Sequence[float]) -> tuple[np.ndarray, list[list[int]]]:
    count = len(points)
    bottom = np.column_stack((points + np.asarray(center), np.zeros(count)))
    top = bottom.copy()
    top[:, 2] = thickness
    faces = [list(reversed(range(count))), list(range(count, 2 * count))]
    faces.extend([index, (index + 1) % count, (index + 1) % count + count, index + count]
                 for index in range(count))
    return np.vstack((bottom, top)), faces


def lab_prisms(spec: Mapping[str, Any], circle_segments: int = 64) -> list[dict[str, Any]]:
    """Cover the laboratory plate with watertight convex collision prisms.

    Three rectangular cells partition the plate at the midpoints between hole
    centers. A ray fan in each cell connects its rectangle to an inscribed
    circular polygon. Corner angles are included, so no sector cuts across an
    outer corner. Each annular quadrilateral becomes its own convex prism;
    independent convex collision hulls therefore leave all three holes open.

    ``circle_segments`` is a requested minimum. It is increased if needed to
    keep the radial circle approximation error at or below 0.05 mm. Cardinal
    rays preserve the exact hole diameter bounds. Each record reports its
    conservative ``approximation_error_m``. Default dimensions produce 204
    quadrilateral prisms, with maximum radial error about 0.03614 mm.
    """
    if isinstance(circle_segments, bool) or int(circle_segments) != circle_segments or circle_segments < 8:
        raise ValueError("circle_segments must be an integer of at least eight")
    dim = _dimensions(spec)
    width = _positive(dim["lab_width"], "lab_width") * _MM
    length = _positive(dim["lab_length"], "lab_length") * _MM
    thickness = _positive(dim["lab_thickness"], "lab_thickness") * _MM
    radius = _positive(dim["lab_hole_diameter"], "lab_hole_diameter") * _MM / 2
    pitch = _positive(dim["lab_hole_pitch"], "lab_hole_pitch") * _MM
    center = np.asarray(spec["placements"]["lab_center"], dtype=float) * _MM
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("lab_center must contain two finite millimetre coordinates")
    if radius >= width / 2 or pitch <= 2 * radius or pitch + radius >= length / 2:
        raise ValueError("Laboratory holes must be separate and wholly inside the plate")
    limit = min(_MAX_HOLE_ERROR_M / radius, 1.0)
    required = math.ceil(math.pi / math.acos(1 - limit))
    segments = max(int(circle_segments), required, 8)
    partition = [-length / 2, -pitch / 2, pitch / 2, length / 2]
    records = []
    for cell, hole_y in enumerate((-pitch, 0.0, pitch)):
        xmin, xmax = -width / 2, width / 2
        ymin, ymax = partition[cell] - hole_y, partition[cell + 1] - hole_y
        angles = [2 * math.pi * index / segments for index in range(segments)]
        angles.extend(index * math.pi / 2 for index in range(4))
        angles.extend(math.atan2(y, x) % (2 * math.pi)
                      for x, y in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))
        angles.sort()
        unique = []
        for angle in angles:
            if not unique or angle - unique[-1] > 1e-12:
                unique.append(angle)
        angles = np.asarray(unique)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        inner = directions * radius
        outer = []
        for dx, dy in directions:
            tx = (xmax if dx > 0 else xmin) / dx if abs(dx) > 1e-14 else math.inf
            ty = (ymax if dy > 0 else ymin) / dy if abs(dy) > 1e-14 else math.inf
            point = np.array((dx, dy)) * min(tx, ty)
            point[0] = np.clip(point[0], xmin, xmax)
            point[1] = np.clip(point[1], ymin, ymax)
            outer.append(point)
        outer = np.asarray(outer)
        angle_steps = np.diff(np.append(angles, angles[0] + 2 * math.pi))
        approximation = radius * (1 - math.cos(float(angle_steps.max()) / 2))
        for sector in range(len(angles)):
            following = (sector + 1) % len(angles)
            quad = np.array((inner[sector], outer[sector], outer[following], inner[following]))
            polygons = [quad] if _convex(quad) else [quad[[0, 1, 2]], quad[[0, 2, 3]]]
            for part, polygon in enumerate(polygons):
                if not _convex(polygon):
                    raise ValueError("A laboratory sector could not be decomposed into convex prisms")
                vertices, faces = _prism(polygon, thickness, center + [0.0, hole_y])
                suffix = f"_Part_{part}" if len(polygons) > 1 else ""
                records.append({
                    "name": f"Laboratory_Cell_{cell + 1:02d}_Sector_{sector:03d}{suffix}",
                    "vertices": vertices,
                    "faces": faces,
                    "hole_index": cell,
                    "hole_center_m": (center + [0.0, hole_y]).tolist(),
                    "hole_radius_m": radius,
                    "circle_segments": len(angles),
                    "approximation_error_m": approximation,
                    "volume_m3": _polygon_area(polygon) * thickness,
                })
    return records
