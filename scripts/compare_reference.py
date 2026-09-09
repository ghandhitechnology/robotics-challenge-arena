"""Compare the source photograph with the orthographic Blender render.

Run with .venv/bin/python scripts/compare_reference.py. Resampling is confined to
this comparison artifact. Original images and model geometry remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
# Clockwise from top left, at the inner timber edge. Pixel coordinates are in
# the original 408 x 406 attachment; the timber itself is excluded.
SOURCE_CORNERS = np.array([(27.0, 37.0), (370.0, 38.0),
                           (371.0, 369.0), (26.0, 369.0)])
# Manual search-window centers, ordered from the top of the source photograph.
SOURCE_MARKERS = [
    ("yellow", 156, 79, 450, 1031), ("yellow", 217, 79, 650, 1031),
    ("green", 157, 109, 450, 931), ("green", 217, 109, 650, 931),
    ("red", 157, 138, 450, 831), ("red", 217, 138, 650, 831),
    ("red", 157, 268, 450, 350), ("red", 217, 268, 650, 350),
    ("green", 157, 298, 450, 250), ("green", 217, 298, 650, 250),
    ("yellow", 156, 326, 450, 150), ("yellow", 217, 326, 650, 150),
]


def homography(source, destination):
    rows, rhs = [], []
    for (x, y), (u, v) in zip(source, destination):
        rows.extend([(x, y, 1, 0, 0, 0, -u*x, -u*y),
                     (0, 0, 0, x, y, 1, -v*x, -v*y)])
        rhs.extend((u, v))
    return np.append(np.linalg.solve(np.array(rows), np.array(rhs)), 1).reshape(3, 3)


def transform(matrix, xy):
    result = matrix @ np.array([*xy, 1.0])
    return (result[:2] / result[2]).tolist()


def warp(image, corners, size):
    width, height = size
    destination = np.array([(0, 0), (width, 0), (width, height), (0, height)])
    inverse = homography(destination, corners).flatten()[:8]
    return image.transform(size, Image.Transform.PERSPECTIVE, inverse,
                           resample=Image.Resampling.BICUBIC)


def marker_centroid(pixels, color, anchor):
    x, y = anchor
    x0, y0, x1, y1 = x - 7, y - 7, x + 8, y + 8
    crop = pixels[y0:y1, x0:x1].astype(float)
    red, green, blue = crop.transpose(2, 0, 1)
    if color == "red":
        weight = red - np.maximum(green, blue)
    elif color == "green":
        weight = green - np.maximum(red, blue)
    else:
        weight = np.minimum(red, green) - blue
    threshold = max(5.0, float(weight.max()) * 0.35)
    weight = np.where(weight >= threshold, weight, 0)
    if weight.sum() <= 0:
        raise ValueError(f"No {color} pixels around source anchor {anchor}")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    center = [float((xx * weight).sum() / weight.sum()),
              float((yy * weight).sum() / weight.sum())]
    return center, [x0, y0, x1, y1], int(np.count_nonzero(weight))


def font(size):
    for path in ["/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def main():
    spec = json.loads((ROOT / "arena_spec.json").read_text())
    width_mm, height_mm = [spec["dimensions"][key] for key in ("surface_x", "surface_y")]
    source = Image.open(ROOT / "reference/block_layout.png").convert("RGB")
    rendered = Image.open(OUT / "arena_top.png").convert("RGB")
    pixels = np.asarray(rendered)
    # Scan clear paper rows/columns, away from the black tape. This excludes the
    # studio background and the support slab side from the orthographic crop.
    brightness = pixels.mean(axis=2)
    threshold = float(brightness[0, 0] + 40)
    horizontal = np.flatnonzero(brightness[int(rendered.height * 0.20)] > threshold)
    vertical = np.flatnonzero(brightness[:, int(rendered.width * 0.13)] > threshold)
    bounds = [int(horizontal.min()), int(vertical.min()),
              int(horizontal.max() + 1), int(vertical.max() + 1)]
    x0, y0, x1, y1 = bounds
    render_corners = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    board_size = (750, round(750 * height_mm / width_mm))
    comparison = Image.new("RGB", (1600, board_size[1] + 96), "white")
    draw = ImageDraw.Draw(comparison)
    draw.text((28, 23), "Reference", font=font(29), fill="#252525")
    draw.text((822, 23), "Blender", font=font(29), fill="#252525")
    comparison.paste(warp(source, SOURCE_CORNERS, board_size), (28, 70))
    comparison.paste(warp(rendered, render_corners, board_size), (822, 70))
    comparison.save(OUT / "reference_comparison.png")

    source_to_world = homography(SOURCE_CORNERS, np.array([
        (0, height_mm), (width_mm, height_mm), (width_mm, 0), (0, 0)]))
    source_pixels = np.asarray(source)
    markers = []
    valid_centers = {(float(x), float(row["y"]), row["color"])
                     for row in spec["placements"]["cylinder_rows"]
                     for x in spec["placements"]["cylinder_x"]}
    for color, px, py, mx, my in SOURCE_MARKERS:
        if (float(mx), float(my), color) not in valid_centers:
            raise ValueError("Source/model marker correspondence differs from arena_spec.json")
        centroid, window, count = marker_centroid(source_pixels, color, (px, py))
        measured = transform(source_to_world, centroid)
        delta = np.array(measured) - [mx, my]
        markers.append({
            "color": color, "manual_search_anchor_px": [px, py],
            "search_window_px": window, "color_weighted_centroid_px": centroid,
            "selected_chromatic_pixels": count, "source_derived_xy_mm": measured,
            "model_xy_mm": [mx, my], "source_minus_model_xy_mm": delta.tolist(),
            "distance_mm": float(np.linalg.norm(delta)),
        })
    distances = np.array([item["distance_mm"] for item in markers])
    report = {
        "source_image": "reference/block_layout.png", "source_size_px": list(source.size),
        "render_image": "output/arena_top.png", "render_size_px": list(rendered.size),
        "source_inner_corners_px_tl_tr_br_bl": SOURCE_CORNERS.tolist(),
        "source_corner_basis": "Manual inner timber-edge observations in the original attachment; edges are blurred.",
        "render_inner_corners_px_tl_tr_br_bl": render_corners.tolist(),
        "physical_extent_mm": [width_mm, height_mm],
        "coordinate_system": "Origin lower left, X right and Y up; source raster Y points down.",
        "normalization": "Independent projective image registration to the chosen physical aspect ratio, only for this comparison; original images and Blender geometry are unchanged.",
        "centroid_method": "Positive color-channel excess in manually anchored 15 x 15 windows, weighted above max(5, 35% of local peak). No snapping of measured centers.",
        "source_precision": "Approximate raster observations, not physical ground truth. Original pixels correspond to about 3.3-3.6 mm; blur, corner choice and object height limit calibration.",
        "illustrative_two_pixel_uncertainty_mm": 2 * max(width_mm / 343, height_mm / 332),
        "markers": markers,
        "marker_error_summary_mm": {"mean": float(distances.mean()),
                                    "rms": float(np.sqrt(np.mean(distances ** 2))),
                                    "maximum": float(distances.max())},
    }
    (OUT / "reference_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"comparison": str(OUT / "reference_comparison.png"),
                      "marker_error_summary_mm": report["marker_error_summary_mm"]}))


if __name__ == "__main__":
    main()
