#!/usr/bin/env python3
"""Select and independently test RGB sample-rim refinement in native renders."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena_mujoco.best_vision import (  # noqa: E402
    BestVisionPredictor,
    CLASS_TO_INDEX,
    MujocoVisionGenerator,
    VisionConfig,
    atomic_json,
    sha256_file,
)
from arena_mujoco.best_vision_refiner import (  # noqa: E402
    BestSampleGeometryRefiner,
    PinholeCamera,
    Plane,
    RefinerThresholds,
    rejection_reason,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path,
                        default=ROOT / "output/best_vision/refined/model.ts")
    result.add_argument("--output", type=Path,
                        default=ROOT / "output/best_vision/sample_refiner")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--development-count", type=int, default=160)
    result.add_argument("--test-count", type=int, default=320)
    result.add_argument("--development-seed", type=int, default=20260941)
    result.add_argument("--test-seed", type=int, default=20264941)
    result.add_argument("--cpu-smoke", action="store_true")
    return result


def camera_from_generator(generator: MujocoVisionGenerator) -> PinholeCamera:
    size = generator.config.image_size
    fovy = float(generator.model.cam_fovy[generator.camera_id])
    focal = 0.5 * size / math.tan(math.radians(fovy) / 2)
    rotation = generator.data.cam_xmat[generator.camera_id].reshape(3, 3)
    position = generator.data.cam_xpos[generator.camera_id]
    return PinholeCamera(
        focal, focal, size / 2, size / 2, tuple(position), tuple(map(tuple, rotation)))


def render_records(count: int, seed: int, predictor: BestVisionPredictor,
                   keep_images: bool) -> tuple[list[dict], list[np.ndarray]]:
    config = VisionConfig(image_size=224, target_center_limit=0.08)
    permissive = RefinerThresholds(
        minimum_cnn_confidence=0.0, minimum_arc_coverage=0.0,
        maximum_radial_residual_px=100.0, maximum_radius_relative_error=10.0,
        maximum_coarse_offset_px=1000.0, minimum_edge_points=1)
    refiner = BestSampleGeometryRefiner(permissive)
    records, images = [], []
    rng = np.random.default_rng(seed)
    with MujocoVisionGenerator(config) as generator:
        sample_z = next(record["z"] for record in generator.object_records
                        if record["class"] == "sample")
        plane = Plane((0.0, 0.0, sample_z))
        for index in range(count):
            image, true_center_normalized = generator.render_sample(
                CLASS_TO_INDEX["sample"], rng)
            prediction = predictor(image)
            camera = camera_from_generator(generator)
            fit = refiner.refine(image, prediction, camera, plane)
            true_pixel = true_center_normalized.astype(np.float64) * config.image_size
            true_world = camera.intersect(true_pixel, plane)
            if fit.get("fit_found"):
                fit_world = np.asarray(fit["world_point_m"])
                error_mm = float(np.linalg.norm(fit_world[:2] - true_world[:2]) * 1000)
                pixel_error = float(np.linalg.norm(np.asarray(fit["center_crop_px"]) - true_pixel))
            else:
                error_mm = None
                pixel_error = None
            rotation = np.asarray(camera.rotation_camera_to_world)
            optical_forward = -rotation[:, 2]
            tilt = math.degrees(math.acos(np.clip(optical_forward @ (0.0, 0.0, -1.0), -1, 1)))
            occluders = sum(generator.data.qpos[address] > -1.0
                            for _, address in generator.occluders)
            fit.update({
                "index": index,
                "true_center_px": true_pixel.tolist(),
                "true_world_xy_m": true_world[:2].tolist(),
                "position_error_mm": error_mm,
                "pixel_error": pixel_error,
                "camera_height_m": float(camera.position_world[2] - sample_z),
                "camera_tilt_degrees": float(tilt),
                "occluder_count": int(occluders),
                "image_mean": float(np.mean(image) / 255.0),
            })
            records.append(fit)
            if keep_images:
                images.append(image)
    return records, images


def accepted_records(records: list[dict], thresholds: RefinerThresholds) -> list[dict]:
    accepted = []
    for record in records:
        reason = rejection_reason(record, thresholds)
        if reason is None:
            accepted.append(record)
    return accepted


def error_metrics(records: list[dict]) -> dict:
    values = np.asarray([record["position_error_mm"] for record in records], dtype=np.float64)
    if not len(values):
        return {"count": 0, "median_mm": None, "p95_mm": None, "maximum_mm": None}
    return {
        "count": int(len(values)),
        "median_mm": float(np.median(values)),
        "p95_mm": float(np.quantile(values, 0.95)),
        "maximum_mm": float(values.max()),
    }


def select_thresholds(records: list[dict]) -> tuple[RefinerThresholds, list[dict]]:
    candidates = []
    minimum_count = max(12, math.ceil(len(records) * 0.85))
    for arc in (0.30, 0.40, 0.48, 0.58, 0.68):
        for residual in (0.65, 0.90, 1.15, 1.45):
            # A calibrated 56 mm circle viewed at <=8 degrees should not need
            # more than 8% radius slack. Larger fits are usually a partially
            # visible yellow distractor or an occlusion chord.
            for radius_error in (0.04, 0.06, 0.08):
                for coarse_offset in (10.0, 14.0, 18.0):
                    thresholds = RefinerThresholds(
                        minimum_arc_coverage=arc,
                        maximum_radial_residual_px=residual,
                        maximum_radius_relative_error=radius_error,
                        maximum_coarse_offset_px=coarse_offset)
                    accepted = accepted_records(records, thresholds)
                    metrics = error_metrics(accepted)
                    coverage = len(accepted) / len(records)
                    feasible = (len(accepted) >= minimum_count and
                                metrics["p95_mm"] <= 0.75 and metrics["maximum_mm"] <= 1.5)
                    candidates.append({
                        "thresholds": asdict(thresholds),
                        "coverage": coverage,
                        "errors": metrics,
                        "meets_development_target": feasible,
                    })
    feasible = [candidate for candidate in candidates if candidate["meets_development_target"]]
    if feasible:
        selected = max(feasible, key=lambda item: (
            item["coverage"], -item["errors"]["p95_mm"], -item["errors"]["maximum_mm"]))
    else:
        eligible = [candidate for candidate in candidates
                    if candidate["errors"]["count"] >= minimum_count]
        selected = min(eligible, key=lambda item: (
            item["errors"]["p95_mm"], item["errors"]["maximum_mm"], -item["coverage"]))
    return RefinerThresholds(**selected["thresholds"]), candidates


def group_metrics(records: list[dict], thresholds: RefinerThresholds,
                  key, names: list[str]) -> dict:
    result = {}
    for index, name in enumerate(names):
        group = [record for record in records if key(record) == index]
        accepted = accepted_records(group, thresholds)
        result[name] = {
            "count": len(group),
            "accepted": len(accepted),
            "coverage": len(accepted) / len(group) if group else None,
            "accepted_errors": error_metrics(accepted),
        }
    return result


def summarize(records: list[dict], thresholds: RefinerThresholds) -> dict:
    accepted = accepted_records(records, thresholds)
    reasons = {}
    for record in records:
        reason = rejection_reason(record, thresholds)
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
    sample_proposals = [record for record in records
                        if record["class_name"] == "sample" and
                        record["cnn_confidence"] >= thresholds.minimum_cnn_confidence]
    return {
        "examples": len(records),
        "accepted": len(accepted),
        "coverage": len(accepted) / len(records),
        "cnn_sample_proposal_coverage": len(sample_proposals) / len(records),
        "geometry_coverage_given_cnn_proposal": (
            len(accepted) / len(sample_proposals) if sample_proposals else 0.0),
        "accepted_position_error": error_metrics(accepted),
        "rejections": reasons,
        "by_camera_height": group_metrics(
            records, thresholds,
            lambda record: 0 if record["camera_height_m"] < 0.127 else
            (1 if record["camera_height_m"] < 0.154 else 2),
            ["100_to_127_mm", "127_to_154_mm", "154_to_180_mm"]),
        "by_tilt": group_metrics(
            records, thresholds,
            lambda record: 0 if record["camera_tilt_degrees"] < 2.67 else
            (1 if record["camera_tilt_degrees"] < 5.34 else 2),
            ["0_to_2.67_deg", "2.67_to_5.34_deg", "5.34_to_8_deg"]),
        "by_occluder_count": group_metrics(
            records, thresholds, lambda record: min(record["occluder_count"], 2),
            ["zero", "one", "two"]),
    }


def contact_sheet(records: list[dict], images: list[np.ndarray],
                  thresholds: RefinerThresholds, output: Path) -> None:
    accepted = [record for record in records if rejection_reason(record, thresholds) is None]
    worst = sorted(accepted, key=lambda record: record["position_error_mm"], reverse=True)[:6]
    rejected = [record for record in records if rejection_reason(record, thresholds) is not None][:6]
    selected = worst + rejected
    if not selected:
        return
    tile = 224
    canvas = Image.new("RGB", (6 * tile, 2 * (tile + 26)), "white")
    draw = ImageDraw.Draw(canvas)
    for slot, record in enumerate(selected[:12]):
        row, column = divmod(slot, 6)
        x, y = column * tile, row * (tile + 26)
        canvas.paste(Image.fromarray(images[record["index"]]), (x, y))
        true_x, true_y = record["true_center_px"]
        draw.ellipse((x + true_x - 4, y + true_y - 4,
                      x + true_x + 4, y + true_y + 4), outline="lime", width=2)
        if record.get("fit_found"):
            fit_x, fit_y = record["center_crop_px"]
            radius = record["radius_px"]
            draw.ellipse((x + fit_x - radius, y + fit_y - radius,
                          x + fit_x + radius, y + fit_y + radius), outline="red", width=2)
        reason = rejection_reason(record, thresholds)
        label = (f"{record['position_error_mm']:.2f} mm" if reason is None
                 else f"reject: {reason}")
        draw.text((x + 4, y + tile + 5), label, fill="black")
    canvas.save(output)


def main() -> None:
    args = parser().parse_args()
    if args.development_count <= 0 or args.test_count <= 0:
        raise ValueError("development and test counts must be positive")
    if args.development_seed == args.test_seed:
        raise ValueError("development and test seeds must differ")
    if args.cpu_smoke:
        args.development_count = min(args.development_count, 18)
        args.test_count = min(args.test_count, 36)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    predictor = BestVisionPredictor(args.model, args.device)
    started = time.perf_counter()

    development, _ = render_records(
        args.development_count, args.development_seed, predictor, keep_images=False)
    thresholds, candidates = select_thresholds(development)
    development_summary = summarize(development, thresholds)
    selection = {
        "role": "development threshold selection; final test was not rendered yet",
        "seed": args.development_seed,
        "examples": len(development),
        "selected_thresholds": asdict(thresholds),
        "selected_metrics": development_summary,
        "candidate_count": len(candidates),
        "development_target": {"p95_mm": 0.75, "maximum_mm": 1.5,
                               "minimum_coverage": 0.85},
    }
    atomic_json(args.output / "selection.json", selection)
    print(json.dumps({"phase": "threshold_selection", **selection}, allow_nan=False), flush=True)

    test, test_images = render_records(
        args.test_count, args.test_seed, predictor, keep_images=True)
    test_summary = summarize(test, thresholds)
    contact_sheet(test, test_images, thresholds, args.output / "test_examples.png")
    report = {
        "schema_version": 1,
        "evaluation_role": "independent final test after development threshold selection",
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "source_sha256": {
            "refiner": sha256_file(ROOT / "arena_mujoco/best_vision_refiner.py"),
            "test_script": sha256_file(Path(__file__)),
        },
        "development_seed": args.development_seed,
        "test_seed": args.test_seed,
        "seeds_disjoint": args.development_seed != args.test_seed,
        "camera_contract": {
            "resolution_px": [224, 224],
            "height_above_sample_center_m": [0.10, 0.18],
            "vertical_fovy_degrees": [48.0, 68.0],
            "tilt_degrees": [0.0, 8.0],
            "target_center_limit_normalized": 0.08,
            "sample_diameter_m": 0.056,
            "inference_inputs": "RGB crop, CNN result, crop origin, intrinsics, extrinsics, and docking plane",
        },
        "selected_thresholds": asdict(thresholds),
        "development": development_summary,
        "final_test": test_summary,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "artifacts": {"selection": "selection.json", "examples": "test_examples.png"},
        "claim_scope": "RGB crop precision under the stated calibrated camera contract; docking control and mission success are separate tests.",
    }
    atomic_json(args.output / "test.json", report)
    print("SAMPLE_REFINER_TEST " + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
