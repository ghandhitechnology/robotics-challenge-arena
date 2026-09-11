#!/usr/bin/env python3
"""Validate the candidate carriage camera with floor, held, and negative RGB views."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

from arena_mujoco.best_onboard_vision import (  # noqa: E402
    CAMERA_BACKPLATE_NAME,
    CAMERA_FOVY_DEGREES,
    CAMERA_FRAME_NAMES,
    CAMERA_LOCAL_POSITION_M,
    CAMERA_NAME,
    ONBOARD_REFINER_THRESHOLDS,
    OnboardLabSimulation,
    randomize_render,
    render_rgb,
    reset_coupon,
)
from arena_mujoco.best_vision import BestVisionPredictor, atomic_json, sha256_file  # noqa: E402
from arena_mujoco.best_vision_refiner import BestSampleGeometryRefiner, Plane  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path,
                        default=ROOT / "output/best_vision/refined/model.ts")
    result.add_argument("--output", type=Path,
                        default=ROOT / "output/best_vision/onboard_sample_camera")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--floor-trials", type=int, default=48)
    result.add_argument("--held-trials", type=int, default=48)
    result.add_argument("--negative-trials", type=int, default=36)
    result.add_argument("--seed", type=int, default=20267941)
    result.add_argument("--cpu-smoke", action="store_true")
    return result


def percentile_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95)), "maximum": float(array.max())}


def mode_summary(records: list[dict], mode: str) -> dict:
    selected = [record for record in records if record["mode"] == mode]
    physical = [record for record in selected if record["physical_setup_valid"]]
    accepted = [record for record in physical if record["accepted"]]
    reasons = {}
    for record in physical:
        if not record["accepted"]:
            reason = record["rejection_reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
    result = {
        "trials": len(selected),
        "physical_setup_valid": len(physical),
        "accepted": len(accepted),
        "coverage_of_valid_setups": len(accepted) / len(physical) if physical else 0.0,
        "rejections": reasons,
        "maximum_robot_tilt_degrees": max(
            (record["robot_tilt_degrees"] for record in selected), default=0.0),
    }
    if mode != "yellow_negative":
        result["accepted_camera_lateral_error_mm"] = percentile_summary(
            [record["camera_lateral_error_mm"] for record in accepted])
        result["accepted_world_horizontal_error_mm"] = percentile_summary(
            [record["world_horizontal_error_mm"] for record in accepted])
        result["cnn_sample_proposal_coverage"] = sum(
            record["class_name"] == "sample" and record["cnn_confidence"] >= 0.995
            for record in physical) / len(physical) if physical else 0.0
    else:
        result["false_accept_rate"] = len(accepted) / len(physical) if physical else 0.0
    if mode == "held":
        result["physical_grasp_rate"] = len(physical) / len(selected) if selected else 0.0
        result["camera_to_sample_distance_mm"] = percentile_summary(
            [record["camera_to_target_distance_mm"] for record in physical])
    return result


def draw_sheet(records: list[dict], output: Path) -> None:
    selected = []
    for mode in ("floor", "held", "yellow_negative"):
        group = [record for record in records if record["mode"] == mode]
        if mode != "yellow_negative":
            group.sort(key=lambda record: record.get("camera_lateral_error_mm") or -1, reverse=True)
        selected.extend(group[:6])
    tile, label_height = 224, 28
    canvas = Image.new("RGB", (6 * tile, 3 * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(selected):
        row, column = divmod(index, 6)
        x, y = column * tile, row * (tile + label_height)
        canvas.paste(Image.open(record["image_path"]).convert("RGB"), (x, y))
        if record.get("center_crop_px"):
            cx, cy = record["center_crop_px"]
            radius = record["radius_px"]
            draw.ellipse((x + cx - radius, y + cy - radius,
                          x + cx + radius, y + cy + radius), outline="red", width=2)
        if record["mode"] == "yellow_negative":
            label = f"negative / {record['rejection_reason']}"
        elif not record["physical_setup_valid"]:
            label = "physical grasp failed"
        elif record["accepted"]:
            label = f"{record['mode']} / {record['camera_lateral_error_mm']:.2f} mm"
        else:
            label = f"{record['mode']} / reject {record['rejection_reason']}"
        draw.text((x + 4, y + tile + 6), label, fill="black")
    canvas.save(output)


def main() -> None:
    args = parser().parse_args()
    if min(args.floor_trials, args.held_trials, args.negative_trials) <= 0:
        raise ValueError("all trial counts must be positive")
    if args.cpu_smoke:
        args.floor_trials = min(args.floor_trials, 4)
        args.held_trials = min(args.held_trials, 4)
        args.negative_trials = min(args.negative_trials, 3)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(exist_ok=True)
    predictor = BestVisionPredictor(args.model, args.device)
    refiner = BestSampleGeometryRefiner(ONBOARD_REFINER_THRESHOLDS)
    rng = np.random.default_rng(args.seed)
    simulation = OnboardLabSimulation(seed=args.seed)
    renderer = mujoco.Renderer(simulation.model, height=224, width=224)
    original_rgba = simulation.model.geom_rgba.copy()
    records = []
    start = time.perf_counter()
    try:
        modes = (["floor"] * args.floor_trials + ["held"] * args.held_trials +
                 ["yellow_negative"] * args.negative_trials)
        rng.shuffle(modes)
        for index, mode in enumerate(modes):
            setup = reset_coupon(simulation, rng, mode)
            randomize_render(simulation, rng, original_rgba)
            image = render_rgb(renderer, simulation, rng)
            image_path = frame_dir / f"{index:04d}_{mode}.png"
            Image.fromarray(image).save(image_path)
            prediction = predictor(image)
            if mode == "held":
                plane = simulation.held_sample_top_plane()
            else:
                plane = Plane((0.0, 0.0, 0.00502))
            result = refiner.refine(
                image, prediction, simulation.camera_calibration(224), plane)
            target = simulation.data.body(setup["target_name"])
            camera_position = simulation.data.cam_xpos[simulation.camera_id]
            physical_valid = bool(mode != "held" or target.xpos[2] > 0.012)
            target_top = target.xpos + target.xmat.reshape(3, 3) @ np.array((0.0, 0.0, 0.0025))
            camera = simulation.camera_calibration(224)
            camera_frame_position, camera_rotation = camera.arrays()
            target_camera = camera_rotation.T @ (target_top - camera_frame_position)
            if mode != "yellow_negative" and result["accepted"] and physical_valid:
                world_error_mm = float(np.linalg.norm(
                    np.asarray(result["world_point_m"])[:2] - target_top[:2]) * 1000)
                camera_error_mm = float(np.linalg.norm(
                    np.asarray(result["camera_lateral_xy_m"]) - target_camera[:2]) * 1000)
            else:
                world_error_mm = None
                camera_error_mm = None
            robot = simulation.data.body(simulation.robots["lab"]["name"])
            tilt = math.degrees(math.acos(np.clip(robot.xmat[8], -1, 1)))
            result.update(setup)
            result.update({
                "image_path": str(image_path.resolve()),
                "physical_setup_valid": physical_valid,
                "world_horizontal_error_mm": world_error_mm,
                "camera_lateral_error_mm": camera_error_mm,
                "target_position_m": target.xpos.tolist(),
                "camera_to_target_distance_mm": float(
                    np.linalg.norm(camera_position - target.xpos) * 1000),
                "camera_to_gripper_plane_mm": float(
                    np.linalg.norm(camera_position - simulation.tool("lab")) * 1000),
                "robot_tilt_degrees": float(tilt),
            })
            records.append(result)
            if (index + 1) % 20 == 0:
                print(json.dumps({"phase": "onboard_coupon", "complete": index + 1,
                                  "total": len(modes)}), flush=True)
    finally:
        renderer.close()
    draw_sheet(records, args.output / "onboard_examples.png")
    report = {
        "schema_version": 1,
        "evaluation_role": "native onboard-camera candidate coupon",
        "model_sha256": sha256_file(args.model),
        "source_sha256": {
            "candidate": sha256_file(ROOT / "arena_mujoco/best_onboard_vision.py"),
            "test": sha256_file(Path(__file__)),
            "refiner": sha256_file(ROOT / "arena_mujoco/best_vision_refiner.py"),
            "baseline_fleet": sha256_file(ROOT / "arena_mujoco/best_fleet.py"),
            "design": sha256_file(ROOT / "best_design.json"),
            "plane_selection": sha256_file(
                ROOT / "output/best_vision/onboard_plane_selection.json"),
        },
        "seed": args.seed,
        "camera": {
            "name": CAMERA_NAME,
            "parent_body": "fleet_lab_comp_lift_body",
            "local_position_m": CAMERA_LOCAL_POSITION_M,
            "local_optical_axis": "-Z",
            "vertical_fovy_degrees": CAMERA_FOVY_DEGREES,
            "resolution_px": [224, 224],
            "camera_module_mass_kg": 0.006,
            "initial_optical_center_height_m": 0.1041,
            "start_height_limit_m": 0.110,
            "compiled_camera_count": sum(
                simulation.model.camera(index).name == CAMERA_NAME
                for index in range(simulation.model.ncam)),
            "compiled_frame_geom_count": sum(
                simulation.model.geom(index).name in CAMERA_FRAME_NAMES
                for index in range(simulation.model.ngeom)),
            "compiled_backplate_geom_count": sum(
                simulation.model.geom(index).name == CAMERA_BACKPLATE_NAME
                for index in range(simulation.model.ngeom)),
            "clear_aperture_m": [0.012, 0.012],
            "optical_half_cone_at_frame_bottom_mm": 3.94,
            "aperture_half_width_mm": 6.0,
            "integrated_lab_mass_kg": simulation.robots["lab"]["mass_kg"],
        },
        "inference_contract": {
            "inputs": "onboard RGB, frozen CNN output, camera calibration, and kinematic visible-rim plane",
            "forbidden_inputs": "segmentation IDs, depth, and target pose",
            "target_pose_use": "evaluation error only after inference",
            "control_output": "camera_lateral_xy_m; world pose is unnecessary for local alignment",
        },
        "visible_rim_planes_selected_on_development_seed_20266941": {
            "floor": "horizontal z=0.00502 m sample top face",
            "held": "camera-local plane 0.0991 m along optical -Z, 0.9 mm above gripper site",
            "selection_report": "../onboard_plane_selection.json",
        },
        "thresholds_from_separate_refiner_development_split": asdict(ONBOARD_REFINER_THRESHOLDS),
        "floor": mode_summary(records, "floor"),
        "held": mode_summary(records, "held"),
        "yellow_negative": mode_summary(records, "yellow_negative"),
        "camera_to_gripper_plane_mm": percentile_summary(
            [record["camera_to_gripper_plane_mm"] for record in records]),
        "runtime": {"seconds": time.perf_counter() - start, "device": args.device,
                    "mujoco": mujoco.__version__, "torch": torch.__version__,
                    "python": platform.python_version()},
        "artifacts": {"records": "records.json", "examples": "onboard_examples.png",
                      "frames": "frames/"},
        "claim_scope": "Candidate camera visibility and RGB localization; packaging, docking motion, and seating remain separate.",
    }
    atomic_json(args.output / "records.json", {"records": records})
    atomic_json(args.output / "test.json", report)
    print("ONBOARD_CAMERA_REPORT " + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
