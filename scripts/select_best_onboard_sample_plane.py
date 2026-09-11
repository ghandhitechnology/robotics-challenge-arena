#!/usr/bin/env python3
"""Compare physical rim-plane assumptions on development-only onboard RGB views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena_mujoco.best_onboard_vision import (  # noqa: E402
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
                        default=ROOT / "output/best_vision/onboard_plane_selection.json")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--trials-per-mode", type=int, default=32)
    result.add_argument("--seed", type=int, default=20266941)
    return result


def metrics(records: list[dict]) -> dict:
    accepted = [record for record in records if record["accepted"]]
    errors = np.asarray([record["camera_lateral_error_mm"] for record in accepted])
    return {
        "trials": len(records),
        "accepted": len(accepted),
        "coverage": len(accepted) / len(records),
        "accepted_camera_lateral_error_mm": {
            "median": float(np.median(errors)) if len(errors) else None,
            "p95": float(np.quantile(errors, 0.95)) if len(errors) else None,
            "maximum": float(errors.max()) if len(errors) else None,
        },
    }


def main() -> None:
    args = parser().parse_args()
    if args.trials_per_mode <= 0:
        raise ValueError("trials per mode must be positive")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = BestVisionPredictor(args.model, args.device)
    refiner = BestSampleGeometryRefiner(ONBOARD_REFINER_THRESHOLDS)
    simulation = OnboardLabSimulation(seed=args.seed)
    renderer = mujoco.Renderer(simulation.model, height=224, width=224)
    original_rgba = simulation.model.geom_rgba.copy()
    rng = np.random.default_rng(args.seed)
    candidates = {
        "floor_center_2.52mm": [],
        "floor_visible_top_5.02mm": [],
        "held_gripper_site_100mm": [],
        "held_visible_top_99.1mm": [],
    }
    try:
        for mode in ("floor", "held"):
            for _ in range(args.trials_per_mode):
                setup = reset_coupon(simulation, rng, mode)
                randomize_render(simulation, rng, original_rgba)
                image = render_rgb(renderer, simulation, rng)
                coarse = predictor(image)
                camera = simulation.camera_calibration(224)
                if mode == "floor":
                    planes = {
                        "floor_center_2.52mm": Plane((0.0, 0.0, 0.00252)),
                        "floor_visible_top_5.02mm": Plane((0.0, 0.0, 0.00502)),
                    }
                else:
                    planes = {
                        "held_gripper_site_100mm": simulation.carriage_plane(0.1000),
                        "held_visible_top_99.1mm": simulation.held_sample_top_plane(),
                    }
                target = simulation.data.body(setup["target_name"])
                target_top = target.xpos + target.xmat.reshape(3, 3) @ np.array((0.0, 0.0, 0.0025))
                camera_position, camera_rotation = camera.arrays()
                target_camera = camera_rotation.T @ (target_top - camera_position)
                for name, plane in planes.items():
                    result = refiner.refine(image, coarse, camera, plane)
                    inferred = np.asarray(result.get("camera_lateral_xy_m", (np.nan, np.nan)))
                    error = float(np.linalg.norm(inferred - target_camera[:2]) * 1000)
                    candidates[name].append({
                        "accepted": bool(result["accepted"]),
                        "camera_lateral_error_mm": error,
                        "rejection_reason": result["rejection_reason"],
                        "extension_m": setup["extension_m"],
                    })
    finally:
        renderer.close()
    summaries = {name: metrics(records) for name, records in candidates.items()}
    chosen = {}
    for mode, names in {
        "floor": ("floor_center_2.52mm", "floor_visible_top_5.02mm"),
        "held": ("held_gripper_site_100mm", "held_visible_top_99.1mm"),
    }.items():
        eligible = [name for name in names if summaries[name]["coverage"] >= 0.80]
        chosen[mode] = min(eligible, key=lambda name: (
            summaries[name]["accepted_camera_lateral_error_mm"]["p95"],
            summaries[name]["accepted_camera_lateral_error_mm"]["maximum"]))
    report = {
        "schema_version": 1,
        "evaluation_role": "development-only plane selection before new onboard final test",
        "seed": args.seed,
        "trials_per_mode": args.trials_per_mode,
        "model_sha256": sha256_file(args.model),
        "candidates": summaries,
        "selected": chosen,
        "selection_rule": "lowest accepted camera-lateral p95 among candidates with at least 80% coverage",
        "target_pose_use": "evaluator only after RGB inference",
        "next_test_seed_must_differ": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print("ONBOARD_PLANE_SELECTION " + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
