#!/usr/bin/env python3
"""Run the one-time frozen test split and benchmark exported vision inference."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena_mujoco.best_vision import (
    BestVisionPredictor,
    FrozenVisionDataset,
    atomic_json,
    classification_metrics,
    localization_metrics,
    preprocess_rgb,
    sha256_file,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=ROOT / "output/best_vision/dataset")
    result.add_argument("--model", type=Path, default=ROOT / "output/best_vision/model")
    result.add_argument("--output", type=Path)
    result.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    result.add_argument("--batch-size", type=int, default=512)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--latency-iterations", type=int, default=200)
    result.add_argument("--min-accuracy", type=float)
    result.add_argument("--max-center-p95-pixels", type=float)
    return result


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values), quantile))


def benchmark(scripted, predictor, example: np.ndarray, image_size: int,
              device: torch.device, iterations: int) -> dict:
    raw = torch.as_tensor(example, device=device).unsqueeze(0)
    prepared = preprocess_rgb(raw)
    if prepared.shape[-2:] != (image_size, image_size):
        prepared = torch.nn.functional.interpolate(
            prepared, (image_size, image_size), mode="bilinear", align_corners=False)
    batch = prepared.expand(32, -1, -1, -1).contiguous()
    with torch.inference_mode():
        for _ in range(20):
            scripted(prepared)
        sync(device)
        model_times = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            scripted(prepared)
            sync(device)
            model_times.append((time.perf_counter_ns() - started) / 1e6)
        end_to_end_times = []
        for _ in range(min(iterations, 100)):
            started = time.perf_counter_ns()
            predictor(example)
            sync(device)
            end_to_end_times.append((time.perf_counter_ns() - started) / 1e6)
        batch_started = time.perf_counter()
        for _ in range(max(10, iterations // 10)):
            scripted(batch)
        sync(device)
        batch_count = max(10, iterations // 10) * len(batch)
        batch_seconds = time.perf_counter() - batch_started
    return {
        "iterations": iterations,
        "batch_1_model_milliseconds": {
            "median": percentile(model_times, 0.5),
            "p95": percentile(model_times, 0.95),
        },
        "batch_1_end_to_end_milliseconds": {
            "median": percentile(end_to_end_times, 0.5),
            "p95": percentile(end_to_end_times, 0.95),
        },
        "batch_32_images_per_second": batch_count / max(batch_seconds, 1e-9),
    }


def main() -> None:
    args = parser().parse_args()
    if args.batch_size <= 0 or args.latency_iterations <= 0:
        raise ValueError("batch size and latency iterations must be positive")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model_path = args.model / "model.ts" if args.model.is_dir() else args.model
    model_dir = model_path.parent
    training_path = model_dir / "training.json"
    if not training_path.is_file():
        raise FileNotFoundError(f"No training report beside {model_path}")
    training = json.loads(training_path.read_text())
    if sha256_file(model_path) != training["artifact_sha256"]:
        raise ValueError("The TorchScript file does not match its training report")
    dataset_manifest_path = args.dataset / "manifest.json"
    if sha256_file(dataset_manifest_path) != training["dataset_manifest_sha256"]:
        raise ValueError("The frozen dataset manifest does not match model selection")
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    test_record = dataset_manifest["splits"]["test"]
    for kind, record in test_record["files"].items():
        if sha256_file(args.dataset / record["file"]) != record["sha256"]:
            raise ValueError(f"Frozen test {kind} hash mismatch")

    test_dataset = FrozenVisionDataset(args.dataset, "test")
    options = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                   pin_memory=device.type == "cuda")
    if args.workers:
        options["persistent_workers"] = True
    loader = DataLoader(test_dataset, **options)
    scripted = torch.jit.load(str(model_path), map_location=device).eval()
    logits_all, labels_all, centers_all, predicted_centers_all = [], [], [], []
    evaluation_started = time.perf_counter()
    with torch.inference_mode():
        for images, labels, centers in loader:
            inputs = preprocess_rgb(images.to(device, non_blocking=True))
            logits, predicted_centers = scripted(inputs)
            logits_all.append(logits.float().cpu())
            labels_all.append(labels)
            centers_all.append(centers)
            predicted_centers_all.append(predicted_centers.float().cpu())
    sync(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()
    centers = torch.cat(centers_all).numpy()
    predicted_centers = torch.cat(predicted_centers_all).numpy()
    probabilities = logits.softmax(1).numpy()
    predictions = probabilities.argmax(1)
    image_size = int(training["vision_config"]["image_size"])
    foreground_correct = (labels != 0) & (predictions == labels)
    correct_localization = localization_metrics(
        np.where(foreground_correct, labels, 0), centers, predicted_centers, image_size)
    predictor = BestVisionPredictor(model_path, str(device))
    latency = benchmark(scripted, predictor, np.array(test_dataset.images[0], copy=True),
                        image_size, device, args.latency_iterations)
    result = {
        "schema_version": 1,
        "evaluation_role": "independent final evaluation after validation-based model selection",
        "model_sha256": training["artifact_sha256"],
        "dataset_manifest_sha256": training["dataset_manifest_sha256"],
        "test_file_sha256": {kind: record["sha256"] for kind, record in test_record["files"].items()},
        "test_seed": test_record["seed"],
        "test_examples": len(test_dataset),
        "classification": classification_metrics(labels, probabilities),
        "localization_all_foreground": localization_metrics(
            labels, centers, predicted_centers, image_size),
        "localization_correctly_classified_foreground": correct_localization,
        "timing": {
            "test_evaluation_seconds": evaluation_seconds,
            "test_images_per_second": len(test_dataset) / max(evaluation_seconds, 1e-9),
            "inference_latency": latency,
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "claim_scope": "This report scores frozen crop perception. Fleet physics and mission success need separate tests.",
    }
    failures = []
    if args.min_accuracy is not None and result["classification"]["accuracy"] < args.min_accuracy:
        failures.append(f"accuracy {result['classification']['accuracy']:.6f} < {args.min_accuracy:.6f}")
    p95 = result["localization_all_foreground"].get("p95_pixels", float("inf"))
    if args.max_center_p95_pixels is not None and p95 > args.max_center_p95_pixels:
        failures.append(f"center p95 {p95:.6f}px > {args.max_center_p95_pixels:.6f}px")
    result["requested_gates"] = {
        "minimum_accuracy": args.min_accuracy,
        "maximum_center_p95_pixels": args.max_center_p95_pixels,
        "passed": not failures,
        "failures": failures,
    }
    output = args.output or model_dir / "test.json"
    atomic_json(output, result)
    print("TEST_REPORT " + json.dumps(result, allow_nan=False), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
