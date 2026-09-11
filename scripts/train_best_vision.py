#!/usr/bin/env python3
"""Render frozen MuJoCo crops and train the senior preliminary vision model."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time


def _select_render_backend() -> None:
    if "--render-backend" not in sys.argv:
        return
    index = sys.argv.index("--render-backend")
    if index + 1 < len(sys.argv) and sys.argv[index + 1] != "auto":
        os.environ.setdefault("MUJOCO_GL", sys.argv[index + 1])


_select_render_backend()

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena_mujoco.best_vision import (
    BestVisionCNN,
    CalibratedVision,
    FrozenVisionDataset,
    VisionConfig,
    atomic_json,
    classification_metrics,
    generate_frozen_dataset,
    localization_metrics,
    preprocess_rgb,
    sha256_file,
    source_provenance,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=ROOT / "output/best_vision/dataset")
    result.add_argument("--output", type=Path, default=ROOT / "output/best_vision/model")
    result.add_argument("--train-count", type=int, default=24000)
    result.add_argument("--validation-count", type=int, default=3000)
    result.add_argument("--test-count", type=int, default=3000)
    result.add_argument("--seed", type=int, default=20260911)
    result.add_argument("--image-size", type=int, default=72)
    result.add_argument("--channels", type=int, default=32)
    result.add_argument("--settle-steps", type=int, default=8)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--cpu-smoke", action="store_true",
                        help="permit a bounded CPU development run")
    result.add_argument("--render-backend", choices=("auto", "egl", "osmesa"), default="auto")
    result.add_argument("--force-generate", action="store_true")
    result.add_argument("--generate-only", action="store_true")
    result.add_argument("--epochs", type=int, default=60)
    result.add_argument("--patience", type=int, default=8)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-3)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--compile", action="store_true",
                        help="use torch.compile on CUDA and record its startup cost")
    result.add_argument("--resume", action="store_true")
    return result


def validate_args(args: argparse.Namespace) -> None:
    counts = (args.train_count, args.validation_count, args.test_count)
    if min(counts) <= 0:
        raise ValueError("All split counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a full run. Use --device cpu --cpu-smoke for a small check.")
    if args.device == "cpu" and not args.cpu_smoke:
        raise RuntimeError("CPU is reserved for small smoke runs; add --cpu-smoke or use --device cuda")
    if args.cpu_smoke and (args.train_count > 512 or args.validation_count > 128 or args.test_count > 128):
        raise ValueError("A CPU smoke run is capped at 512 train and 128 validation/test images")
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch size must be positive")
    if args.compile and args.device != "cuda":
        raise ValueError("--compile is only supported for CUDA training")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def augment(images: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply label-preserving sensor transforms on the training device."""
    batch = len(images)
    horizontal = torch.rand(batch, device=images.device) < 0.5
    vertical = torch.rand(batch, device=images.device) < 0.5
    images[horizontal] = images[horizontal].flip(-1)
    images[vertical] = images[vertical].flip(-2)
    centers = centers.clone()
    centers[horizontal, 0] = 1 - centers[horizontal, 0]
    centers[vertical, 1] = 1 - centers[vertical, 1]
    gain = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.90, 1.10)
    bias = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(-0.10, 0.10)
    noise = torch.randn_like(images) * torch.empty(
        batch, 1, 1, 1, device=images.device).uniform_(0, 0.025)
    return (images * gain + bias + noise).clamp(-2.5, 2.5), centers


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device,
                        temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    logits_all, labels_all, centers_all, center_predictions = [], [], [], []
    losses = []
    with torch.inference_mode():
        for images, labels, centers in loader:
            images = preprocess_rgb(images.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            centers = centers.to(device, non_blocking=True)
            with autocast_context(device):
                logits, predicted_centers = model(images)
                class_loss = F.cross_entropy(logits, labels)
                foreground = labels != 0
                center_loss = (F.smooth_l1_loss(predicted_centers[foreground], centers[foreground])
                               if foreground.any() else logits.new_zeros(()))
                loss = class_loss + 6.0 * center_loss
            logits_all.append((logits.float() / temperature).cpu())
            labels_all.append(labels.cpu())
            centers_all.append(centers.cpu())
            center_predictions.append(predicted_centers.float().cpu())
            losses.append(float(loss))
    return (torch.cat(logits_all).numpy(), torch.cat(labels_all).numpy(),
            torch.cat(centers_all).numpy(), torch.cat(center_predictions).numpy(),
            float(np.mean(losses)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive temperature using validation NLL only."""
    logit_tensor = torch.as_tensor(logits, dtype=torch.float64)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.2, max_iter=80,
                                  tolerance_grad=1e-9, tolerance_change=1e-10)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logit_tensor / log_temperature.exp().clamp(0.05, 20), label_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 20).detach())


def write_history(path: Path, history: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in history))
    temporary.replace(path)


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    seed_everything(args.seed)
    overall_started = time.perf_counter()
    config = VisionConfig(image_size=args.image_size, channels=args.channels,
                          settle_steps=args.settle_steps)
    split_counts = {"train": args.train_count, "validation": args.validation_count,
                    "test": args.test_count}
    split_seeds = {"train": args.seed + 101, "validation": args.seed + 100_101,
                   "test": args.seed + 200_101}
    dataset_started = time.perf_counter()
    dataset_manifest = generate_frozen_dataset(
        args.dataset, config, split_counts, split_seeds, ROOT, force=args.force_generate)
    dataset_seconds = time.perf_counter() - dataset_started
    print(json.dumps({"phase": "dataset_ready", "seconds_this_run": dataset_seconds,
                      "dataset": str(args.dataset)}), flush=True)
    if args.generate_only:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    run_config = {
        "vision_config": asdict(config),
        "dataset": str(args.dataset.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset / "manifest.json"),
        "seed": args.seed,
        "device": args.device,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "workers": args.workers,
        "compile": args.compile,
    }
    atomic_json(args.output / "config.json", run_config)
    device = torch.device(args.device)
    train_dataset = FrozenVisionDataset(args.dataset, "train")
    validation_dataset = FrozenVisionDataset(args.dataset, "validation")
    generator = torch.Generator().manual_seed(args.seed + 7)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == "cuda")
    if args.workers:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator,
                              drop_last=len(train_dataset) >= args.batch_size, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = BestVisionCNN(config.channels).to(device)
    model = model.to(memory_format=torch.channels_last)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4,
        fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_loss, stale_epochs, best_epoch, history = 0, float("inf"), 0, 0, []
    last_path = args.output / "last.pt"
    best_path = args.output / "best.pt"
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"No resumable checkpoint at {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["run_config"] != run_config:
            raise ValueError("Resume configuration does not match the saved run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        stale_epochs = checkpoint["stale_epochs"]
        best_epoch = checkpoint["best_epoch"]
        history = checkpoint["history"]

    compile_seconds = 0.0
    training_model = model
    if args.compile:
        compile_started = time.perf_counter()
        training_model = torch.compile(model)
        sample = torch.zeros((args.batch_size, 3, config.image_size, config.image_size),
                             device=device).to(memory_format=torch.channels_last)
        with autocast_context(device):
            training_model(sample)
        device_sync(device)
        compile_seconds = time.perf_counter() - compile_started
    training_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        training_model.train()
        train_losses = []
        for images, labels, centers in train_loader:
            images = preprocess_rgb(images.to(device, non_blocking=True)).to(
                memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            centers = centers.to(device, non_blocking=True)
            images, centers = augment(images, centers)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits, predicted_centers = training_model(images)
                class_loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
                foreground = labels != 0
                center_loss = F.smooth_l1_loss(
                    predicted_centers[foreground], centers[foreground])
                loss = class_loss + 6.0 * center_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach()))
        scheduler.step()
        device_sync(device)
        validation_logits, validation_labels, validation_centers, validation_predictions, validation_loss = (
            collect_predictions(model, validation_loader, device))
        probabilities = torch.from_numpy(validation_logits).softmax(1).numpy()
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "validation_accuracy": float((probabilities.argmax(1) == validation_labels).mean()),
            "validation_center_mean_pixels": localization_metrics(
                validation_labels, validation_centers, validation_predictions,
                config.image_size).get("mean_pixels"),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        improved = validation_loss < best_loss - 1e-5
        if improved:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch + 1, 0
            torch.save({"model": model.state_dict(), "epoch": best_epoch,
                        "validation_loss": best_loss}, best_path)
        else:
            stale_epochs += 1
        checkpoint = {
            "run_config": run_config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch + 1,
            "best_loss": best_loss, "stale_epochs": stale_epochs,
            "best_epoch": best_epoch, "history": history,
        }
        torch.save(checkpoint, last_path)
        write_history(args.output / "history.jsonl", history)
        atomic_json(args.output / "progress.json", {
            "phase": "training", **record, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "stale_epochs": stale_epochs})
        print(json.dumps(record, allow_nan=False), flush=True)
        if stale_epochs >= args.patience:
            break
    training_seconds = time.perf_counter() - training_started

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    raw_logits, labels, centers, predicted_centers, selected_validation_loss = collect_predictions(
        model, validation_loader, device)
    temperature = fit_temperature(raw_logits, labels)
    calibrated_probabilities = torch.from_numpy(raw_logits / temperature).softmax(1).numpy()
    raw_probabilities = torch.from_numpy(raw_logits).softmax(1).numpy()
    calibrated = CalibratedVision(model.eval().cpu(), temperature).eval()
    scripted = torch.jit.freeze(torch.jit.script(calibrated))
    model_path = args.output / "model.ts"
    temporary_model = args.output / "model.ts.tmp"
    torch.jit.save(scripted, str(temporary_model))
    temporary_model.replace(model_path)

    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "cuda_runtime": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    training_report = {
        "schema_version": 1,
        "artifact": model_path.name,
        "artifact_sha256": sha256_file(model_path),
        "architecture": "depthwise compact CNN with class and 2D center heads",
        "parameters": parameter_count,
        "classes": list(dataset_manifest["classes"]),
        "vision_config": asdict(config),
        "seed": args.seed,
        "torch_version": torch.__version__,
        "mujoco_version": dataset_manifest["mujoco_version"],
        "device": str(device),
        "gpu": gpu,
        "precision": "CUDA FP16 autocast with FP32 parameters" if device.type == "cuda" else "FP32",
        "cuda_optimizations": {
            "channels_last": True,
            "fused_adamw": device.type == "cuda",
            "tf32_allowed": True,
            "torch_compile": args.compile,
            "compile_seconds": compile_seconds,
        },
        "timing": {
            "dataset_seconds_this_run": dataset_seconds,
            "training_seconds": training_seconds,
            "total_seconds": time.perf_counter() - overall_started,
        },
        "dataset_manifest_sha256": run_config["dataset_manifest_sha256"],
        "dataset_split_sha256": {
            split: {kind: record["sha256"] for kind, record in values["files"].items()}
            for split, values in dataset_manifest["splits"].items()
        },
        "source_sha256": source_provenance(ROOT),
        "selection": {
            "criterion": "minimum validation class-plus-center loss",
            "selected_epoch": best_epoch,
            "validation_loss": selected_validation_loss,
            "early_stopped": len(history) < args.epochs,
            "epochs_completed": len(history),
            "patience": args.patience,
        },
        "validation": {
            "uncalibrated": classification_metrics(labels, raw_probabilities),
            "temperature": temperature,
            "calibrated": classification_metrics(labels, calibrated_probabilities),
            "localization": localization_metrics(
                labels, centers, predicted_centers, config.image_size),
        },
        "test_evaluation": "Not run during training or model selection. Run scripts/test_best_vision.py once on the frozen test split.",
        "claim_scope": "These image metrics measure crop perception only. They do not establish robot task success.",
    }
    atomic_json(args.output / "training.json", training_report)
    atomic_json(args.output / "progress.json", {
        "phase": "complete", "artifact": str(model_path),
        "artifact_sha256": training_report["artifact_sha256"],
        "selected_epoch": best_epoch,
    })
    print("TRAINING_REPORT " + json.dumps(training_report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
