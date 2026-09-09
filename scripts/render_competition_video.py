#!/usr/bin/env python3
"""Render recorded MuJoCo poses, then assemble the proof in HyperFrames.

No simulation steps or pose interpolation run during playback. Every displayed
pose, time, and phase comes from the saved trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCENE_WIDTH, SCENE_HEIGHT, HUD_HEIGHT = 1440, 960, 80


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_recording(directory: Path, speed: float, fps: int, max_seconds: float | None):
    with np.load(directory / "trajectory.npz", allow_pickle=False) as saved:
        times = np.array(saved["times"], dtype=np.float64)
        poses = np.array(saved["qpos"], dtype=np.float64)
        phases = np.array(saved["phases"], dtype=str)
        if len(times) and times[0] > 1e-8 and "initial_qpos" in saved:
            times = np.concatenate(([0.0], times))
            poses = np.vstack((saved["initial_qpos"], poses))
            phases = np.concatenate((["initial state"], phases))
    if times.ndim != 1 or not len(times) or poses.ndim != 2:
        raise ValueError("Trajectory must contain nonempty times and a 2D qpos array")
    if len(poses) != len(times) or phases.shape != times.shape:
        raise ValueError("Trajectory arrays must have matching frame counts")
    if not np.isfinite(times).all() or not np.isfinite(poses).all():
        raise ValueError("Trajectory contains nonfinite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Recorded simulation times must be strictly increasing")
    end = times[-1] if max_seconds is None else min(times[-1], times[0] + max_seconds)
    targets = np.arange(times[0], end + 1e-8, speed / fps)
    if end - targets[-1] > 1e-7:
        targets = np.append(targets, end)
    right = np.searchsorted(times, targets).clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    indices = np.where(abs(times[right] - targets) < abs(times[left] - targets), right, left)
    return times, poses, phases, indices, bool(end < times[-1] - 1e-7)


def render_native(args, inputs, report):
    times, poses, phases, indices, partial = inputs
    project = args.project
    assets = project / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.input / "scene.xml"))
    if poses.shape[1] != model.nq:
        raise ValueError(f"Recorded qpos has {poses.shape[1]} values; model requires {model.nq}")
    model.vis.global_.offwidth = SCENE_WIDTH
    model.vis.global_.offheight = SCENE_HEIGHT
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if camera_id < 0:
        raise ValueError(f"Saved model has no camera named {args.camera!r}")
    model.cam_fovy[camera_id] = args.camera_fovy
    font_path = assets / "fonts/Barlow-SemiBold.ttf"
    mono_path = assets / "fonts/IBMPlexMono-Regular.ttf"
    phase_font = ImageFont.truetype(str(font_path), 34)
    clock_font = ImageFont.truetype(str(mono_path), 30)
    source = assets / "trajectory.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{SCENE_WIDTH}x{SCENE_HEIGHT + HUD_HEIGHT}",
               "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
               "-crf", "19", "-g", str(args.fps), "-keyint_min", str(args.fps),
               "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(source)]
    started = time.perf_counter()
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    samples = {}
    sample_frames = {0, len(indices) // 2, len(indices) - 1}
    hold_frames = round(args.hold_final * args.fps)
    try:
        with mujoco.Renderer(model, height=SCENE_HEIGHT, width=SCENE_WIDTH) as renderer:
            for frame_number, recorded_index in enumerate(indices):
                data.qpos[:] = poses[recorded_index]
                data.time = times[recorded_index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=args.camera)
                frame = Image.new("RGB", (SCENE_WIDTH, SCENE_HEIGHT + HUD_HEIGHT), "#12181b")
                frame.paste(Image.fromarray(renderer.render()), (0, 0))
                draw = ImageDraw.Draw(frame)
                phase = str(phases[recorded_index]).replace("_", " ")
                phase = phase[:1].upper() + phase[1:]
                draw.text((24, SCENE_HEIGHT + 21), phase, font=phase_font, fill="#eef1e9")
                clock = f"SIM {times[recorded_index]:06.1f} s  |  {args.speed:g}x"
                draw.text((SCENE_WIDTH - 24, SCENE_HEIGHT + 25), clock, anchor="ra", font=clock_font, fill="#dfc779")
                if frame_number in sample_frames:
                    samples[frame_number] = frame.copy()
                process.stdin.write(frame.tobytes())
                if frame_number % 150 == 0:
                    print(f"Native frame {frame_number + 1}/{len(indices)}; simulation {times[recorded_index]:.1f} s", flush=True)
            for _ in range(hold_frames):
                process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg exited {return_code}")
    frame.save(assets / "native-poster.png")
    sheet = Image.new("RGB", (1920, 463), "#12181b")
    for column, (_, sampled_frame) in enumerate(sorted(samples.items())):
        sheet.paste(sampled_frame.resize((640, 462), Image.Resampling.LANCZOS), (column * 640, 0))
    sheet.save(project / "native-contact-sheet.png")
    manifest = {
        "scene_sha256": digest(args.input / "scene.xml"),
        "trajectory_sha256": digest(args.input / "trajectory.npz"),
        "report_sha256": digest(args.input / "report.json"),
        "source_video_sha256": digest(source),
        "mujoco_version": mujoco.__version__, "camera": args.camera, "camera_fovy": args.camera_fovy,
        "fps": args.fps, "playback_speed": args.speed, "pose_interpolation": False,
        "recorded_pose_count": len(times), "rendered_pose_frames": len(indices),
        "final_hold_frames": hold_frames, "duration_seconds": (len(indices) + hold_frames) / args.fps,
        "first_simulation_time": float(times[indices[0]]), "last_simulation_time": float(times[indices[-1]]),
        "full_simulation_time": float(report["simulation_seconds"]), "partial_replay": partial,
        "native_render_wall_seconds": time.perf_counter() - started,
    }
    (project / "render-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verified_training(report: dict) -> dict:
    policy = ROOT / "output/competition/policy"
    training = json.loads((policy / "training.json").read_text())
    weights_sha256 = digest(policy / "weights.npz")
    if report.get("weights_sha256") != weights_sha256 or training.get("weights_sha256") != weights_sha256:
        raise ValueError("Mission, training report, and policy weights must have matching SHA-256 hashes")
    if "A100" not in training.get("gpu", ""):
        raise ValueError("Training report does not identify an A100 GPU")
    if not math.isfinite(training["training_seconds"]) or training["training_seconds"] <= 0:
        raise ValueError("Training duration must be finite and positive")
    return training


def compose(project: Path, report: dict, manifest: dict, training: dict):
    score = report.get("score", {})
    names = {"samples": "Samples", "kits": "Medical kits", "red_patients": "Red patients",
             "yellow_patients": "Yellow patients", "green_patients": "Green patients"}
    rows = []
    for key, task in score.get("per_task", {}).items():
        label = html.escape(names.get(key, key.replace("_", " ")))
        rows.append(f'<div class="task"><span>{label}</span><span class="number">{task["scored_count"]}/{task["required_count"]}</span></div>')
    limit = score.get("official_time_limit_s")
    mode = "Feasibility run" if score.get("mode") == "unlimited_feasibility" else "Competition replay"
    if manifest["partial_replay"]:
        mode = "Pipeline test"
    values = {
        "DURATION": f'{manifest["duration_seconds"]:.8f}',
        "MODE": mode, "CONTROLLER": f'{report.get("policy", "Recorded").capitalize()} controller',
        "SCORE": f'{score.get("score", "—")}/{score.get("maximum_score", "—")}',
        "STATUS": "All tasks complete" if report.get("success") and score.get("all_tasks_complete") else "Run incomplete",
        "SIM_SECONDS": f'{report["simulation_seconds"]:.1f}',
        "OFFICIAL_LIMIT": f"{limit:g} s" if limit is not None else "Not specified",
        "TRAINING_SECONDS": f'{training["training_seconds"]:.1f}',
        "FOOTNOTE": ("Test excerpt · " if manifest["partial_replay"] else "") +
                    f'Native MuJoCo · simulator state feedback · {report.get("tape_mode", "unspecified")} tape',
    }
    template = (project / "composition.template.html.in").read_text()
    for key, value in values.items():
        template = template.replace(f"__{key}__", html.escape(str(value)))
    template = template.replace("__TASK_ROWS__", "\n".join(rows))
    (project / "index.html").write_text(template)
    (project / "source-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (project / "source-training.json").write_text(json.dumps(training, indent=2) + "\n")


def compact_attachment(source: Path, target: Path, duration: float, budget_mb: float):
    bitrate = max(64, math.floor(budget_mb * 1_000_000 * 8 * .92 / duration / 1000))
    passlog = target.parent / ".attachment-pass"
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-an",
            "-vf", "scale=1280:720", "-c:v", "libx264", "-preset", "medium", "-b:v", f"{bitrate}k",
            "-pix_fmt", "yuv420p", "-threads", "2", "-passlogfile", str(passlog)]
    subprocess.run(base + ["-pass", "1", "-f", "null", "-"], check=True)
    subprocess.run(base + ["-pass", "2", "-movflags", "+faststart", str(target)], check=True)
    for path in passlog.parent.glob(passlog.name + "*"):
        path.unlink()
    if target.stat().st_size >= budget_mb * 1_000_000:
        raise RuntimeError("PR video exceeds the requested attachment budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/competition/proof")
    parser.add_argument("--project", type=Path, default=ROOT / "videos/competition-proof")
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--camera-fovy", type=float, default=38.0)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--fps", type=int, choices=(24, 30, 60), default=30)
    parser.add_argument("--hold-final", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, help="Limit a pipeline-test excerpt in simulation seconds")
    parser.add_argument("--reuse-source", action="store_true", help="Reuse a source only if its input hashes and options match")
    parser.add_argument("--native-only", action="store_true", help="Generate native footage and HTML without launching HyperFrames")
    parser.add_argument("--render", action="store_true", help="Render the verified HyperFrames composition and PR-sized MP4")
    parser.add_argument("--attachment-mb", type=float, default=9.5)
    args = parser.parse_args()
    if args.native_only and args.render:
        parser.error("--native-only and --render cannot be combined")
    if args.speed <= 0 or args.hold_final < 0 or (args.max_seconds is not None and args.max_seconds <= 0):
        parser.error("Speed and excerpt length must be positive; final hold must be nonnegative")
    args.input, args.project = args.input.resolve(), args.project.resolve()
    for binary in ("ffmpeg", "ffprobe", "npm"):
        if not shutil.which(binary):
            parser.error(f"Required executable is missing: {binary}")
    report = json.loads((args.input / "report.json").read_text())
    training = verified_training(report)
    inputs = load_recording(args.input, args.speed, args.fps, args.max_seconds)
    manifest_path = args.project / "render-manifest.json"
    if args.reuse_source:
        manifest = json.loads(manifest_path.read_text())
        expected = {"scene_sha256": digest(args.input / "scene.xml"), "trajectory_sha256": digest(args.input / "trajectory.npz"),
                    "report_sha256": digest(args.input / "report.json"), "camera": args.camera, "camera_fovy": args.camera_fovy,
                    "fps": args.fps, "playback_speed": args.speed, "rendered_pose_frames": len(inputs[3]),
                    "final_hold_frames": round(args.hold_final * args.fps), "partial_replay": inputs[4],
                    "last_simulation_time": float(inputs[0][inputs[3][-1]])}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Cached source does not match the selected recording or playback options")
        if digest(args.project / "assets/trajectory.mp4") != manifest["source_video_sha256"]:
            raise ValueError("Cached source video checksum changed")
    else:
        manifest = render_native(args, inputs, report)
    compose(args.project, report, manifest, training)
    if not args.native_only:
        subprocess.run(["npm", "run", "check", "--", "--samples", "3", "--snapshots"], cwd=args.project, check=True)
    if args.render:
        output = args.project / "renders"
        output.mkdir(exist_ok=True)
        video = output / "competition-proof.mp4"
        subprocess.run(["npm", "run", "render", "--", "--quality", "high", "--fps", str(args.fps), "--workers", "4",
                        "--output", str(video)], cwd=args.project, check=True)
        info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_format", "-of", "json", str(video)]))
        duration = float(info["format"]["duration"])
        if abs(duration - manifest["duration_seconds"]) > 2 / args.fps:
            raise RuntimeError("Rendered duration does not match the recorded playback")
        compact_attachment(video, output / "competition-proof-pr.mp4", duration, args.attachment_mb)
        print(json.dumps({"video": str(video), "attachment": str(output / "competition-proof-pr.mp4"), "duration_seconds": duration}))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
