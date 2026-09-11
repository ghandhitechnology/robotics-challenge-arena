#!/usr/bin/env python3
"""Capture a native MuJoCo recording from saved fleet states at real-time speed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="directory containing scene.xml and trajectory.npz")
    parser.add_argument("output", type=Path)
    parser.add_argument("--view", choices=("top", "oblique"), default="top")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error("fps must be between 1 and 60")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required")
    sources = {name: digest(args.input / name) for name in
               ("scene.xml", "trajectory.npz", "report.json", "metadata.json")}
    model = mujoco.MjModel.from_xml_path(str(args.input / "scene.xml"))
    data = mujoco.MjData(model)
    with np.load(args.input / "trajectory.npz", allow_pickle=False) as trace:
        times, poses, velocities = trace["time"], trace["qpos"], trace["qvel"]
    if (len(times) < 2 or poses.shape != (len(times), model.nq)
            or velocities.shape != (len(times), model.nv)
            or not np.isfinite(times).all() or not np.isfinite(poses).all()
            or not np.isfinite(velocities).all() or not (np.diff(times) > 0).all()):
        raise ValueError("Invalid recorded state sequence")
    target_times = np.arange(times[0], times[-1] + 1 / args.fps, 1 / args.fps)
    right = np.searchsorted(times, target_times).clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    indices = np.where(abs(times[right] - target_times) < abs(times[left] - target_times), right, left)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (.5715, .5905, .015)
    camera.distance = 1.85 if args.view == "top" else 2.05
    camera.azimuth = 90 if args.view == "top" else 135
    camera.elevation = -90 if args.view == "top" else -55
    model.vis.global_.fovy = 45
    width, height = 960, 720
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".mp4.tmp")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(args.fps),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(temporary)]
    # State assignment here is playback. Physics results come from the source rollout.
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for index in indices:
                data.qpos[:] = poses[index]
                data.qvel[:] = velocities[index]
                data.time = times[index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                process.stdin.write(renderer.render().tobytes())
        finally:
            process.stdin.close()
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f"ffmpeg exited with {returncode}")
    if sources != {name: digest(args.input / name) for name in sources}:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Source recording changed during capture")
    temporary.replace(args.output)
    report = json.loads((args.input / "report.json").read_text())
    manifest = {
        "kind": "native MuJoCo saved-state simulation capture",
        "view": args.view, "source_hashes": sources,
        "source_directory": str(args.input), "video_sha256": digest(args.output),
        "fps": args.fps, "width": width, "height": height,
        "frame_count": len(indices), "playback_speed": 1.0,
        "first_simulation_time": float(times[0]), "last_simulation_time": float(times[-1]),
        "sampling": "nearest saved state; no pose interpolation or physics advancement",
        "robot_names": [robot["name"] for robot in
                        json.loads((args.input / "metadata.json").read_text())["robots"]],
        "source_success": report.get("success"),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
