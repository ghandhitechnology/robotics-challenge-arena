#!/usr/bin/env python3
"""Render saved swarm poses from fixed cameras and assemble proof videos.

Playback never steps physics or interpolates poses. Every displayed scene,
simulation time, and phase comes from one saved trajectory sample.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCENE_WIDTH, SCENE_HEIGHT, HUD_HEIGHT = 1440, 960, 80
VIEW_CONFIG = {
    "top": {
        "camera": "overview",
        "default_fovy": 38.0,
        "project": ROOT / "videos/swarm-top",
        "label": "Fixed overhead view",
    },
    "side": {
        "camera": "free",
        "default_fovy": 45.0,
        "project": ROOT / "videos/swarm-side",
        "label": "Fixed low oblique view",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_hashes(directory: Path) -> dict[str, str]:
    names = ("scene.xml", "trajectory.npz", "report.json", "metadata.json", "policy_trace.npz")
    missing = [name for name in names if not (directory / name).is_file()]
    require(not missing, f"Proof input is missing: {', '.join(missing)}")
    return {name: digest(directory / name) for name in names}


def run_proof_audit(args) -> dict:
    """Require action checks and a full native physics replay before video work."""
    verifier = ROOT / "scripts/verify_swarm_proof.py"
    command = [sys.executable, str(verifier), str(args.input), "--replay-physics"]
    if args.allow_teacher_test:
        command.append("--allow-teacher")
    if args.policy_dir:
        command.extend(["--weights", str(args.policy_dir / "weights.npz"),
                        "--training-report", str(args.policy_dir / "training.json")])
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    try:
        audit = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Swarm proof verifier returned invalid output: {completed.stderr.strip()}") from error
    require(completed.returncode == 0 and audit.get("valid") is True,
            "Swarm proof verification failed: " + "; ".join(audit.get("errors", [])))
    require(audit.get("physics_replayed") is True, "Video input must pass full native physics replay")
    if audit.get("evidence_kind") == "teacher_pipeline_test":
        require(args.allow_teacher_test and args.max_seconds is not None,
                "Teacher verification is restricted to labeled pipeline excerpts")
    else:
        require(audit.get("final_neural_proof") is True,
                "Final video input must pass the neural proof gate")
    return audit


def verified_sources(args):
    """Validate one successful recorded benchmark and its policy provenance."""
    hashes = source_hashes(args.input)
    report = json.loads((args.input / "report.json").read_text())
    metadata = json.loads((args.input / "metadata.json").read_text())
    require(report.get("task") == "cooperative_transport_benchmark",
            "Video input must be the cooperative transport benchmark")
    require("official_competition_score" in report and report["official_competition_score"] is None,
            "Benchmark report must explicitly omit an official competition score")
    simulator = str(report.get("simulator", "")).lower().replace("_", "").replace(" ", "")
    require(simulator == "nativemujoco", "Proof must come from native MuJoCo")
    require(report.get("success") is True, "Source benchmark did not complete successfully")
    require(report.get("failure") is None, "Source benchmark records a failure")
    require(report.get("robot_count") == args.expected_robots,
            f"Expected {args.expected_robots} robots")
    require(report.get("object_count") == args.expected_objects,
            f"Expected {args.expected_objects} objects")
    require(report.get("completed_objects") == report.get("object_count"),
            "Every recorded object must be delivered")
    completion = report.get("completion_time_s")
    require(isinstance(completion, (int, float)) and math.isfinite(completion) and completion > 0,
            "Completion time must be finite and positive")
    require(report.get("final_hold_valid") is True and
            isinstance(report.get("final_hold_s"), (int, float)) and
            math.isfinite(report["final_hold_s"]) and report["final_hold_s"] >= 5.0 - 1e-7,
            "Successful placement must remain valid for at least five recorded seconds")
    require(isinstance(report.get("requested_hold_seconds"), (int, float)) and
            math.isfinite(report["requested_hold_seconds"]) and
            report["requested_hold_seconds"] >= 5.0,
            "Proof must request a final hold of at least five seconds")
    require(isinstance(report.get("episode_seconds"), (int, float)) and
            math.isfinite(report["episode_seconds"]) and report["episode_seconds"] > 0 and
            completion <= report["episode_seconds"] + 1e-7,
            "Completion time must fit within the requested episode")
    timestep = report.get("physics_timestep_s")
    require(isinstance(timestep, (int, float)) and math.isfinite(timestep) and timestep > 0,
            "Physics timestep must be finite and positive")

    robots, objects = metadata.get("robots", []), metadata.get("objects", [])
    require(len(robots) == report["robot_count"], "Metadata robot count differs from the report")
    require(len(objects) == report["object_count"], "Metadata object count differs from the report")
    require(len({robot.get("name") for robot in robots}) == len(robots),
            "Metadata robot names must be unique")
    require(len({obj.get("name") for obj in objects}) == len(objects),
            "Metadata object names must be unique")
    for obj in objects:
        require(obj.get("kind") in {"cylinder", "kit", "sample"},
                "Every object must identify its grasp geometry")
        require(all(isinstance(obj.get(key), (int, float)) and math.isfinite(obj[key]) and obj[key] > 0
                    for key in ("radius", "half_height")),
                "Every object must have finite positive dimensions")
    require({obj["kind"] for obj in objects} == {"cylinder", "kit", "sample"},
            "The four-object proof must cover cylinders, a kit, and a sample")

    artifacts = report.get("artifacts", {})
    for name in ("scene.xml", "metadata.json", "trajectory.npz", "policy_trace.npz"):
        require(artifacts.get(name) == hashes[name], f"Report hash mismatch for {name}")

    with np.load(args.input / "trajectory.npz", allow_pickle=False) as saved:
        raw_times = saved["times"]
        raw_poses = saved["qpos"]
        raw_phases = saved["phases"]
        require(np.issubdtype(raw_times.dtype, np.floating), "Trajectory times must be floating point")
        require(np.issubdtype(raw_poses.dtype, np.floating), "Trajectory qpos must be floating point")
        require(raw_phases.dtype.kind in "US", "Trajectory phases must be strings")
        times = np.asarray(raw_times, dtype=np.float64)
        poses = np.asarray(raw_poses, dtype=np.float64)
        phases = np.asarray(raw_phases, dtype=str)
    require(times.ndim == 1 and len(times) > 1, "Trajectory must contain multiple timestamps")
    require(poses.ndim == 2 and len(poses) == len(times), "Trajectory qpos shape is invalid")
    require(phases.shape == times.shape, "Trajectory phases must match the timestamps")
    require(np.isfinite(times).all() and np.isfinite(poses).all(), "Trajectory contains nonfinite values")
    require(abs(float(times[0])) <= 1e-8 and np.all(np.diff(times) > 0),
            "Trajectory must start at zero with strictly increasing times")
    require(completion <= float(times[-1]) + 1e-7, "Completion time lies beyond the trajectory")

    with np.load(args.input / "policy_trace.npz", allow_pickle=False) as saved:
        action_times = np.asarray(saved["times"], dtype=np.float64)
        actions = np.asarray(saved["actions"], dtype=np.float64)
        local = np.asarray(saved["local"], dtype=np.float64)
        active = np.asarray(saved["active"], dtype=np.float64)
    require(action_times.ndim == 1 and actions.ndim == 3 and len(actions) == len(action_times),
            "Policy trace action shape is invalid")
    require(np.isfinite(action_times).all() and np.isfinite(actions).all(),
            "Policy trace contains nonfinite values")
    require(actions.shape[1] == report["robot_count"], "Policy trace robot count differs from the report")
    require(local.ndim == 3 and local.shape[:2] == actions.shape[:2],
            "Policy trace local observations do not match the actions")
    require(active.shape == actions.shape[:2] and np.all(active > .5),
            "All 40 modules must be active throughout the proof")
    features = report.get("local_feature_names", [])
    require("carrier" in features and local.shape[2] == len(features),
            "Policy trace must expose the named carrier role")
    carrier_mask = local[..., features.index("carrier")] > .5
    carrier_counts = carrier_mask.sum(axis=1)
    require(np.all(carrier_counts == carrier_counts[0]),
            "Carrier role count must remain stable throughout the proof")
    role_counts = {
        "carriers": int(carrier_counts[0]),
        "formation": int(report["robot_count"] - carrier_counts[0]),
    }
    require(role_counts == {"carriers": report["object_count"] * 2,
                            "formation": report["robot_count"] - report["object_count"] * 2},
            "The four-payload proof must identify eight carriers and 32 formation modules")

    training = None
    if report.get("policy") == "neural":
        weights = args.policy_dir / "weights.npz"
        training_path = args.policy_dir / "training.json"
        require(weights.is_file() and training_path.is_file(), "Neural proof requires weights.npz and training.json")
        training = json.loads(training_path.read_text())
        weights_sha256 = digest(weights)
        require(report.get("weights_sha256") == weights_sha256 and
                training.get("weights_sha256") == weights_sha256,
                "Proof, training report, and actor weights must have matching SHA-256 hashes")
        require(training.get("acceptance_passed") is True,
                "Training acceptance gate must pass before final video rendering")
        require(isinstance(training.get("gpu"), str) and training["gpu"].strip(),
                "Training report must identify the GPU")
        require(isinstance(training.get("training_seconds"), (int, float)) and
                math.isfinite(training["training_seconds"]) and training["training_seconds"] > 0,
                "Training duration must be finite and positive")
        require(report.get("neural_calls") == len(action_times),
                "Recorded neural call count differs from the policy trace")
    else:
        require(args.allow_teacher_test and args.max_seconds is not None and report.get("policy") == "teacher",
                "Only a labeled, excerpted physical-teacher run may bypass neural provenance")

    model = mujoco.MjModel.from_xml_path(str(args.input / "scene.xml"))
    require(poses.shape[1] == model.nq,
            f"Recorded qpos has {poses.shape[1]} values; saved model requires {model.nq}")
    return report, metadata, training, role_counts, hashes, times, poses, phases


def sampled_indices(times: np.ndarray, speed: float, fps: int, max_seconds: float | None):
    end = float(times[-1]) if max_seconds is None else min(float(times[-1]), float(times[0]) + max_seconds)
    targets = np.arange(float(times[0]), end + 1e-9, speed / fps)
    if end - targets[-1] > 1e-7:
        targets = np.append(targets, end)
    right = np.searchsorted(times, targets).clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    indices = np.where(abs(times[right] - targets) < abs(times[left] - targets), right, left)
    return indices.astype(np.int64), bool(end < float(times[-1]) - 1e-7)


def text_font(path: Path, preferred: int, text: str, maximum_width: int) -> ImageFont.FreeTypeFont:
    size = preferred
    while size > 18:
        font = ImageFont.truetype(str(path), size)
        if font.getlength(text) <= maximum_width:
            return font
        size -= 1
    return ImageFont.truetype(str(path), 18)


def camera_settings(args, view: str) -> dict:
    if view == "top":
        return {
            "camera": VIEW_CONFIG[view]["camera"],
            "camera_fovy": args.top_fovy,
            "camera_fixed": True,
        }
    return {
        "camera": "free",
        "camera_fovy": args.side_fovy,
        "camera_fixed": True,
        "camera_lookat_m": list(args.side_lookat),
        "camera_distance_m": args.side_distance,
        "camera_azimuth_degrees": args.side_azimuth,
        "camera_elevation_degrees": args.side_elevation,
    }


def render_native(args, view: str, project: Path, report: dict, hashes: dict,
                  times: np.ndarray, poses: np.ndarray, phases: np.ndarray,
                  indices: np.ndarray, partial: bool) -> dict:
    assets = project / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.input / "scene.xml"))
    model.vis.global_.offwidth = SCENE_WIDTH
    model.vis.global_.offheight = SCENE_HEIGHT
    config = VIEW_CONFIG[view]
    if view == "top":
        camera_name = config["camera"]
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        require(camera_id >= 0, f"Saved model has no camera named {camera_name!r}")
        model.cam_fovy[camera_id] = args.top_fovy
        camera = camera_name
    else:
        model.vis.global_.fovy = args.side_fovy
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = args.side_lookat
        camera.distance = args.side_distance
        camera.azimuth = args.side_azimuth
        camera.elevation = args.side_elevation
    data = mujoco.MjData(model)
    phase_font_path = assets / "fonts/Barlow-SemiBold.ttf"
    mono_font = ImageFont.truetype(str(assets / "fonts/IBMPlexMono-Regular.ttf"), 27)
    source = assets / "trajectory.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{SCENE_WIDTH}x{SCENE_HEIGHT + HUD_HEIGHT}", "-r", str(args.fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-g", str(args.fps), "-keyint_min", str(args.fps),
        "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(source),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    sample_numbers = [0, len(indices) // 2, len(indices) - 1]
    samples = []
    hold_frames = round(args.hold_final * args.fps)
    started = time.perf_counter()
    try:
        with mujoco.Renderer(model, height=SCENE_HEIGHT, width=SCENE_WIDTH) as renderer:
            for frame_number, recorded_index in enumerate(indices):
                data.qpos[:] = poses[recorded_index]
                data.time = times[recorded_index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                frame = Image.new("RGB", (SCENE_WIDTH, SCENE_HEIGHT + HUD_HEIGHT), "#12181b")
                frame.paste(Image.fromarray(renderer.render()), (0, 0))
                draw = ImageDraw.Draw(frame)
                phase = str(phases[recorded_index])
                clock = f"SIM {times[recorded_index]:07.3f} s  |  {args.speed:g}x replay"
                maximum_phase_width = SCENE_WIDTH - 48 - int(mono_font.getlength(clock)) - 42
                phase_font = text_font(phase_font_path, 28, phase, maximum_phase_width)
                draw.text((24, SCENE_HEIGHT + 23), phase, font=phase_font, fill="#eef1e9")
                draw.text((SCENE_WIDTH - 24, SCENE_HEIGHT + 25), clock, anchor="ra",
                          font=mono_font, fill="#dfc779")
                if frame_number in sample_numbers:
                    samples.append(frame.copy())
                process.stdin.write(frame.tobytes())
                if frame_number % 150 == 0:
                    print(f"{view}: native frame {frame_number + 1}/{len(indices)}; "
                          f"simulation {times[recorded_index]:.3f} s", flush=True)
            for _ in range(hold_frames):
                process.stdin.write(frame.tobytes())
    finally:
        if process.stdin:
            process.stdin.close()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg exited {return_code} while rendering {view}")

    frame.save(assets / "native-poster.png")
    while len(samples) < 3:
        samples.append(frame.copy())
    sheet = Image.new("RGB", (1920, 463), "#12181b")
    for column, sampled_frame in enumerate(samples[:3]):
        sheet.paste(sampled_frame.resize((640, 462), Image.Resampling.LANCZOS), (column * 640, 0))
    sheet.save(project / "native-contact-sheet.png")
    manifest = {
        "scene_sha256": hashes["scene.xml"],
        "trajectory_sha256": hashes["trajectory.npz"],
        "report_sha256": hashes["report.json"],
        "metadata_sha256": hashes["metadata.json"],
        "policy_trace_sha256": hashes["policy_trace.npz"],
        "source_video_sha256": digest(source),
        "mujoco_version": mujoco.__version__,
        "view": view,
        **camera_settings(args, view),
        "fps": args.fps,
        "playback_speed": args.speed,
        "pose_interpolation": False,
        "frame_selection": "nearest_saved_pose",
        "recorded_pose_count": len(times),
        "rendered_pose_frames": len(indices),
        "rendered_recording_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "final_hold_frames": hold_frames,
        "duration_seconds": (len(indices) + hold_frames) / args.fps,
        "first_simulation_time": float(times[indices[0]]),
        "last_simulation_time": float(times[indices[-1]]),
        "completion_time_s": float(report["completion_time_s"]),
        "recorded_final_hold_s": float(report["final_hold_s"]),
        "requested_final_hold_s": float(report["requested_hold_seconds"]),
        "requested_episode_seconds": float(report["episode_seconds"]),
        "partial_replay": partial,
        "native_render_wall_seconds": time.perf_counter() - started,
    }
    (project / "render-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_media_inventory(project, manifest)
    return manifest


def write_media_inventory(project: Path, manifest: dict) -> None:
    media = project / ".media"
    media.mkdir(exist_ok=True)
    duration = manifest["duration_seconds"]
    records = [
        {"id": "image_001", "type": "image", "path": "assets/native-poster.png",
         "source": "existing", "description": f"{manifest['view']} native poster",
         "width": SCENE_WIDTH, "height": SCENE_HEIGHT + HUD_HEIGHT,
         "provenance": {"provider": "local", "adopted": True}},
        {"id": "video_001", "type": "video", "path": "assets/trajectory.mp4",
         "source": "existing", "description": f"{manifest['view']} saved-qpos trajectory",
         "duration": duration, "width": SCENE_WIDTH, "height": SCENE_HEIGHT + HUD_HEIGHT,
         "provenance": {"provider": "local", "adopted": True,
                        "scene_sha256": manifest["scene_sha256"],
                        "trajectory_sha256": manifest["trajectory_sha256"]}},
    ]
    (media / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (media / "index.md").write_text(
        "# .media · 2 assets\n\n"
        "id         type   dur     dims       path                      description\n"
        f"image_001  image  —       1440×1040  assets/native-poster.png  {manifest['view']} native poster\n"
        f"video_001  video  {duration:.1f}s  1440×1040  assets/trajectory.mp4     "
        f"{manifest['view']} saved-qpos trajectory\n"
    )


def compose(project: Path, view: str, report: dict, metadata: dict,
            training: dict | None, role_counts: dict, audit: dict, manifest: dict) -> None:
    partial = manifest["partial_replay"]
    if partial:
        mode = "Pipeline test"
        delivered = "—"
        completion = "—"
        status = ("Temporary physical-teacher excerpt" if report.get("policy") == "teacher"
                  else "Trained-policy excerpt")
    else:
        mode = "Cooperative transport benchmark"
        delivered = f'{report["completed_objects"]} / {report["object_count"]}'
        completion = f'{report["completion_time_s"]:.2f} s'
        status = f'{report["completed_objects"]} payloads delivered'
    controller = ("Trained shared policy" if report.get("policy") == "neural"
                  else "Temporary physical teacher")
    values = {
        "COMPOSITION_ID": project.name,
        "DURATION": f'{manifest["duration_seconds"]:.8f}',
        "MODE": mode,
        "VIEW": VIEW_CONFIG[view]["label"],
        "TITLE": f'{report["robot_count"]}-robot swarm',
        "CONTROLLER": controller,
        "STATUS": status,
        "ROBOTS": str(report["robot_count"]),
        "CARRIERS": str(role_counts["carriers"]),
        "FORMATION": str(role_counts["formation"]),
        "DELIVERED": delivered,
        "COMPLETION": completion,
        "FINAL_HOLD": f'{report["final_hold_s"]:.2f} s verified',
        "PHYSICS_STEP": f'{report["physics_timestep_s"] * 1000:g} ms',
        "FOOTNOTE": "Native MuJoCo · saved qpos · no interpolation",
    }
    template = (project / "composition.template.html.in").read_text()
    for key, value in values.items():
        template = template.replace(f"__{key}__", html.escape(str(value)))
    unresolved = re.findall(r"__[A-Z][A-Z_]*__", template)
    require(not unresolved, f"Composition template contains unresolved values: {unresolved}")
    (project / "index.html").write_text(template)
    (project / "source-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (project / "source-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (project / "source-verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    if training is not None:
        (project / "source-training.json").write_text(json.dumps(training, indent=2) + "\n")


def expected_cache(args, view: str, hashes: dict, times: np.ndarray,
                   indices: np.ndarray, partial: bool) -> dict:
    return {
        "scene_sha256": hashes["scene.xml"],
        "trajectory_sha256": hashes["trajectory.npz"],
        "report_sha256": hashes["report.json"],
        "metadata_sha256": hashes["metadata.json"],
        "policy_trace_sha256": hashes["policy_trace.npz"],
        "proof_verifier_sha256": digest(ROOT / "scripts/verify_swarm_proof.py"),
        "view": view,
        **camera_settings(args, view),
        "fps": args.fps,
        "playback_speed": args.speed,
        "rendered_pose_frames": len(indices),
        "rendered_recording_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "final_hold_frames": round(args.hold_final * args.fps),
        "partial_replay": partial,
        "last_simulation_time": float(times[indices[-1]]),
    }


def compact_attachment(source: Path, target: Path, duration: float, budget_mb: float) -> None:
    bitrate = max(64, math.floor(budget_mb * 1_000_000 * 8 * .92 / duration / 1000))
    passlog = target.parent / f".{target.stem}-pass"
    base = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-an",
        "-vf", "scale=1280:720", "-c:v", "libx264", "-preset", "medium",
        "-b:v", f"{bitrate}k", "-pix_fmt", "yuv420p", "-threads", "2",
        "-passlogfile", str(passlog),
    ]
    subprocess.run(base + ["-pass", "1", "-f", "null", "-"], check=True)
    subprocess.run(base + ["-pass", "2", "-movflags", "+faststart", str(target)], check=True)
    for path in passlog.parent.glob(passlog.name + "*"):
        path.unlink()
    if target.stat().st_size >= budget_mb * 1_000_000:
        raise RuntimeError(f"{target.name} exceeds the {budget_mb:g} MB attachment budget")


def probe_video(path: Path) -> dict:
    payload = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ]))
    video = next(stream for stream in payload["streams"] if stream.get("codec_type") == "video")
    return {
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "width": int(video["width"]),
        "height": int(video["height"]),
        "duration_seconds": float(payload["format"]["duration"]),
    }


def process_view(args, view: str, project: Path, report: dict, metadata: dict,
                 training: dict | None, role_counts: dict, audit: dict, hashes: dict, times: np.ndarray,
                 poses: np.ndarray, phases: np.ndarray) -> dict:
    require(project.is_dir(), f"Video project does not exist: {project}")
    indices, partial = sampled_indices(times, args.speed, args.fps, args.max_seconds)
    manifest_path = project / "render-manifest.json"
    if args.reuse_source:
        require(manifest_path.is_file(), f"No cached render manifest for {view}")
        manifest = json.loads(manifest_path.read_text())
        expected = expected_cache(args, view, hashes, times, indices, partial)
        require(all(manifest.get(key) == value for key, value in expected.items()),
                f"Cached {view} source does not match the selected proof or replay options")
        source = project / "assets/trajectory.mp4"
        require(source.is_file() and digest(source) == manifest.get("source_video_sha256"),
                f"Cached {view} source video checksum changed")
    else:
        manifest = render_native(args, view, project, report, hashes, times, poses, phases,
                                 indices, partial)
    manifest["proof_verifier_sha256"] = digest(ROOT / "scripts/verify_swarm_proof.py")
    manifest["proof_verification"] = {
        key: audit.get(key) for key in ("valid", "final_neural_proof", "evidence_kind")
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    compose(project, view, report, metadata, training, role_counts, audit, manifest)
    if not args.native_only:
        subprocess.run(["npm", "run", "check", "--", "--samples", "3", "--snapshots"],
                       cwd=project, check=True)
    if args.render:
        output = project / "renders"
        output.mkdir(exist_ok=True)
        master = output / f"{project.name}.mp4"
        subprocess.run([
            "npm", "run", "render", "--", "--quality", "high", "--fps", str(args.fps),
            "--workers", "4", "--output", str(master),
        ], cwd=project, check=True)
        master_info = probe_video(master)
        require((master_info["width"], master_info["height"]) == (1920, 1080),
                f"{view} master has unexpected dimensions")
        require(abs(master_info["duration_seconds"] - manifest["duration_seconds"]) <= 2 / args.fps,
                f"{view} master duration does not match the saved-pose replay")
        attachment = output / f"{project.name}-pr.mp4"
        compact_attachment(master, attachment, master_info["duration_seconds"], args.attachment_mb)
        attachment_info = probe_video(attachment)
        require((attachment_info["width"], attachment_info["height"]) == (1280, 720),
                f"{view} attachment has unexpected dimensions")
        manifest["outputs"] = {"master": master_info, "pr_attachment": attachment_info}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/swarm/proof")
    parser.add_argument("--policy-dir", type=Path, default=ROOT / "output/swarm/policy")
    parser.add_argument("--views", nargs="+", choices=tuple(VIEW_CONFIG), default=list(VIEW_CONFIG))
    parser.add_argument("--top-project", type=Path, default=VIEW_CONFIG["top"]["project"])
    parser.add_argument("--side-project", type=Path, default=VIEW_CONFIG["side"]["project"])
    parser.add_argument("--top-fovy", type=float, default=VIEW_CONFIG["top"]["default_fovy"])
    parser.add_argument("--side-fovy", type=float, default=VIEW_CONFIG["side"]["default_fovy"])
    parser.add_argument("--side-lookat", type=float, nargs=3, default=(.5715, .5905, -.05),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--side-distance", type=float, default=1.45)
    parser.add_argument("--side-azimuth", type=float, default=135.0)
    parser.add_argument("--side-elevation", type=float, default=-35.0)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--fps", type=int, choices=(24, 30, 60), default=30)
    parser.add_argument("--hold-final", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float,
                        help="Render a clearly labeled pipeline-test excerpt in simulation seconds")
    parser.add_argument("--expected-robots", type=int, default=40)
    parser.add_argument("--expected-objects", type=int, default=4)
    parser.add_argument("--allow-teacher-test", action="store_true",
                        help="Allow only an excerpted physical-teacher pipeline test")
    parser.add_argument("--reuse-source", action="store_true",
                        help="Reuse native videos only when every source hash and option matches")
    parser.add_argument("--native-only", action="store_true",
                        help="Generate native footage and HTML without launching HyperFrames")
    parser.add_argument("--render", action="store_true",
                        help="Render 1080p masters and sub-9.5 MB 720p PR attachments")
    parser.add_argument("--attachment-mb", type=float, default=9.5)
    args = parser.parse_args()
    if args.native_only and args.render:
        parser.error("--native-only and --render cannot be combined")
    if args.speed <= 0 or args.hold_final < 0 or args.attachment_mb <= 0:
        parser.error("Speed and attachment budget must be positive; final hold must be nonnegative")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if not all(math.isfinite(value) and 0 < value < 180 for value in (args.top_fovy, args.side_fovy)):
        parser.error("Camera fields of view must be finite values between 0 and 180 degrees")
    if (not np.isfinite(args.side_lookat).all() or not math.isfinite(args.side_distance) or
            args.side_distance <= 0 or not math.isfinite(args.side_azimuth) or
            not math.isfinite(args.side_elevation) or not -90 <= args.side_elevation <= 90):
        parser.error("Side-camera look-at, distance, azimuth, and elevation must be finite and valid")
    args.input = args.input.resolve()
    args.policy_dir = args.policy_dir.resolve()
    projects = {"top": args.top_project.resolve(), "side": args.side_project.resolve()}
    for binary in ("ffmpeg", "ffprobe", "npm"):
        if not shutil.which(binary):
            parser.error(f"Required executable is missing: {binary}")
    audit = run_proof_audit(args)
    report, metadata, training, role_counts, hashes, times, poses, phases = verified_sources(args)
    results = {}
    for view in dict.fromkeys(args.views):
        results[view] = process_view(args, view, projects[view], report, metadata, training, role_counts, audit,
                                     hashes, times, poses, phases)
    print(json.dumps({
        "views": list(results),
        "projects": {view: str(projects[view]) for view in results},
        "renders": {
            view: {name: str(projects[view] / "renders" / filename) for name, filename in {
                "master": f"{projects[view].name}.mp4",
                "pr_attachment": f"{projects[view].name}-pr.mp4",
            }.items()} for view in results
        } if args.render else None,
    }, indent=2))


if __name__ == "__main__":
    main()
