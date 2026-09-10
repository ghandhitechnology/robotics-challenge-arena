#!/usr/bin/env python3
"""Render a saved magnetic-body proof from fixed cameras and assemble proof videos.

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
        "camera": "free",
        "default_fovy": 42.0,
        "default_lookat": (.76, .89, .0),
        "default_distance": 1.05,
        "default_azimuth": 0.0,
        "default_elevation": -90.0,
        "project": ROOT / "videos/swarm-body-top",
        "label": "Fixed overhead body view",
    },
    "side": {
        "camera": "free",
        "default_fovy": 42.0,
        "default_lookat": (.72, .89, .02),
        "default_distance": 1.25,
        "default_azimuth": 135.0,
        "default_elevation": -32.0,
        "project": ROOT / "videos/swarm-body-side",
        "label": "Fixed low oblique body view",
    },
}
BODY_PHASE_LABELS = {0: "DEPLOY", 1: "GRASP", 2: "CARRY", 3: "REGROUP"}


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
    verifier = ROOT / "scripts/verify_swarm_body.py"
    command = [sys.executable, str(verifier), str(args.input)]
    if args.allow_teacher_diagnostic:
        command.append("--allow-teacher")
    if args.policy_dir:
        command.extend(["--weights", str(args.policy_dir / "weights.npz"),
                        "--training-report", str(args.policy_dir / "training.json")])
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    try:
        audit = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Magnetic-body verifier returned invalid output: {completed.stderr.strip()}") from error
    require(completed.returncode == 0 and audit.get("valid") is True,
            "Magnetic-body verification failed: " + "; ".join(audit.get("errors", [])))
    require(audit.get("physics_replayed") is True, "Video input must pass full native physics replay")
    if audit.get("evidence_kind") == "teacher_diagnostic":
        require(args.allow_teacher_diagnostic and args.max_seconds is not None,
                "Teacher verification is restricted to labeled diagnostic excerpts")
        require(audit.get("recording_valid") is True,
                "Teacher diagnostic input must pass every recording-integrity check")
    else:
        require(audit.get("final_neural_proof") is True and audit.get("task_passed") is True,
                "Final video input must pass the neural magnetic-body task gate")
    return audit


def verified_sources(args):
    """Validate one recorded magnetic-body benchmark and its policy provenance."""
    hashes = source_hashes(args.input)
    report = json.loads((args.input / "report.json").read_text())
    metadata = json.loads((args.input / "metadata.json").read_text())
    require(report.get("task") == "magnetic_swarm_body_transport" and
            metadata.get("task") == report.get("task"),
            "Video input must be the magnetic swarm-body benchmark")
    require("official_competition_score" in report and report["official_competition_score"] is None,
            "Benchmark report must explicitly omit an official competition score")
    require(report.get("simulator") == "native_MuJoCo" and
            report.get("env_factory") == "arena_mujoco.swarm_body_env:SwarmBodyEnv",
            "Proof must come from the native magnetic-body environment")
    require(report.get("robot_count") == args.expected_robots,
            f"Expected {args.expected_robots} robots")
    require(report.get("object_count") == args.expected_objects,
            f"Expected {args.expected_objects} objects")
    require(report.get("magnetic_force_threshold_n") == .002,
            "Load-bearing connectivity must use the verified 2 mN threshold")
    contract = report.get("policy_contract", {})
    require(contract.get("local_dim") == 40 and contract.get("neighbor_dim") == 12 and
            contract.get("global_dim") == 88 and contract.get("action_dim") == 4,
            "Proof does not use the four-action magnetic-body policy contract")
    require(isinstance(report.get("requested_hold_seconds"), (int, float)) and
            math.isfinite(report["requested_hold_seconds"]) and
            report["requested_hold_seconds"] >= 5.0,
            "Proof must request a final hold of at least five seconds")
    require(report.get("sources_stable_during_recording") is True,
            "Proof sources changed while the recording was running")
    timestep = report.get("physics_timestep_s")
    require(timestep in (.001, .002) and report.get("control_timestep_s") == .02,
            "Proof does not use a validated physics/control timestep")

    robots, objects = metadata.get("robots", []), metadata.get("objects", [])
    require(len(robots) == report["robot_count"], "Metadata robot count differs from the report")
    require(len(objects) == report["object_count"], "Metadata object count differs from the report")
    require(len({robot.get("name") for robot in robots}) == len(robots),
            "Metadata robot names must be unique")
    require(len({obj.get("name") for obj in objects}) == len(objects),
            "Metadata object names must be unique")
    require(all(len(robot.get("magnetic_sites", [])) == 4 for robot in robots),
            "Every module must expose four physical magnetic sites")
    require(metadata.get("magnetic_force_targets") == "robot_root_bodies_only",
            "Magnetic forces must target module root bodies only")

    artifacts = report.get("artifacts", {})
    for name in ("scene.xml", "metadata.json", "trajectory.npz", "policy_trace.npz"):
        require(artifacts.get(name) == hashes[name], f"Report hash mismatch for {name}")

    with np.load(args.input / "trajectory.npz", allow_pickle=False) as saved:
        raw_times, raw_poses = saved["times"], saved["qpos"]
        require(np.issubdtype(raw_times.dtype, np.floating), "Trajectory times must be floating point")
        require(np.issubdtype(raw_poses.dtype, np.floating), "Trajectory qpos must be floating point")
        times = np.asarray(raw_times, dtype=np.float64)
        poses = np.asarray(raw_poses, dtype=np.float64)
        body_phases = np.asarray(saved["body_phase"], dtype=np.int64)
        object_phases = np.asarray(saved["object_phases"], dtype=np.int64)
        magnetic_graphs = np.asarray(saved["magnetic_graph"], dtype=bool)
        magnetic_enabled = np.asarray(saved["magnetic_enabled"], dtype=bool)
        substep_components = np.asarray(saved["substep_min_largest_component"], dtype=np.int64)
    require(times.ndim == 1 and len(times) > 1, "Trajectory must contain multiple timestamps")
    require(poses.ndim == 2 and len(poses) == len(times), "Trajectory qpos shape is invalid")
    require(body_phases.shape == times.shape and object_phases.shape == (len(times), args.expected_objects),
            "Body or payload phases do not match the saved frames")
    require(magnetic_graphs.shape == (len(times), args.expected_robots, args.expected_robots),
            "Magnetic connectivity does not match the saved frames")
    require(magnetic_enabled.shape == (len(times), args.expected_robots),
            "Magnet-enable state does not match the saved frames")
    require(substep_components.shape == times.shape and
            np.all((1 <= substep_components) & (substep_components <= args.expected_robots)),
            "Substep connectivity does not match the saved frames")
    require(np.isfinite(times).all() and np.isfinite(poses).all(), "Trajectory contains nonfinite values")
    require(abs(float(times[0])) <= 1e-8 and np.all(np.diff(times) > 0),
            "Trajectory must start at zero with strictly increasing times")

    with np.load(args.input / "policy_trace.npz", allow_pickle=False) as saved:
        action_times = np.asarray(saved["times"], dtype=np.float64)
        actions = np.asarray(saved["actions"], dtype=np.float64)
        local = np.asarray(saved["local"], dtype=np.float64)
        active = np.asarray(saved["active"], dtype=np.float64)
    require(action_times.ndim == 1 and actions.shape == (len(action_times), args.expected_robots, 4),
            "Policy trace action shape is invalid")
    require(np.isfinite(action_times).all() and np.isfinite(actions).all(),
            "Policy trace contains nonfinite values")
    require(actions.shape[1] == report["robot_count"], "Policy trace robot count differs from the report")
    require(local.ndim == 3 and local.shape[:2] == actions.shape[:2],
            "Policy trace local observations do not match the actions")
    require(local.shape[2] == 40, "Policy trace does not use the 40-feature body observation")
    require(active.shape == actions.shape[:2] and np.all(active > .5),
            "All 40 modules must be active throughout the proof")
    features = report.get("local_feature_names", [])
    role_counts = {
        "carriers": report["object_count"] * 2,
        "body": report["robot_count"] - report["object_count"] * 2,
    }
    require(role_counts == {"carriers": 4, "body": 36},
            "The two-payload body proof must identify four carriers and 36 body modules")

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
        require(report.get("training_report_sha256") == digest(training_path),
                "Proof does not match the selected training report")
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
        require(args.allow_teacher_diagnostic and args.max_seconds is not None and report.get("policy") == "teacher",
                "Only a labeled, excerpted physical-teacher run may bypass neural provenance")

    model = mujoco.MjModel.from_xml_path(str(args.input / "scene.xml"))
    require(poses.shape[1] == model.nq,
            f"Recorded qpos has {poses.shape[1]} values; saved model requires {model.nq}")
    telemetry = {
        "body_phases": body_phases,
        "object_phases": object_phases,
        "magnetic_graphs": magnetic_graphs,
        "magnetic_enabled": magnetic_enabled,
        "substep_components": substep_components,
    }
    return report, metadata, training, role_counts, hashes, times, poses, telemetry


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


def largest_component(graph: np.ndarray) -> int:
    seen: set[int] = set()
    largest = 0
    for start in range(len(graph)):
        if start in seen:
            continue
        pending, size = [start], 0
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            size += 1
            pending.extend(np.flatnonzero(graph[node]).tolist())
        largest = max(largest, size)
    return largest


def camera_settings(args, view: str) -> dict:
    prefix = "top" if view == "top" else "side"
    return {
        "camera": "free",
        "camera_fovy": getattr(args, f"{prefix}_fovy"),
        "camera_fixed": True,
        "camera_lookat_m": list(getattr(args, f"{prefix}_lookat")),
        "camera_distance_m": getattr(args, f"{prefix}_distance"),
        "camera_azimuth_degrees": getattr(args, f"{prefix}_azimuth"),
        "camera_elevation_degrees": getattr(args, f"{prefix}_elevation"),
    }


def render_native(args, view: str, project: Path, report: dict, hashes: dict,
                  times: np.ndarray, poses: np.ndarray, telemetry: dict, audit: dict,
                  indices: np.ndarray, partial: bool) -> dict:
    assets = project / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.input / "scene.xml"))
    model.vis.global_.offwidth = SCENE_WIDTH
    model.vis.global_.offheight = SCENE_HEIGHT
    settings = camera_settings(args, view)
    model.vis.global_.fovy = settings["camera_fovy"]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = settings["camera_lookat_m"]
    camera.distance = settings["camera_distance_m"]
    camera.azimuth = settings["camera_azimuth_degrees"]
    camera.elevation = settings["camera_elevation_degrees"]
    data = mujoco.MjData(model)
    phase_font_path = assets / "fonts/Barlow-SemiBold.ttf"
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
                body_phase = int(telemetry["body_phases"][recorded_index])
                phase = BODY_PHASE_LABELS[body_phase]
                delivered = int(np.sum(telemetry["object_phases"][recorded_index] == 5))
                graph = telemetry["magnetic_graphs"][recorded_index]
                edges = int(np.triu(graph, 1).sum())
                core = largest_component(graph)
                enabled = int(telemetry["magnetic_enabled"][recorded_index].sum())
                minimum = int(telemetry["substep_components"][recorded_index])
                phase_text = f"{phase} · {delivered}/{report['object_count']} delivered"
                evidence = (f"{edges} links ≥2 mN · core {core}/{report['robot_count']} · step min {minimum} · "
                            f"enabled {enabled}/{report['robot_count']} · SIM {times[recorded_index]:07.3f} s")
                phase_font = text_font(phase_font_path, 28, phase_text, 370)
                evidence_font = text_font(assets / "fonts/IBMPlexMono-Regular.ttf", 23,
                                          evidence, SCENE_WIDTH - 450)
                draw.text((24, SCENE_HEIGHT + 23), phase_text, font=phase_font, fill="#eef1e9")
                draw.text((SCENE_WIDTH - 24, SCENE_HEIGHT + 25), evidence, anchor="ra",
                          font=evidence_font, fill="#dfc779")
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
        "hud_evidence": ("saved body_phase, object_phases, magnetic_graph, magnetic_enabled, "
                         "substep_min_largest_component"),
        "magnetic_force_threshold_n": float(report["magnetic_force_threshold_n"]),
        "magnetic_link_visualization": "measured telemetry only; no synthetic attraction geometry",
        "recorded_pose_count": len(times),
        "rendered_pose_frames": len(indices),
        "rendered_recording_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "final_hold_frames": hold_frames,
        "duration_seconds": (len(indices) + hold_frames) / args.fps,
        "first_simulation_time": float(times[indices[0]]),
        "last_simulation_time": float(times[indices[-1]]),
        "completion_time_s": report.get("completion_time_s"),
        "verified_final_hold_s": audit_hold_s(report, audit),
        "requested_final_hold_s": float(report["requested_hold_seconds"]),
        "requested_episode_seconds": float(report["episode_seconds"]),
        "partial_replay": partial,
        "final_delivered_payloads": int(np.sum(telemetry["object_phases"][indices[-1]] == 5)),
        "final_load_bearing_edges": int(np.triu(telemetry["magnetic_graphs"][indices[-1]], 1).sum()),
        "final_largest_load_bearing_component": largest_component(
            telemetry["magnetic_graphs"][indices[-1]]),
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


def audit_hold_s(report: dict, audit: dict | None = None) -> float | None:
    value = (audit or {}).get("checks", {}).get("final_hold_s", report.get("final_hold_s"))
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def compose(project: Path, view: str, report: dict, metadata: dict,
            training: dict | None, role_counts: dict, audit: dict, manifest: dict) -> None:
    if manifest["diagnostic"]:
        mode = "Recording diagnostic"
        delivered = f'{manifest["final_delivered_payloads"]} / {report["object_count"]}'
        completion = "—"
        status = ("Temporary physical-teacher excerpt" if report.get("policy") == "teacher"
                  else "Trained-policy diagnostic excerpt")
    else:
        mode = "Magnetic body transport benchmark"
        delivered = f'{report["object_count"]} / {report["object_count"]}'
        completion = f'{report["completion_time_s"]:.2f} s'
        status = f'{report["object_count"]} payloads delivered · body rejoined'
    controller = ("Trained shared four-action policy" if report.get("policy") == "neural"
                  else "Temporary physical teacher")
    hold = audit_hold_s(report, audit)
    values = {
        "COMPOSITION_ID": project.name,
        "DURATION": f'{manifest["duration_seconds"]:.8f}',
        "MODE": mode,
        "VIEW": VIEW_CONFIG[view]["label"],
        "TITLE": f'{report["robot_count"]}-module magnetic body',
        "CONTROLLER": controller,
        "STATUS": status,
        "ROBOTS": str(report["robot_count"]),
        "CARRIERS": str(role_counts["carriers"]),
        "BODY": str(role_counts["body"]),
        "DELIVERED": delivered,
        "MAGNETIC_CORE": f'{manifest["final_largest_load_bearing_component"]} / {report["robot_count"]}',
        "COMPLETION": completion,
        "FINAL_HOLD": f"{hold:.2f} s verified" if hold is not None else "—",
        "PHYSICS_STEP": f'{report["physics_timestep_s"] * 1000:g} ms',
        "FOOTNOTE": "Native MuJoCo · ≥2 mN load links · saved qpos",
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
    else:
        (project / "source-training.json").unlink(missing_ok=True)


def expected_cache(args, view: str, hashes: dict, times: np.ndarray,
                   indices: np.ndarray, partial: bool, diagnostic: bool) -> dict:
    return {
        "scene_sha256": hashes["scene.xml"],
        "trajectory_sha256": hashes["trajectory.npz"],
        "report_sha256": hashes["report.json"],
        "metadata_sha256": hashes["metadata.json"],
        "policy_trace_sha256": hashes["policy_trace.npz"],
        "proof_verifier_sha256": digest(ROOT / "scripts/verify_swarm_body.py"),
        "view": view,
        **camera_settings(args, view),
        "fps": args.fps,
        "playback_speed": args.speed,
        "rendered_pose_frames": len(indices),
        "rendered_recording_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "final_hold_frames": round(args.hold_final * args.fps),
        "partial_replay": partial,
        "diagnostic": diagnostic,
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
                 poses: np.ndarray, telemetry: dict) -> dict:
    require(project.is_dir(), f"Video project does not exist: {project}")
    indices, partial = sampled_indices(times, args.speed, args.fps, args.max_seconds)
    diagnostic = args.max_seconds is not None or report.get("policy") == "teacher"
    manifest_path = project / "render-manifest.json"
    if args.reuse_source:
        require(manifest_path.is_file(), f"No cached render manifest for {view}")
        manifest = json.loads(manifest_path.read_text())
        expected = expected_cache(args, view, hashes, times, indices, partial, diagnostic)
        require(all(manifest.get(key) == value for key, value in expected.items()),
                f"Cached {view} source does not match the selected proof or replay options")
        source = project / "assets/trajectory.mp4"
        require(source.is_file() and digest(source) == manifest.get("source_video_sha256"),
                f"Cached {view} source video checksum changed")
    else:
        manifest = render_native(args, view, project, report, hashes, times, poses, telemetry, audit,
                                 indices, partial)
        manifest["diagnostic"] = diagnostic
    manifest["proof_verifier_sha256"] = digest(ROOT / "scripts/verify_swarm_body.py")
    manifest["proof_verification"] = {
        key: audit.get(key) for key in (
            "valid", "recording_valid", "task_passed", "final_neural_proof",
            "physics_replayed", "evidence_kind",
        )
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
    parser.add_argument("--input", type=Path, default=ROOT / "output/swarm/body_proof")
    parser.add_argument("--policy-dir", type=Path, default=ROOT / "output/swarm/body_policy")
    parser.add_argument("--views", nargs="+", choices=tuple(VIEW_CONFIG), default=list(VIEW_CONFIG))
    parser.add_argument("--top-project", type=Path, default=VIEW_CONFIG["top"]["project"])
    parser.add_argument("--side-project", type=Path, default=VIEW_CONFIG["side"]["project"])
    parser.add_argument("--top-fovy", type=float, default=VIEW_CONFIG["top"]["default_fovy"])
    parser.add_argument("--side-fovy", type=float, default=VIEW_CONFIG["side"]["default_fovy"])
    for view in VIEW_CONFIG:
        parser.add_argument(f"--{view}-lookat", type=float, nargs=3,
                            default=VIEW_CONFIG[view]["default_lookat"], metavar=("X", "Y", "Z"))
        parser.add_argument(f"--{view}-distance", type=float,
                            default=VIEW_CONFIG[view]["default_distance"])
        parser.add_argument(f"--{view}-azimuth", type=float,
                            default=VIEW_CONFIG[view]["default_azimuth"])
        parser.add_argument(f"--{view}-elevation", type=float,
                            default=VIEW_CONFIG[view]["default_elevation"])
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--fps", type=int, choices=(24, 30, 60), default=30)
    parser.add_argument("--hold-final", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float,
                        help="Render a clearly labeled recording-diagnostic excerpt in simulation seconds")
    parser.add_argument("--expected-robots", type=int, default=40)
    parser.add_argument("--expected-objects", type=int, default=2)
    parser.add_argument("--allow-teacher-diagnostic", action="store_true",
                        help="Allow only an excerpted physical-teacher recording diagnostic")
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
    for view in VIEW_CONFIG:
        lookat = getattr(args, f"{view}_lookat")
        distance = getattr(args, f"{view}_distance")
        azimuth = getattr(args, f"{view}_azimuth")
        elevation = getattr(args, f"{view}_elevation")
        if (not np.isfinite(lookat).all() or not math.isfinite(distance) or distance <= 0 or
                not math.isfinite(azimuth) or not math.isfinite(elevation) or
                not -90 <= elevation <= 90):
            parser.error(f"{view.title()} camera pose must be finite and valid")
    args.input = args.input.resolve()
    args.policy_dir = args.policy_dir.resolve()
    projects = {"top": args.top_project.resolve(), "side": args.side_project.resolve()}
    for binary in ("ffmpeg", "ffprobe", "npm"):
        if not shutil.which(binary):
            parser.error(f"Required executable is missing: {binary}")
    audit = run_proof_audit(args)
    report, metadata, training, role_counts, hashes, times, poses, telemetry = verified_sources(args)
    if args.render and (args.max_seconds is not None or report.get("policy") != "neural"):
        parser.error("Final masters require a complete neural proof; diagnostics cannot use --render")
    results = {}
    for view in dict.fromkeys(args.views):
        results[view] = process_view(args, view, projects[view], report, metadata, training, role_counts, audit,
                                     hashes, times, poses, telemetry)
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
