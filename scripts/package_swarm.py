#!/usr/bin/env python3
"""Build a portable, provenance-checked archive of the complete swarm result."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
PROOF_NAMES = ("scene.xml", "metadata.json", "trajectory.npz", "policy_trace.npz")
POLICY_NAMES = ("checkpoint.pt", "weights.npz", "warmstart.npz", "history.jsonl", "training.json")
OPTIONAL_POLICY_NAMES = ("bc_initial.npz", "best.npz", "dagger.json", "dagger.npz", "progress.json",
                         "demonstration_summary.json", "run_manifest.json")
VIDEO_PROJECT_FILES = (
    "BRIEF.md",
    "README.md",
    "composition.template.html.in",
    "design.md",
    "hyperframes.json",
    "index.html",
    "meta.json",
    "package.json",
    "render-manifest.json",
    "source-metadata.json",
    "source-report.json",
    "source-training.json",
    "source-verification.json",
    "assets/gsap.min.js",
    "assets/native-poster.png",
    "assets/trajectory.mp4",
    "assets/fonts/Barlow-OFL.txt",
    "assets/fonts/Barlow-SemiBold.ttf",
    "assets/fonts/IBMPlexMono-OFL.txt",
    "assets/fonts/IBMPlexMono-Regular.ttf",
)
SOURCE_FILES = (
    "README.md",
    "docs/swarm_learning.md",
    "docs/swarm_robot_design.md",
    "arena_mujoco/__init__.py",
    "arena_mujoco/assets/biohazard.json",
    "arena_mujoco/builder.py",
    "arena_mujoco/geometry.py",
    "arena_mujoco/materials.py",
    "arena_mujoco/swarm_env.py",
    "arena_mujoco/swarm_gpu.py",
    "arena_mujoco/swarm_policy.py",
    "arena_mujoco/swarm_robot.py",
    "scripts/benchmark_swarm.py",
    "scripts/export_swarm_robot.py",
    "scripts/package_swarm.py",
    "scripts/render_swarm_robot.py",
    "scripts/render_swarm_video.py",
    "scripts/run_swarm.py",
    "scripts/test_swarm_evaluation.py",
    "scripts/test_swarm_imitation.py",
    "scripts/test_swarm_proof.py",
    "scripts/test_swarm_rewards.py",
    "scripts/train_swarm_policy.py",
    "scripts/verify_swarm_policy.py",
    "scripts/verify_swarm_proof.py",
)
REQUIREMENT_FILES = ("requirements-mujoco.txt", "requirements-qa.txt", "requirements-swarm.txt")
ARENA_INPUT_FILES = (
    "arena_spec.json",
    "swarm_robot_design.json",
    "profiles/default.json",
    "profiles/measurements_template.json",
)
CAD_FILES = (
    "README.md",
    "assembly_reference_mm.stl",
    "cad_manifest.json",
    "cad_validation.json",
    "contact_detail.png",
    "contact_validation.json",
    "geometry.json",
    "module.png",
    "module_overview.png",
    "module_top.png",
    "robot.xml",
    "swarm_module.blend",
)
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class BundleError(ValueError):
    """A required artifact or provenance relationship is invalid."""


@dataclass(frozen=True)
class BundleFile:
    source: Path
    path: str
    category: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"Invalid JSON artifact {path}: {error}") from error
    require(isinstance(value, dict), f"JSON artifact must contain an object: {path}")
    return value


def portable_path(value: str) -> str:
    path = PurePosixPath(value)
    require("\\" not in value and value == path.as_posix() and
            not path.is_absolute() and ".." not in path.parts,
            f"Archive path is not portable: {value}")
    require(path.parts and all(part not in ("", ".") for part in path.parts),
            f"Archive path is not portable: {value}")
    return value


def add_file(files: dict[str, BundleFile], source: Path, archive_path: str, category: str,
             *, allow_empty: bool = False) -> None:
    archive_path = portable_path(archive_path)
    require(source.is_file() and not source.is_symlink(), f"Missing required file: {source}")
    require(allow_empty or source.stat().st_size > 0, f"Required file is empty: {source}")
    existing = files.get(archive_path)
    if existing:
        require(existing.source.resolve() == source.resolve(), f"Duplicate archive path: {archive_path}")
        return
    files[archive_path] = BundleFile(source.resolve(), archive_path, category)


def logical_files(files: dict[str, BundleFile], directory: Path, logical_dir: str,
                  names: tuple[str, ...], category: str) -> None:
    for name in names:
        add_file(files, directory / name, f"{logical_dir}/{name}", category)


def validate_policy(policy_dir: Path, proof_dir: Path) -> tuple[dict, dict, dict, dict, dict[str, str]]:
    for name in POLICY_NAMES:
        require((policy_dir / name).is_file(), f"Accepted policy artifact is missing: {policy_dir / name}")
    for name in (*PROOF_NAMES, "report.json", "verification.json"):
        require((proof_dir / name).is_file(), f"Native proof artifact is missing: {proof_dir / name}")

    training = json_object(policy_dir / "training.json")
    report = json_object(proof_dir / "report.json")
    verification = json_object(proof_dir / "verification.json")
    metadata = json_object(proof_dir / "metadata.json")
    weight_hash = sha256(policy_dir / "weights.npz")

    require(training.get("acceptance_passed") is True, "Training acceptance gate did not pass")
    require(report.get("policy") == "neural", "The final bundle requires a neural proof")
    require(report.get("success") is True and report.get("failure") is None,
            "The native proof report is not successful")
    require(report.get("simulator") == "native_MuJoCo", "The proof was not recorded in native MuJoCo")
    require(report.get("robot_count") == 40 and report.get("object_count") == 4 and
            report.get("completed_objects") == 4, "The proof must complete four payloads with 40 robots")
    require(report.get("official_competition_score", "missing") is None,
            "The cooperative benchmark must not claim an official score")
    require(training.get("weights_sha256") == weight_hash and report.get("weights_sha256") == weight_hash,
            "Training, proof, and exported weights do not share one SHA-256")

    require(verification.get("valid") is True, "Saved proof verification is invalid")
    require(verification.get("final_neural_proof") is True,
            "Saved verification is not a final neural proof")
    require(verification.get("physics_replayed") is True,
            "Saved verification does not contain a successful native physics replay")
    require(verification.get("evidence_kind") == "neural_transport",
            "Saved verification identifies teacher or unknown evidence")
    require(verification.get("errors") == [], "Saved proof verification contains errors")
    checks = verification.get("checks")
    require(isinstance(checks, dict), "Saved proof verification has no checks object")
    require(checks.get("report_sha256") == sha256(proof_dir / "report.json"),
            "Saved verification does not match the proof report")
    require(checks.get("training_report_sha256") == sha256(policy_dir / "training.json"),
            "Saved verification does not match the accepted training report")

    artifact_hashes = report.get("artifacts")
    require(isinstance(artifact_hashes, dict), "Proof report has no artifact hash manifest")
    for name in PROOF_NAMES:
        require(artifact_hashes.get(name) == sha256(proof_dir / name),
                f"Proof report hash mismatch for {name}")
    source_hashes = report.get("source_hashes")
    require(isinstance(source_hashes, dict) and source_hashes,
            "Proof report has no source hash manifest")
    for name, expected in source_hashes.items():
        relative = portable_path(name)
        source = (ROOT / relative).resolve()
        require(source.is_relative_to(ROOT) and source.is_file() and not source.is_symlink(),
                f"Proof source is missing or outside the repository: {name}")
        require(expected == sha256(source), f"Proof source hash mismatch for {name}")

    source_commit = report.get("source_commit")
    require(isinstance(source_commit, str) and len(source_commit) == 40 and
            all(character in "0123456789abcdef" for character in source_commit.lower()),
            "Proof report has no valid source commit")
    return training, report, verification, metadata, {name: sha256(proof_dir / name) for name in PROOF_NAMES}


def close_duration(left: object, right: object, tolerance: float = 0.08) -> bool:
    return (isinstance(left, (int, float)) and isinstance(right, (int, float)) and
            math.isfinite(float(left)) and math.isfinite(float(right)) and
            abs(float(left) - float(right)) <= tolerance)


def validate_render(project: Path, view: str, policy_dir: Path, proof_dir: Path,
                    training: dict, report: dict, verification: dict, metadata: dict,
                    proof_hashes: dict[str, str]) -> tuple[Path, Path]:
    manifest_path = project / "render-manifest.json"
    manifest = json_object(manifest_path)
    expected_hash_keys = {
        "scene.xml": "scene_sha256",
        "metadata.json": "metadata_sha256",
        "trajectory.npz": "trajectory_sha256",
        "policy_trace.npz": "policy_trace_sha256",
    }
    require(manifest.get("view") == view.removeprefix("swarm-"),
            f"{view} render manifest identifies the wrong view")
    require(manifest.get("partial_replay") is False, f"{view} is a partial render")
    require(manifest.get("report_sha256") == sha256(proof_dir / "report.json"),
            f"{view} render manifest does not match the proof report")
    for proof_name, manifest_key in expected_hash_keys.items():
        require(manifest.get(manifest_key) == proof_hashes[proof_name],
                f"{view} render manifest does not match {proof_name}")
    require(manifest.get("proof_verifier_sha256") == sha256(ROOT / "scripts/verify_swarm_proof.py"),
            f"{view} render manifest does not match the packaged verifier")
    native_video = project / "assets/trajectory.mp4"
    require(native_video.is_file() and not native_video.is_symlink() and
            manifest.get("source_video_sha256") == sha256(native_video),
            f"{view} native trajectory does not match its render manifest")
    require((project / "assets/native-poster.png").is_file(), f"{view} native poster is missing")
    render_check = manifest.get("proof_verification")
    require(isinstance(render_check, dict) and render_check.get("valid") is True and
            render_check.get("final_neural_proof") is True and
            render_check.get("evidence_kind") == "neural_transport",
            f"{view} render manifest is not tied to final neural evidence")

    require(json_object(project / "source-report.json") == report,
            f"{view} copied report does not match the packaged proof")
    require(json_object(project / "source-training.json") == training,
            f"{view} copied training record does not match the accepted policy")
    require(json_object(project / "source-metadata.json") == metadata,
            f"{view} copied metadata does not match the packaged proof")
    copied_verification = json_object(project / "source-verification.json")
    for key in ("valid", "final_neural_proof", "physics_replayed", "evidence_kind"):
        require(copied_verification.get(key) == verification.get(key),
                f"{view} copied verification differs at {key}")
    copied_checks = copied_verification.get("checks", {})
    require(copied_checks.get("report_sha256") == sha256(proof_dir / "report.json") and
            copied_checks.get("training_report_sha256") == sha256(policy_dir / "training.json"),
            f"{view} copied verification does not bind the proof and weights")

    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), f"{view} render manifest has no final outputs")
    master = project / "renders" / f"{view}.mp4"
    attachment = project / "renders" / f"{view}-pr.mp4"
    for label, path, dimensions in (
        ("master", master, (1920, 1080)),
        ("pr_attachment", attachment, (1280, 720)),
    ):
        info = outputs.get(label)
        require(isinstance(info, dict), f"{view} render manifest has no {label} record")
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                f"{view} {label} is missing")
        require(info.get("sha256") == sha256(path) and info.get("bytes") == path.stat().st_size,
                f"{view} {label} hash or byte count changed")
        require((info.get("width"), info.get("height")) == dimensions,
                f"{view} {label} dimensions are not {dimensions[0]}x{dimensions[1]}")
        require(close_duration(info.get("duration_seconds"), manifest.get("duration_seconds")),
                f"{view} {label} duration does not match its saved-pose render")
    require(report.get("weights_sha256") == sha256(policy_dir / "weights.npz"),
            f"{view} is not bound to the packaged weights")
    return master, attachment


def is_job_log(path: Path) -> bool:
    name = path.name.lower()
    return (path.suffix.lower() in {".log", ".out", ".err"} or
            name in {"stdout.txt", "stderr.txt", "training.stdout", "training.stderr"} or
            name.startswith("slurm-"))


def collect_job_logs(policy_dir: Path, swarm_output: Path) -> list[tuple[Path, str]]:
    found: dict[str, Path] = {}
    for path in policy_dir.rglob("*") if policy_dir.is_dir() else ():
        if path.is_file() and not path.is_symlink() and is_job_log(path):
            logical = f"output/swarm/policy/{path.relative_to(policy_dir).as_posix()}"
            found[logical] = path
    for directory_name in ("logs", "jobs"):
        directory = swarm_output / directory_name
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file() and not path.is_symlink() and is_job_log(path):
                    logical = f"output/swarm/{directory_name}/{path.relative_to(directory).as_posix()}"
                    found[logical] = path
    if swarm_output.is_dir():
        for path in swarm_output.iterdir():
            if path.is_file() and not path.is_symlink() and is_job_log(path):
                found[f"output/swarm/{path.name}"] = path
    return [(path, logical) for logical, path in sorted(found.items())]


def collect_files(args, training: dict, report: dict, verification: dict,
                  metadata: dict, proof_hashes: dict[str, str]) -> tuple[dict[str, BundleFile], dict]:
    files: dict[str, BundleFile] = {}
    for name in SOURCE_FILES:
        add_file(files, ROOT / name, name, "source")
    for name in REQUIREMENT_FILES:
        add_file(files, ROOT / name, name, "requirements")
    for name in ARENA_INPUT_FILES:
        add_file(files, ROOT / name, name, "arena_input")
    for name in report["source_hashes"]:
        add_file(files, ROOT / name, name, "proof_source")

    cad_dir = ROOT / "output/swarm/robot"
    logical_files(files, cad_dir, "output/swarm/robot", CAD_FILES, "robot_cad")
    parts = sorted((cad_dir / "parts").glob("*.stl"))
    require(parts, "Robot CAD parts are missing")
    for path in parts:
        add_file(files, path, f"output/swarm/robot/parts/{path.name}", "robot_cad")

    logical_files(files, args.policy_dir, "output/swarm/policy", POLICY_NAMES, "accepted_policy")
    for name in OPTIONAL_POLICY_NAMES:
        path = args.policy_dir / name
        if path.is_file():
            add_file(files, path, f"output/swarm/policy/{name}", "policy_support")
    reported_exports = training.get("imitation_exports", {})
    require(isinstance(reported_exports, dict), "Training imitation_exports must be an object")
    for name in reported_exports:
        require(isinstance(name, str) and PurePosixPath(name).name == name and name.endswith(".npz"),
                f"Training report names an unsafe imitation export: {name!r}")
        add_file(files, args.policy_dir / name, f"output/swarm/policy/{name}",
                 "imitation_policy")
    for path, logical in collect_job_logs(args.policy_dir, ROOT / "output/swarm"):
        add_file(files, path, logical, "training_job_log", allow_empty=True)

    logical_files(files, args.proof_dir, "output/swarm/proof",
                  (*PROOF_NAMES, "report.json", "verification.json"), "native_proof")

    render_summary = {}
    for view, project in (("swarm-top", args.top_project), ("swarm-side", args.side_project)):
        master, attachment = validate_render(project, view, args.policy_dir, args.proof_dir,
                                             training, report, verification, metadata, proof_hashes)
        logical_files(files, project, f"videos/{view}", VIDEO_PROJECT_FILES, "video_source")
        add_file(files, master, f"videos/{view}/renders/{view}.mp4", "video_master")
        add_file(files, attachment, f"videos/{view}/renders/{view}-pr.mp4", "video_attachment")
        render_summary[view] = {
            "manifest": f"videos/{view}/render-manifest.json",
            "master": f"videos/{view}/renders/{view}.mp4",
            "attachment": f"videos/{view}/renders/{view}-pr.mp4",
        }

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(len(revision) == 40, "Could not determine the repository source revision")
    manifest = {
        "schema_version": 1,
        "bundle": "swarm_complete",
        "source_revision": revision,
        "proof_source_revision": report["source_commit"],
        "policy": {
            "acceptance_passed": True,
            "weights_sha256": training["weights_sha256"],
            "training_report": "output/swarm/policy/training.json",
        },
        "proof": {
            "final_neural_proof": True,
            "physics_replayed": True,
            "report": "output/swarm/proof/report.json",
            "verification": "output/swarm/proof/verification.json",
        },
        "renders": render_summary,
        "job_logs": sorted(item.path for item in files.values() if item.category == "training_job_log"),
        "archive_layout": "Paths are relative to the extracted swarm_complete directory.",
        "validation_gates": ["zip_crc", "clean_native_scene_import"],
        "files": [],
    }
    return files, manifest


def file_record(item: BundleFile) -> dict:
    return {
        "path": item.path,
        "category": item.category,
        "bytes": item.source.stat().st_size,
        "sha256": sha256(item.source),
    }


def zip_info(name: str) -> ZipInfo:
    info = ZipInfo(f"swarm_complete/{portable_path(name)}", ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def write_member(bundle: ZipFile, item: BundleFile) -> None:
    with item.source.open("rb") as source, bundle.open(zip_info(item.path), "w", force_zip64=True) as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def clean_reproduction_check(bundle: ZipFile, records: list[dict]) -> None:
    """Import the packaged native swarm source from an otherwise clean directory."""
    categories = {"source", "proof_source", "requirements", "arena_input"}
    selected = [f"swarm_complete/{record['path']}" for record in records
                if record["category"] in categories]
    with tempfile.TemporaryDirectory(prefix="swarm-package-check-") as temporary:
        for name in selected:
            bundle.extract(name, temporary)
        extracted = Path(temporary) / "swarm_complete"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        check = """
from pathlib import Path
root = Path.cwd().resolve()
import arena_mujoco
import arena_mujoco.swarm_gpu
from arena_mujoco.swarm_env import build_swarm_scene
from arena_mujoco.swarm_policy import NumpySwarmPolicy
assert Path(arena_mujoco.__file__).resolve().is_relative_to(root)
xml, metadata = build_swarm_scene(num_robots=2, num_objects=1, timestep=.002)
assert '<mujoco' in xml and len(metadata['robots']) == 2 and len(metadata['objects']) == 1
assert (root / 'requirements-swarm.txt').is_file()
"""
        completed = subprocess.run([sys.executable, "-c", check], cwd=extracted,
                                   env=environment, text=True, capture_output=True)
        require(completed.returncode == 0,
                "Clean packaged-source import failed: " +
                (completed.stderr.strip() or completed.stdout.strip() or "unknown error"))


def build_archive(output: Path, files: dict[str, BundleFile], manifest: dict) -> tuple[Path, Path]:
    records = [file_record(files[name]) for name in sorted(files)]
    manifest["files"] = records
    manifest["file_count"] = len(records)
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    checksums = [(record["sha256"], record["path"]) for record in records]
    checksums.append((hashlib.sha256(manifest_payload).hexdigest(), "manifest.json"))
    checksum_payload = ("".join(f"{digest}  {path}\n" for digest, path in checksums)).encode()

    manifest_sidecar = output.with_suffix(".manifest.json")
    checksum_sidecar = output.with_suffix(".SHA256SUMS")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9, allowZip64=True) as bundle:
            for name in sorted(files):
                write_member(bundle, files[name])
            bundle.writestr(zip_info("manifest.json"), manifest_payload, compresslevel=9)
            bundle.writestr(zip_info("SHA256SUMS"), checksum_payload, compresslevel=9)
        with ZipFile(temporary) as bundle:
            corrupt = bundle.testzip()
            require(corrupt is None, f"Archive CRC check failed at {corrupt}")
            names = bundle.namelist()
            require(len(names) == len(set(names)) and
                    all(name.startswith("swarm_complete/") and ".." not in PurePosixPath(name).parts
                        for name in names), "Archive contains duplicate or nonportable paths")
            clean_reproduction_check(bundle, records)
        os.replace(temporary, output)
        atomic_write(manifest_sidecar, manifest_payload)
        sidecar_payload = (
            f"{sha256(output)}  {output.name}\n"
            f"{sha256(manifest_sidecar)}  {manifest_sidecar.name}\n"
        ).encode()
        atomic_write(checksum_sidecar, sidecar_payload)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_sidecar, checksum_sidecar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=ROOT / "output/swarm/policy")
    parser.add_argument("--proof-dir", type=Path, default=ROOT / "output/swarm/proof")
    parser.add_argument("--top-project", type=Path, default=ROOT / "videos/swarm-top")
    parser.add_argument("--side-project", type=Path, default=ROOT / "videos/swarm-side")
    parser.add_argument("--output", type=Path, default=ROOT / "output/swarm/swarm_complete.zip")
    args = parser.parse_args(argv)
    args.policy_dir = args.policy_dir.resolve()
    args.proof_dir = args.proof_dir.resolve()
    args.top_project = args.top_project.resolve()
    args.side_project = args.side_project.resolve()
    args.output = args.output.resolve()
    if args.output.suffix.lower() != ".zip":
        parser.error("--output must end in .zip")
    try:
        training, report, verification, metadata, proof_hashes = validate_policy(
            args.policy_dir, args.proof_dir)
        files, manifest = collect_files(args, training, report, verification, metadata, proof_hashes)
        manifest_path, checksums_path = build_archive(args.output, files, manifest)
    except (BundleError, OSError, subprocess.CalledProcessError) as error:
        print(f"package_swarm: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "archive": str(args.output),
        "manifest": str(manifest_path),
        "checksums": str(checksums_path),
        "files": manifest["file_count"],
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "crc_check": "passed",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
