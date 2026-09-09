#!/usr/bin/env python3
"""Tamper checks for swarm proof artifacts and full-trace action verification."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_mujoco.swarm_policy import NumpySwarmPolicy, PolicyConfig, export_policy, make_actor_critic
from verify_swarm_proof import audit_neural_actions, audit_timing, audit_transport, digest, verify_proof


def unit_checks():
    times = np.arange(501) * .02
    x = .2 + .19 * np.clip((times - 1) / 3, 0, 1)
    clearance = .008 * np.minimum(np.clip((times-.5)/.5, 0, 1), np.clip((4.3-times)/.3, 0, 1))
    positions = np.stack([x, np.full_like(x, .2), .01+clearance], -1)[:, None]
    phases = np.zeros((len(times), 1), dtype=int)
    for phase, start in enumerate((.5, 1, 4, 4.3, 4.6), start=1):
        phases[times >= start] = phase
    pads = np.tile(((times >= .5) & (times < 4.3))[:, None], (1, 2)).astype(np.uint8)
    floor = (clearance < 1e-8)[:, None].astype(np.uint8)
    goals = np.array([[.39, .2]])
    errors = []
    audit_transport(times, positions, clearance[:, None], pads, floor, phases, goals, 5., .15, errors)
    assert not errors, errors
    drifted = positions.copy()
    drifted[-1, 0, 1] += .003
    errors = []
    audit_transport(times, drifted, clearance[:, None], pads, floor, phases, goals, 5., .15, errors)
    assert any("drifts more than 2 mm" in error for error in errors), errors
    errors = []
    assert not audit_timing(times[:-1], np.zeros((500, 14)), times[:-1], np.zeros((500, 2, 3)), 14, 2, .02, errors)
    assert any("Incomplete recording" in error for error in errors)
    torch.manual_seed(21)
    model = make_actor_critic(PolicyConfig(local_dim=32, global_dim=72, mirror=True))
    rng = np.random.default_rng(15)
    trace = {"local": rng.normal(size=(257, 2, 32)).astype(np.float32),
             "neighbors": rng.normal(size=(257, 2, 1, 8)).astype(np.float32),
             "neighbor_mask": np.ones((257, 2, 1), dtype=bool), "active": np.ones((257, 2), dtype=np.float32)}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "weights.npz"
        export_policy(model, path)
        policy = NumpySwarmPolicy(path)
        trace["actions"] = policy(trace)
        errors = []
        checked = audit_neural_actions(policy, trace, errors, chunk_size=128)
        assert not errors and checked["neural_action_rows_recomputed"] == 257
        forged = dict(trace, actions=trace["actions"].copy())
        forged["actions"][-1, -1, 0] += .01
        errors = []
        audit_neural_actions(policy, forged, errors, chunk_size=128)
        assert any("Neural actions disagree" in error for error in errors), errors
        with torch.no_grad():
            model.actor.bias[2] += .3
        export_policy(model, path)
        errors = []
        audit_neural_actions(NumpySwarmPolicy(path), trace, errors, chunk_size=128)
        assert any("Neural actions disagree" in error for error in errors), errors
    return ["valid geometric transport", "final hold drift", "partial recording", "last action beyond chunk boundary", "changed model weights"]


def artifact_checks(fixture):
    original = verify_proof(fixture, allow_teacher=True)
    assert original["valid"] and not original["final_neural_proof"], original["errors"]
    results = []
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        def fresh(name):
            path = directory / name
            shutil.copytree(fixture, path)
            return path
        def update_hash(path, name):
            report = json.loads((path / "report.json").read_text())
            report["artifacts"][name] = digest(path / name)
            (path / "report.json").write_text(json.dumps(report))
        path = fresh("hash")
        with (path / "scene.xml").open("a") as stream:
            stream.write("\n")
        result = verify_proof(path, allow_teacher=True)
        assert not result["valid"] and any("Artifact hash mismatch" in error for error in result["errors"])
        results.append("artifact hash changed")
        path = fresh("partial")
        with np.load(path / "trajectory.npz") as archive:
            arrays = {key: archive[key][:-1] for key in archive.files}
        np.savez_compressed(path / "trajectory.npz", **arrays)
        update_hash(path, "trajectory.npz")
        result = verify_proof(path, allow_teacher=True)
        assert not result["valid"] and any("Incomplete recording" in error for error in result["errors"])
        results.append("partial recording with refreshed hash")
        path = fresh("motor")
        xml = ET.parse(path / "scene.xml")
        xml.find("./actuator/motor").set("forcerange", "-2 2")
        xml.write(path / "scene.xml")
        update_hash(path, "scene.xml")
        result = verify_proof(path, allow_teacher=True)
        assert not result["valid"] and any("wheel torque limit" in error for error in result["errors"])
        results.append("motor limit forgery with refreshed hash")
        path = fresh("drift")
        with np.load(path / "trajectory.npz") as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
        adr = model.joint("payload_00_free").qposadr[0]
        arrays["qpos"][-1, adr] += .003
        np.savez_compressed(path / "trajectory.npz", **arrays)
        update_hash(path, "trajectory.npz")
        result = verify_proof(path, allow_teacher=True)
        assert not result["valid"] and any("drifts more than 2 mm" in error for error in result["errors"])
        results.append("final hold drift with refreshed hash")
        result = verify_proof(fixture)
        assert not result["valid"] and any("Final proof requires a neural policy" in error for error in result["errors"])
        results.append("teacher cannot pass final neural gate")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="Successful current-source teacher fixture for artifact tamper cases")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checks = unit_checks()
    if args.fixture:
        checks.extend(artifact_checks(args.fixture))
    print(json.dumps({"passed": True, "checks": checks, "artifact_fixture_tested": args.fixture is not None}))


if __name__ == "__main__":
    main()
