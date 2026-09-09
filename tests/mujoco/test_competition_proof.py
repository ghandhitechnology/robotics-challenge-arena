"""Artifact-consistency tests use synthetic poses, not a simulated mission."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from arena_mujoco.competition_events import initialize_competition_events
from arena_mujoco.competition_policy import ACTION_NAMES, ERROR_SCALES, OBSERVATION_NAMES, VELOCITY_LIMITS
from arena_mujoco.competition_scoring import score_competition
from arena_mujoco.competition_runtime import CompetitionSimulation
from scripts.verify_competition_proof import ROOT, digest, verify_proof


class CompetitionProofTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.sim = CompetitionSimulation(tape_mode="none")
        sim = self.sim
        self.assertTrue(initialize_competition_events(sim.model, sim.data, sim.metadata)["valid"])
        initial = sim.data.qpos.copy()
        groups = {}
        for entry in sim.metadata["objects"]:
            groups.setdefault(entry["category"], []).append(entry["name"])

        def place(name, xyz):
            address = sim.model.joint(f"{name}_free").qposadr[0]
            sim.data.qpos[address:address + 3] = xyz

        place("competition_robot", [.55, .59, .02])
        for name, y in zip(groups["sample"], (.3905, .4905, .5905)):
            place(name, [1.068, y, .0025])
        for name, xy in zip(groups["medical_kit"], ((.05, .5), (.1, .5), (.05, .1), (.05, 1.0))):
            place(name, [*xy, .01])
        for name, x in zip(groups["patient"][:3], (.04, .09, .14)):
            place(name, [x, .4, .01])
        for name, xy in zip(groups["observation"][:3], ((.09, .1), (.14, .1), (.1, 1.0))):
            place(name, [*xy, .01])
        for name, x in zip(groups["low_risk"][:3], (1.02, 1.07, 1.12)):
            place(name, [x, 1.0, .01])
        sim.data.time = 1.
        score = score_competition(sim.model, sim.data, sim.metadata)
        self.assertTrue(score["success"])
        path = self.directory
        (path / "scene.xml").write_text(sim.xml)
        sim.metadata["competition_events"]["last_observed_time_s"] = 6.
        (path / "metadata.json").write_text(json.dumps(sim.metadata))
        for name, time in (("declaration_state", 1.), ("final_state", 6.)):
            np.savez_compressed(path / f"{name}.npz", qpos=sim.data.qpos, qvel=sim.data.qvel,
                                ctrl=sim.data.ctrl, time=time)
        times = np.arange(1, 61) / 10
        poses = np.tile(sim.data.qpos, (len(times), 1))
        poses[times < 1.] = initial
        np.savez_compressed(path / "trajectory.npz", times=times, qpos=poses, initial_qpos=initial)
        self.weights = path / "weights.npz"
        np.savez_compressed(self.weights, w0=np.zeros((48, 4)), w1=np.zeros((48, 48)), w2=np.zeros((4, 48)))
        np.savez_compressed(path / "policy_trace.npz", time_s=np.arange(1, 301) * .02,
                            observation=np.zeros((300, 4)), action=np.zeros((300, 4)),
                            velocity_limits=VELOCITY_LIMITS, error_scales=ERROR_SCALES,
                            observation_names=OBSERVATION_NAMES, action_names=ACTION_NAMES)
        sources = list((ROOT / "arena_mujoco").glob("*.py")) + [
            ROOT / name for name in ("arena_spec.json", "competition_rules.json", "robot_design.json")]
        self.report = {"success": True, "failure": None, "policy": "neural", "policy_calls": 300,
                       "completed_deliveries": 16, "planned_deliveries": 16,
                       "scene_sha256": digest(path / "scene.xml"),
                       "trajectory_sha256": digest(path / "trajectory.npz"),
                       "weights_sha256": digest(self.weights), "score": score,
                       "completion_seconds": 1., "simulation_seconds": 6., "final_view_valid": True,
                       "source_files_sha256": {str(source.relative_to(ROOT)): digest(source) for source in sources},
                       "final_view_checks": [{"time_s": 1. + i * .5, "task_score": 160, "success": True}
                                             for i in range(1, 11)]}
        self.write_report()

    def write_report(self):
        (self.directory / "report.json").write_text(json.dumps(self.report))

    def verify(self):
        return verify_proof(self.directory, weights=self.weights)

    def rewrite_archive(self, filename, change):
        path = self.directory / filename
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        change(values)
        np.savez_compressed(path, **values)

    def test_consistent_artifacts_pass_without_claiming_physics_replay(self):
        result = self.verify()
        self.assertTrue(result["valid"], result["errors"])
        self.assertFalse(result["physics_replayed"])
        self.assertEqual(result["checks"]["policy_calls_checked"], 300)

    def test_forged_success_cannot_override_declaration_geometry(self):
        address = self.sim.model.joint("Biological_Sample_01_free").qposadr[0]
        self.rewrite_archive("declaration_state.npz", lambda values: values["qpos"].__setitem__(address, .90))
        result = self.verify()
        self.assertFalse(result["valid"])
        self.assertTrue(any("declaration geometry" in error for error in result["errors"]))

    def test_changed_actions_fail_exported_network_check(self):
        self.rewrite_archive("policy_trace.npz", lambda values: values["action"].__setitem__((100, 0), .2))
        result = self.verify()
        self.assertFalse(result["valid"])
        self.assertTrue(any("exported neural network" in error for error in result["errors"]))

    def test_changed_source_hash_or_incomplete_view_fails(self):
        self.report["source_files_sha256"]["arena_mujoco/competition_runtime.py"] = "0" * 64
        self.report["final_view_checks"] = self.report["final_view_checks"][:-1]
        self.write_report()
        result = self.verify()
        self.assertFalse(result["valid"])
        self.assertTrue(any("Source hash mismatch" in error for error in result["errors"]))
        self.assertTrue(any("final-view" in error.lower() for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
