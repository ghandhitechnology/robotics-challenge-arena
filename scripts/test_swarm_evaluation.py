#!/usr/bin/env python3
"""Check held-out sampling when vectorized worlds finish at different times."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from evaluate_swarm_training import (ARTIFACTS, verify_immutable_inputs,
                                       verify_training_runtime)
from train_swarm_policy import (ROOT, TRAINING_SOURCES, acceptance_passed, evaluate,
                                file_sha256, full_stage_validation_pass,
                                training_source_provenance)


class AsyncWorlds:
    """Independent episodes with controlled durations and terminal outcomes."""

    def __init__(self, durations, outcomes):
        self.durations = np.asarray(durations)
        self.outcomes = outcomes
        self.max_steps = max(durations)
        self.active = np.ones((len(durations), 1), dtype=np.float32)

    def set_curriculum(self, spec):
        pass

    def reset(self):
        self.age = np.zeros(len(self.durations), dtype=int)
        self.episodes = np.zeros(len(self.durations), dtype=int)
        self.steps = 0
        return {"active": self.active}

    def step(self, actions):
        self.steps += 1
        self.age += 1
        done = self.age >= self.durations
        success = np.zeros(len(self.durations), dtype=bool)
        for world in np.flatnonzero(done):
            results = self.outcomes[world]
            success[world] = results[min(self.episodes[world], len(results) - 1)]
            self.episodes[world] += 1
        return ({"active": self.active}, np.ones_like(self.active),
                done & success, done & ~success,
                {"success": success, "delivered": success.astype(float)})

    def reset_done(self, done):
        self.age[np.asarray(done)] = 0
        return {"active": self.active}


class EvaluationSamplingTests(unittest.TestCase):
    def run_evaluation(self, env, episodes, max_steps=0):
        args = SimpleNamespace(eval_episodes=episodes, stage_eval_episodes=episodes,
                               eval_max_steps=max_steps, robots=40, objects=4)
        model = SimpleNamespace(config=SimpleNamespace(action_dim=3))
        with patch("train_swarm_policy.progress"):
            return evaluate(model, env, 3, args, "cpu", policy="zero")

    def test_slow_failures_cannot_be_replaced_by_fast_successes(self):
        env = AsyncWorlds([1] * 4 + [10] * 4, [[1]] * 4 + [[0]] * 4)
        result = self.run_evaluation(env, 32)
        self.assertTrue(result["complete"])
        self.assertEqual(result["episode_quotas"], [4] * 8)
        self.assertEqual(result["episodes_per_world"], [4] * 8)
        self.assertEqual(result["episodes"], 32)
        self.assertEqual(result["success_rate"], .5)
        self.assertEqual(result["mean_return"], 5.5)
        self.assertEqual(env.steps, 40)
        self.assertFalse(full_stage_validation_pass(result, 3, .8))

    def test_only_first_quota_episodes_contribute(self):
        env = AsyncWorlds([1, 10], [[0, 0, 1], [0]])
        result = self.run_evaluation(env, 4)
        self.assertEqual(env.episodes.tolist(), [20, 2])
        self.assertEqual(result["episodes_per_world"], [2, 2])
        self.assertEqual(result["success_rate"], 0)

    def test_nondivisible_episode_count_is_balanced(self):
        env = AsyncWorlds([1, 2, 7], [[1], [0], [1]])
        result = self.run_evaluation(env, 5)
        self.assertTrue(result["complete"])
        self.assertEqual(result["episode_quotas"], [2, 2, 1])
        self.assertEqual(result["episodes_per_world"], [2, 2, 1])
        self.assertEqual(result["success_rate"], .6)
        self.assertAlmostEqual(result["mean_return"], 2.6)
        self.assertEqual(env.steps, 7)

    def test_worlds_with_zero_quota_do_not_delay_completion(self):
        env = AsyncWorlds([2, 3, 100, 100], [[1]] * 4)
        result = self.run_evaluation(env, 2)
        self.assertTrue(result["complete"])
        self.assertEqual(result["episodes_per_world"], [1, 1, 0, 0])
        self.assertEqual(env.steps, 3)

    def test_timeout_cannot_pass_with_unfinished_quotas(self):
        env = AsyncWorlds([1, 100], [[1], [0]])
        result = self.run_evaluation(env, 32, max_steps=20)
        self.assertFalse(result["complete"])
        self.assertEqual(result["episodes_per_world"], [16, 0])
        self.assertEqual(result["success_rate"], 1)
        self.assertFalse(full_stage_validation_pass(result, 3, .8))


class AcceptanceTests(unittest.TestCase):
    def result(self, **overrides):
        result = {
            "complete": True,
            "episodes": 32,
            "success_rate": .8,
            "success_wilson_lower_95": .6,
        }
        result.update(overrides)
        return result

    def zero_result(self, **overrides):
        result = {"complete": True, "episodes": 32, "success_rate": .5}
        result.update(overrides)
        return result

    def test_accepts_complete_full_scale_heldout_improvement(self):
        args = SimpleNamespace(cpu_smoke=False, robots=40, objects=2,
                               final_success=.8)

        self.assertTrue(acceptance_passed(
            args, 3, 1.000001e-6, self.result(), self.zero_result()))

    def test_rejects_each_failed_acceptance_condition(self):
        valid_args = {"cpu_smoke": False, "robots": 40, "objects": 2,
                      "final_success": .8}
        cases = [
            ("cpu smoke", {"cpu_smoke": True}, 3, 2e-6, {}, {}),
            ("wrong robot count", {"robots": 39}, 3, 2e-6, {}, {}),
            ("too few objects", {"objects": 1}, 3, 2e-6, {}, {}),
            ("not final stage", {}, 2, 2e-6, {}, {}),
            ("actor update at floor", {}, 3, 1e-6, {}, {}),
            ("incomplete heldout", {}, 3, 2e-6, {"complete": False}, {}),
            ("too few heldout episodes", {}, 3, 2e-6, {"episodes": 31}, {}),
            ("below final success", {}, 3, 2e-6, {"success_rate": .79}, {}),
            ("below Wilson floor", {}, 3, 2e-6,
             {"success_wilson_lower_95": .599999}, {}),
            ("incomplete zero baseline", {}, 3, 2e-6, {}, {"complete": False}),
            ("too few zero episodes", {}, 3, 2e-6, {}, {"episodes": 31}),
            ("only matches zero margin", {}, 3, 2e-6,
             {"success_rate": .8}, {"success_rate": .6}),
        ]

        for name, arg_changes, stage, actor_update, final_changes, zero_eval in cases:
            with self.subTest(name=name):
                args = SimpleNamespace(**(valid_args | arg_changes))
                self.assertFalse(acceptance_passed(
                    args, stage, actor_update, self.result(**final_changes),
                    self.zero_result(**zero_eval)))

    def test_fixed_success_floor_cannot_be_lowered_by_argument(self):
        args = SimpleNamespace(cpu_smoke=False, robots=40, objects=2,
                               final_success=.7)

        self.assertFalse(acceptance_passed(
            args, 3, 2e-6, self.result(success_rate=.79),
            self.zero_result(success_rate=.5)))


class DeferredEvaluationArtifactTests(unittest.TestCase):
    def test_provenance_records_every_training_source_hash(self):
        provenance = training_source_provenance()

        self.assertEqual(set(provenance["files"]), set(TRAINING_SOURCES))
        for name in TRAINING_SOURCES:
            self.assertEqual(provenance["files"][name], file_sha256(ROOT / name))

    def test_detects_artifact_changes(self):
        sources = {"commit": "source-head", "files": {"source.py": "abc"}}
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ARTIFACTS:
                (directory / name).write_bytes(name.encode())
            hashes = {name: file_sha256(directory / name) for name in ARTIFACTS}
            pending = {
                "source_commit": sources["commit"],
                "source_hashes": sources["files"],
                "artifact_hashes": hashes,
                "weights_sha256": hashes["weights.npz"],
            }
            with patch("evaluate_swarm_training.training_source_provenance",
                       return_value=sources):
                verify_immutable_inputs(directory, pending)
                (directory / "weights.npz").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "artifacts changed"):
                    verify_immutable_inputs(directory, pending)

    def test_runtime_accepts_h100_and_requires_matching_mujoco(self):
        pending = {"gpu": "NVIDIA H100 80GB HBM3", "cuda_optimization": True,
                   "policy_device": "cuda:0", "cuda": "12.8",
                   "mujoco": mujoco.__version__}
        verify_training_runtime(pending)
        pending["mujoco"] = "different"
        with self.assertRaisesRegex(ValueError, "MuJoCo version"):
            verify_training_runtime(pending)


if __name__ == "__main__":
    unittest.main()
