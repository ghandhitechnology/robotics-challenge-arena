#!/usr/bin/env python3
"""Check held-out sampling when vectorized worlds finish at different times."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from train_swarm_policy import evaluate, full_stage_validation_pass


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


if __name__ == "__main__":
    unittest.main()
