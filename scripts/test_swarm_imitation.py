#!/usr/bin/env python3
"""Check DAgger state labeling, role weights, aggregation, and saved exploration."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from train_swarm_policy import (
    collect_demonstrations, configure_initial_action_std, dagger_teacher_probability,
    demo_weights, demonstration_errors, fit_demonstrations, merge_demonstrations,
)


class ToyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([.3, .1, -.2]))
        self.log_std = torch.nn.Parameter(torch.full((3,), -3.2))
        self.max_batch = 0

    def mean(self, obs):
        self.max_batch = max(self.max_batch, len(obs["local"]))
        return self.bias.expand(*obs["local"].shape[:-1], 3)


class ToyWorlds:
    def __init__(self):
        self.actions = []
        self.active = torch.tensor([[1., 1.], [1., 0.]])

    def observation(self):
        local = torch.zeros(2, 2, 32)
        local[..., 0] = self.position
        local[:, 0, 15] = 1
        return {"local": local, "active": self.active,
                "phase": torch.zeros(2, 2, dtype=torch.long),
                "learning_weight": torch.tensor([[4., 1.], [4., 0.]]),
                "global": torch.zeros(2, 72)}

    def reset(self):
        self.position = torch.zeros(2, 2)
        self.age = torch.zeros(2, dtype=torch.long)
        return self.observation()

    def teacher_action(self):
        return torch.stack(((.25 - self.position).clamp(-1, 1),
                            torch.full_like(self.position, .2), torch.ones_like(self.position)), -1)

    def step(self, action):
        self.actions.append(action.clone())
        self.position += action[..., 0]
        self.age += 1
        done = self.age >= 2
        return (self.observation(), torch.zeros(2, 2), done, torch.zeros(2, dtype=torch.bool),
                {"success": torch.tensor([False, True]), "delivered": torch.tensor([0, 1])})

    def reset_done(self, done):
        self.position[done] = 0
        self.age[done] = 0
        return self.observation()


def args():
    return SimpleNamespace(demo_noise=.01, balance_demo_roles=True, bc_batch_size=3, max_grad_norm=.5)


class ImitationTests(unittest.TestCase):
    def test_carrier_phases_and_roles_have_equal_mass(self):
        local = torch.zeros(12, 42)
        local[:4, 15] = 1
        phase = torch.tensor([0, 1, 1, 1] + [0] * 8)
        data = {"local": local, "phase": phase,
                "learning_weight": torch.tensor([4.] * 4 + [1.] * 8)}
        weight = demo_weights(data, True)
        self.assertAlmostEqual(float(weight[:4].sum()), float(weight[4:].sum()), places=5)
        self.assertAlmostEqual(float(weight[0]), float(weight[1:4].sum()), places=5)
        self.assertAlmostEqual(float(weight.mean()), 1, places=6)
        legacy = data["learning_weight"] / torch.tensor([9., 3., 3., 3.] + [9.] * 8)
        self.assertTrue(torch.allclose(demo_weights(data), legacy / legacy.mean()))
        carriers_only = {key: value[:4] for key, value in data.items()}
        self.assertTrue(torch.isfinite(demo_weights(carriers_only, True)).all())
        errors = demonstration_errors(
            ToyActor(), {"obs": {"local": local, "phase": phase},
                         "target": torch.zeros(12, 3), "weight": weight}, 3)
        self.assertEqual(set(errors["groups"]), {"all", "formation", "carrier_phase_0", "carrier_phase_1"})

    def test_learner_visited_states_receive_teacher_labels(self):
        model, env = ToyActor(), ToyWorlds()
        with patch("train_swarm_policy.progress"):
            data = collect_demonstrations(model, env, args(), "cpu", 4, teacher_prob=0, dagger_round=1)
        self.assertEqual(len(data["target"]), 12)
        self.assertNotIn("global", data["obs"])
        expected = (.25 - data["obs"]["local"][:, 0]).clamp(-1, 1)
        self.assertTrue(torch.equal(data["target"][:, 0], expected))
        for action in env.actions:
            self.assertTrue(torch.equal(action, torch.tanh(model.bias).expand(2, 2, 3) * env.active[..., None]))
        report = data["report"]
        self.assertEqual(report["teacher_world_steps"], 0)
        self.assertEqual(report["learner_world_steps"], 8)
        self.assertEqual(report["completed_episodes"], 4)
        self.assertEqual(report["success_rate"], .5)
        self.assertEqual(report["mean_delivered"], .5)
        self.assertEqual(report["action_noise_std"], 0)
        self.assertFalse(data["target"].requires_grad)

    def test_teacher_mixture_is_per_world_and_schedule_ends_at_zero(self):
        model, env = ToyActor(), ToyWorlds()
        draws = torch.tensor([[[.1]], [[.9]]])
        with patch("train_swarm_policy.progress"), patch("torch.rand", return_value=draws):
            data = collect_demonstrations(model, env, args(), "cpu", 1, teacher_prob=.5, dagger_round=1)
        self.assertTrue(torch.equal(env.actions[0][0], data["target"][:2]))
        self.assertTrue(torch.equal(env.actions[0][1, 0], torch.tanh(model.bias)))
        self.assertEqual(data["report"]["teacher_world_steps"], 1)
        self.assertEqual(data["report"]["learner_world_steps"], 1)
        self.assertEqual([dagger_teacher_probability(i, 3, .6) for i in range(3)], [.6, .3, 0])
        self.assertEqual(dagger_teacher_probability(0, 1, 1), 0)

    def test_merge_keeps_labels_aligned_and_caps_without_randomness(self):
        def dataset(start):
            local = torch.zeros(5, 32)
            local[:, 0] = torch.arange(start, start + 5)
            return {"obs": {"local": local, "phase": torch.zeros(5, dtype=torch.long)},
                    "target": local[:, :3].clone()}
        rng = torch.random.get_rng_state()
        merged = merge_demonstrations(dataset(0), dataset(5), True, limit=4)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertEqual(merged["target"][:, 0].tolist(), [0., 3., 6., 9.])
        self.assertTrue(torch.equal(merged["target"], merged["obs"]["local"][:, :3]))
        self.assertEqual(merged["report"]["samples_before_cap"], 10)

    def test_fit_reports_action_errors_in_bounded_batches(self):
        model = ToyActor()
        with patch("train_swarm_policy.progress"):
            data = collect_demonstrations(model, ToyWorlds(), args(), "cpu", 4, 0, 1)
            before = model.bias.detach().clone()
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            report = fit_demonstrations(model, optimizer, data, args(), 2)
        self.assertFalse(torch.equal(before, model.bias))
        self.assertLessEqual(model.max_batch, 3)
        self.assertEqual(report["fitting_errors"]["groups"]["all"]["examples"], 12)
        self.assertIn("carrier_phase_0", report["fitting_errors"]["groups"])
        self.assertIn("formation", report["fitting_errors"]["groups"])
        self.assertEqual(len(report["fitting_errors"]["groups"]["all"]["per_action_mse"]), 3)
        self.assertAlmostEqual(report["final_mse"], demonstration_errors(model, data, 2)["weighted_mse"], places=6)

    def test_initial_std_override_never_changes_resumed_std(self):
        model = ToyActor()
        default = configure_initial_action_std(model, None)
        self.assertEqual(default["source"], "model_default")
        self.assertAlmostEqual(default["actual_action_std"][0], .0407622)
        fresh = configure_initial_action_std(model, .01)
        self.assertTrue(fresh["applied"])
        saved = model.log_std.detach().clone()
        resumed = configure_initial_action_std(model, .2, resumed=True)
        self.assertEqual(resumed["source"], "checkpoint")
        self.assertFalse(resumed["applied"])
        self.assertTrue(torch.equal(model.log_std, saved))


if __name__ == "__main__":
    unittest.main()
