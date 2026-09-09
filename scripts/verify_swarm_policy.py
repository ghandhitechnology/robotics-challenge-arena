#!/usr/bin/env python3
"""Check physical reflection symmetry and the portable actor export."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.swarm_env import FEATURES
from arena_mujoco.swarm_policy import (
    NumpySwarmPolicy, PolicyConfig, export_policy, make_actor_critic, mirror_observation,
)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(417)
    lateral = {"target_right", "heading_sin", "object_right", "goal_right",
               "velocity_right", "yaw_velocity", "axis_right", "object_velocity_right"}
    max_export_error = 0.
    for robot_count in (2, 8, 40):
        k = min(6, robot_count - 1)
        obs = {"local": torch.randn(2, robot_count, 32),
               "neighbors": torch.randn(2, robot_count, k, 8),
               "neighbor_mask": torch.rand(2, robot_count, k) > .3,
               "active": torch.ones(2, robot_count), "global": torch.randn(2, 72)}
        obs["neighbor_mask"][0, 0] = False
        obs["active"][0, -1] = 0
        original = obs["local"].clone()
        mirrored = mirror_observation(obs)
        # Check coordinate semantics independently of the actor's transform.
        for i, feature in enumerate(FEATURES):
            source = FEATURES.index("previous_right" if feature == "previous_left" else
                                    "previous_left" if feature == "previous_right" else feature)
            expected = obs["local"][..., source] * (-1 if feature in lateral else 1)
            assert torch.equal(mirrored["local"][..., i], expected), feature
        assert torch.equal(mirrored["neighbors"], obs["neighbors"] * torch.tensor([-1, 1, -1, 1, -1, 1, 1, 1]))
        assert torch.equal(obs["local"], original), "Reflection mutated the original observation"
        assert mirrored["global"] is obs["global"]
        assert mirrored["neighbor_mask"] is obs["neighbor_mask"]
        assert torch.equal(mirror_observation(mirrored)["local"], obs["local"])
        model = make_actor_critic(PolicyConfig(local_dim=32, global_dim=72, mirror=True))
        action = model.act(obs, deterministic=True)[0]
        reflected_action = model.act(mirrored, deterministic=True)[0]
        assert torch.equal(reflected_action, action[..., [1, 0, 2]]), "Wheel exchange symmetry is not exact"
        assert torch.isfinite(action).all() and torch.all(action[0, -1] == 0)
        loss = (action - .1).square().mean()
        loss.backward()
        assert torch.isfinite(model.actor.weight.grad).all() and model.actor.weight.grad.abs().sum() > 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            export_policy(model, path)
            policy = NumpySwarmPolicy(path)
            numpy_obs = {key: value.numpy() for key, value in obs.items()}
            actual = policy(numpy_obs)
            error = float(np.max(np.abs(actual - action.detach().numpy())))
            max_export_error = max(max_export_error, error)
            assert error < 1e-6, f"Torch/NumPy export discrepancy: {error}"
            assert np.array_equal(policy(mirror_observation(numpy_obs)), actual[..., [1, 0, 2]])

    # Existing exports omitted the mirror field. Their original behavior stays valid.
    model = make_actor_critic(PolicyConfig(local_dim=32, global_dim=72))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "legacy.npz"
        export_policy(model, path)
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        config = json.loads(str(arrays["config"]))
        del config["mirror"]
        arrays["config"] = np.array(json.dumps(config))
        np.savez_compressed(path, **arrays)
        legacy = NumpySwarmPolicy(path)
        assert not legacy.config.mirror
        with torch.no_grad():
            assert np.allclose(legacy(numpy_obs), model.act(obs, deterministic=True)[0].numpy(), atol=1e-6)
    print(json.dumps({"reflection_exact": True, "robot_counts": [2, 8, 40],
                      "numpy_max_error": max_export_error, "legacy_export_compatible": True}))


if __name__ == "__main__":
    main()
