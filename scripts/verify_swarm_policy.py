#!/usr/bin/env python3
"""Check physical reflection symmetry and the portable actor export."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.swarm_env import FEATURES as TRANSPORT_FEATURES
from arena_mujoco.swarm_body_env import FEATURES as BODY_FEATURES
from arena_mujoco.swarm_policy import (
    NumpySwarmPolicy, PolicyConfig, export_policy, make_actor_critic, mirror_observation,
)


CONTRACTS = (
    {"name": "transport", "local_dim": 32, "neighbor_dim": 8,
     "global_dim": 72, "action_dim": 3, "features": TRANSPORT_FEATURES},
    {"name": "magnetic_body_legacy", "local_dim": 40, "neighbor_dim": 12,
     "global_dim": 88, "action_dim": 4, "features": BODY_FEATURES[:-2]},
    {"name": "magnetic_body", "local_dim": 42, "neighbor_dim": 12,
     "global_dim": 92, "action_dim": 4, "features": BODY_FEATURES},
)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(417)
    lateral = {"target_right", "heading_sin", "object_right", "goal_right",
               "velocity_right", "yaw_velocity", "axis_right", "object_velocity_right",
               "body_centroid_right", "body_goal_right"}
    assert BODY_FEATURES[:len(TRANSPORT_FEATURES)] == TRANSPORT_FEATURES
    assert BODY_FEATURES[len(TRANSPORT_FEATURES):] == [
        "body_centroid_right", "body_centroid_forward", "body_goal_right", "body_goal_forward",
        "magnets_enabled", "magnetic_degree", "magnetic_component_fraction", "body_phase",
        "body_approach_stage", "docking_stage",
    ]
    max_export_error = 0.
    checked = {}
    legacy_obs = None
    for contract in CONTRACTS:
        features = contract["features"]
        assert len(features) == contract["local_dim"]
        for robot_count in (2, 8, 40):
            k = min(6, robot_count - 1)
            obs = {"local": torch.randn(2, robot_count, contract["local_dim"]),
                   "neighbors": torch.randn(2, robot_count, k, contract["neighbor_dim"]),
                   "neighbor_mask": torch.rand(2, robot_count, k) > .3,
                   "active": torch.ones(2, robot_count),
                   "global": torch.randn(2, contract["global_dim"])}
            obs["neighbor_mask"][0, 0] = False
            obs["active"][0, -1] = 0
            original = {"local": obs["local"].clone(), "neighbors": obs["neighbors"].clone()}
            mirrored = mirror_observation(obs)
            # Check coordinate semantics independently of the actor's transform.
            for i, feature in enumerate(features):
                source = features.index("previous_right" if feature == "previous_left" else
                                        "previous_left" if feature == "previous_right" else feature)
                expected = obs["local"][..., source] * (-1 if feature in lateral else 1)
                assert torch.equal(mirrored["local"][..., i], expected), feature
            neighbor_sign = torch.ones(contract["neighbor_dim"])
            neighbor_sign[[0, 2, 4]] = -1
            assert torch.equal(mirrored["neighbors"], obs["neighbors"] * neighbor_sign)
            assert torch.equal(obs["local"], original["local"]), "Reflection mutated local features"
            assert torch.equal(obs["neighbors"], original["neighbors"]), "Reflection mutated neighbors"
            assert mirrored["global"] is obs["global"]
            assert mirrored["neighbor_mask"] is obs["neighbor_mask"]
            assert torch.equal(mirror_observation(mirrored)["local"], obs["local"])
            assert torch.equal(mirror_observation(mirrored)["neighbors"], obs["neighbors"])
            config = PolicyConfig(local_dim=contract["local_dim"], neighbor_dim=contract["neighbor_dim"],
                                  global_dim=contract["global_dim"], action_dim=contract["action_dim"],
                                  mirror=True)
            model = make_actor_critic(config)
            action = model.act(obs, deterministic=True)[0]
            action_order = [1, 0, *range(2, contract["action_dim"])]
            reflected_action = model.act(mirrored, deterministic=True)[0]
            assert torch.equal(reflected_action, action[..., action_order]), "Reflection action order is not exact"
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
                assert np.array_equal(policy(mirror_observation(numpy_obs)), actual[..., action_order])
                assert np.all(actual[0, -1] == 0), "NumPy policy did not mask inactive actions"
            if contract["name"] == "transport":
                legacy_obs = obs
            checked[contract["name"]] = {key: contract[key] for key in
                                         ("local_dim", "neighbor_dim", "global_dim", "action_dim")}

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
        numpy_obs = {key: value.numpy() for key, value in legacy_obs.items()}
        with torch.no_grad():
            assert np.allclose(legacy(numpy_obs), model.act(legacy_obs, deterministic=True)[0].numpy(), atol=1e-6)
    print(json.dumps({"reflection_exact": True, "robot_counts": [2, 8, 40],
                      "contracts": checked, "numpy_max_error": max_export_error,
                      "legacy_export_compatible": True}))


if __name__ == "__main__":
    main()
