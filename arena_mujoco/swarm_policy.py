"""Shared neighbor-attention policy with a portable NumPy inference path."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


MIRROR_LOCAL_NEGATE = [0, 2, 4, 6, 8, 10, 25, 28]
MIRROR_NEIGHBOR_NEGATE = [0, 2, 4]
MIRROR_ACTION_ORDER = [1, 0, 2]


def mirror_observation(obs):
    """Reflect robot-local lateral coordinates and exchange previous wheels.

    Supports Torch and NumPy. Privileged critic state and neighborhood membership
    are unchanged; only the actor's 32/8-feature input contract is reflected.
    """
    result = dict(obs)
    for key in ("local", "neighbors"):
        value = obs[key]
        result[key] = value.clone() if hasattr(value, "clone") else np.array(value, copy=True)
    result["local"][..., MIRROR_LOCAL_NEGATE] *= -1
    result["local"][..., [22, 23]] = obs["local"][..., [23, 22]]
    result["neighbors"][..., MIRROR_NEIGHBOR_NEGATE] *= -1
    return result


@dataclass
class PolicyConfig:
    local_dim: int = 24
    neighbor_dim: int = 8
    global_dim: int = 32
    action_dim: int = 3
    hidden: int = 64
    heads: int = 2
    mirror: bool = False

    def __post_init__(self):
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if self.mirror and (self.local_dim, self.neighbor_dim, self.action_dim) != (32, 8, 3):
            raise ValueError("Reflection requires the 32 local / 8 neighbor / 3 actuator feature contract")


def make_actor_critic(config):
    """Import Torch only during training, so native evaluation needs NumPy only."""
    import torch
    from torch import nn

    class SwarmActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            h = config.hidden
            self.config = config
            self.local = nn.Linear(config.local_dim, h)
            self.neighbor = nn.Linear(config.neighbor_dim, h)
            self.query = nn.Linear(h, h)
            self.key = nn.Linear(h, h)
            self.message = nn.Linear(h, h)
            self.fusion = nn.Linear(h * 2, h)
            self.actor = nn.Linear(h, config.action_dim)
            # Contact transport commands are small; begin with 4% exploration.
            self.log_std = nn.Parameter(torch.full((config.action_dim,), -3.2))
            self.critic_global = nn.Linear(config.global_dim, h)
            self.critic_local = nn.Linear(config.local_dim, h)
            self.critic_hidden = nn.Linear(h * 2, h)
            self.critic_out = nn.Linear(h, 1)
            nn.init.orthogonal_(self.actor.weight, gain=.01)
            nn.init.zeros_(self.actor.bias)

        def raw_mean(self, obs):
            local = torch.tanh(self.local(obs["local"]))
            neighbor = torch.tanh(self.neighbor(obs["neighbors"]))
            heads, width = config.heads, config.hidden // config.heads
            query = self.query(local).unflatten(-1, (heads, width))
            key = self.key(neighbor).unflatten(-1, (heads, width))
            message = self.message(neighbor).unflatten(-1, (heads, width))
            score = (query.unsqueeze(-3) * key).sum(-1) / width ** .5
            mask = obs["neighbor_mask"].bool().unsqueeze(-1)
            # Multiplication and renormalization keep empty neighborhoods finite.
            weight = torch.softmax(score.masked_fill(~mask, -1e9), dim=-2) * mask
            weight = weight / weight.sum(-2, keepdim=True).clamp_min(1e-8)
            pooled = (weight.unsqueeze(-1) * message).sum(-3).flatten(-2)
            hidden = torch.tanh(self.fusion(torch.cat((local, pooled), dim=-1)))
            return self.actor(hidden)

        def mean(self, obs):
            raw = self.raw_mean(obs)
            if config.mirror:
                reflected = self.raw_mean(mirror_observation(obs))
                raw = .5 * (raw + reflected[..., MIRROR_ACTION_ORDER])
            return raw

        def value(self, obs):
            global_state = torch.tanh(self.critic_global(obs["global"]))
            global_state = global_state.unsqueeze(-2).expand(*obs["local"].shape[:-1], config.hidden)
            local = torch.tanh(self.critic_local(obs["local"]))
            hidden = torch.tanh(self.critic_hidden(torch.cat((local, global_state), dim=-1)))
            return self.critic_out(hidden).squeeze(-1)

        def distribution(self, obs):
            return torch.distributions.Normal(self.mean(obs), self.log_std.clamp(-5, .5).exp())

        def act(self, obs, deterministic=False):
            distribution = self.distribution(obs)
            raw = distribution.mean if deterministic else distribution.sample()
            actions = torch.tanh(raw)
            log_prob = self.log_probability(distribution, raw)
            return actions * obs["active"].unsqueeze(-1), raw, log_prob, self.value(obs)

        @staticmethod
        def log_probability(distribution, raw):
            # Stable log(1 - tanh(x)^2), including saturated commands.
            correction = 2 * (np.log(2) - raw - torch.nn.functional.softplus(-2 * raw))
            return (distribution.log_prob(raw) - correction).sum(-1)

        def evaluate(self, obs, raw):
            distribution = self.distribution(obs)
            return self.log_probability(distribution, raw), distribution.entropy().sum(-1), self.value(obs)

    return SwarmActorCritic()


def export_policy(model, path):
    """Save actor parameters and architecture without pickle or a Torch dependency."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()
              if not key.startswith("critic_")}
    arrays["config"] = np.array(json.dumps(asdict(model.config)))
    np.savez_compressed(path, **arrays)


class NumpySwarmPolicy:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.config = PolicyConfig(**json.loads(str(archive["config"])))
            self.weights = {key: archive[key].astype(np.float32) for key in archive.files if key != "config"}
        self.calls = 0

    def linear(self, name, value):
        return value @ self.weights[f"{name}.weight"].T + self.weights[f"{name}.bias"]

    def raw_mean(self, obs):
        local = np.tanh(self.linear("local", np.asarray(obs["local"], dtype=np.float32)))
        neighbor = np.tanh(self.linear("neighbor", np.asarray(obs["neighbors"], dtype=np.float32)))
        heads, width = self.config.heads, self.config.hidden // self.config.heads
        query = self.linear("query", local).reshape(*local.shape[:-1], heads, width)
        key = self.linear("key", neighbor).reshape(*neighbor.shape[:-1], heads, width)
        message = self.linear("message", neighbor).reshape(*neighbor.shape[:-1], heads, width)
        score = (np.expand_dims(query, -3) * key).sum(-1) / width ** .5
        mask = np.asarray(obs["neighbor_mask"], dtype=bool)[..., None]
        score = np.where(mask, score, -1e9)
        weight = np.exp(score - score.max(axis=-2, keepdims=True)) * mask
        weight /= np.maximum(weight.sum(axis=-2, keepdims=True), 1e-8)
        pooled = (weight[..., None] * message).sum(axis=-3).reshape(*local.shape[:-1], -1)
        hidden = np.tanh(self.linear("fusion", np.concatenate((local, pooled), axis=-1)))
        return self.linear("actor", hidden)

    def __call__(self, obs):
        raw = self.raw_mean(obs)
        if self.config.mirror:
            reflected = self.raw_mean(mirror_observation(obs))
            raw = .5 * (raw + reflected[..., MIRROR_ACTION_ORDER])
        action = np.tanh(raw) * np.asarray(obs["active"])[..., None]
        self.calls += 1
        return action


class ZeroSwarmPolicy:
    def __call__(self, obs):
        return np.zeros((*np.asarray(obs["active"]).shape, 3), dtype=np.float32)
