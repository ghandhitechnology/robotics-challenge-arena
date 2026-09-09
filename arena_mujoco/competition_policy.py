"""Goal feedback policy. The mission planner supplies geometric subgoals.

The trained, odd-symmetric network controls forward/yaw and both manipulator
velocities. Symmetry makes the zero-error action exactly zero, including after
NumPy export, which matters for a sample with 2 mm radial insertion clearance.
"""
from pathlib import Path
import numpy as np

ERROR_SCALES = np.array([.055, .35, .012, .006])
VELOCITY_LIMITS = np.array([.22, 1.6, .035, .025])
OBSERVATION_NAMES = ['forward_error_m', 'heading_error_rad', 'lift_error_m', 'jaw_error_m']
ACTION_NAMES = ['forward_m_s', 'yaw_rad_s', 'lift_m_s', 'jaw_m_s']


def observation(errors):
    return np.clip(np.asarray(errors, dtype=float) / ERROR_SCALES, -8, 8)


def expert_action(obs):
    return np.tanh(obs)


class NeuralPolicy:
    def __init__(self, path):
        self.path = str(path)
        with np.load(path) as archive:
            self.weights = [archive[f'w{i}'].astype(float) for i in range(3)]
        if [w.shape for w in self.weights] != [(48,4), (48,48), (4,48)]:
            raise ValueError('Unexpected policy architecture')
        self.calls = 0

    def __call__(self, obs):
        x = np.asarray(obs, dtype=float)
        for weight in self.weights:
            x = np.tanh(x @ weight.T)
        self.calls += 1
        return x


class ExpertPolicy:
    calls = 0
    def __call__(self, obs):
        self.calls += 1
        return expert_action(obs)


class ZeroPolicy:
    calls = 0
    def __call__(self, obs):
        self.calls += 1
        return np.zeros(4)
