#!/usr/bin/env python3
"""Reject forged body success, commands, poses, contact graphs and provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_mujoco.swarm_policy import NumpySwarmPolicy, PolicyConfig, export_policy, make_actor_critic
from scripts.run_swarm_body import record_body
from scripts.verify_swarm_body import (
    CONTRACT, ENV_FACTORY, audit_body_task, audit_neural_actions, audit_training,
    digest, make_environment, verify_body,
)


def synthetic_task():
    setup = {"robot_count": 40, "object_count": 2, "seed": 37, "physics_timestep_s": .002,
             "episode_seconds": 70., "difficulty": 1.}
    env, _ = make_environment(setup)
    model, metadata, initial = env.model, env.metadata, env.native_data[0].qpos.copy()
    env.close()
    times = np.arange(501)*.02
    qpos = np.tile(initial, (len(times), 1))
    progress = np.clip((times-1)/3, 0, 1)
    robot_addresses = [model.joint(r['name']+'_free').qposadr[0] for r in metadata['robots']]
    for adr in robot_addresses:
        qpos[:, adr] -= .25*progress
    clearance = .008*np.minimum(np.clip((times-.5)/.5, 0, 1), np.clip((4.3-times)/.3, 0, 1))
    goals = []
    for obj in metadata['objects']:
        adr = model.joint(obj['name']+'_free').qposadr[0]
        qpos[:, adr] -= .19*progress
        qpos[:, adr+2] = obj['half_height']+clearance
        goals.append([initial[adr]-.19, initial[adr+1]])
    phases = np.zeros((len(times), 2), dtype=int)
    for phase, start in enumerate((.5, 1, 4, 4.3, 4.6), start=1):
        phases[times >= start] = phase
    pads = np.zeros((len(times), 40), dtype=np.uint8)
    pads[:, :4] = ((times >= .5) & (times < 4.3))[:, None]
    graph = np.zeros((len(times), 40, 40), dtype=bool)
    graph[1:, 0, 1:] = True; graph[1:, 1:, 0] = True
    graph[101:, 0, 1] = graph[101:, 1, 0] = False
    graph[102:, 1, 2] = graph[102:, 2, 1] = True
    minimum = np.full(len(times), 40); minimum[0] = 1; minimum[101] = 39
    actions = np.zeros((len(times)-1, 40, 4), dtype=np.float32)
    actions[..., 2] = -1; actions[..., 3] = 1; actions[100, 1, 3] = -1
    trajectory = {"times": times, "qpos": qpos, "object_phases": phases, "pad_contacts": pads,
                  "object_floor_contacts": np.tile((clearance < 1e-8)[:, None], (1, 2)),
                  "payload_robot_contacts": pads[:, :4].reshape(-1, 2, 2).sum(-1),
                  "magnetic_graph": graph, "substep_min_largest_component": minimum}
    report = {"goals_m": goals, "completion_time_s": 5., "requested_hold_seconds": 5., "failure": None}
    return trajectory, {"actions": actions}, model, metadata, report, robot_addresses


class BodyTaskTests(unittest.TestCase):
    def test_valid_synthetic_geometry_and_substep_disconnect_rejection(self):
        trajectory, trace, model, metadata, report, _ = synthetic_task()
        errors = []
        checks = audit_body_task(trajectory, trace, model, metadata, report, errors)
        self.assertEqual(errors, [])
        self.assertGreaterEqual(checks['all_forty_connected_before_release_s'], 1.)
        self.assertGreater(checks['magnetic_edges_formed_after_release'], 0)
        # The close/end-of-control graph remains connected. A transient break
        # inside one physics interval must still invalidate the final hold.
        trajectory['substep_min_largest_component'][300] = 39
        errors = []
        audit_body_task(trajectory, trace, model, metadata, report, errors)
        self.assertTrue(any('throughout the final hold' in error for error in errors), errors)

    def test_early_release_and_idle_fortieth_module_fail(self):
        trajectory, trace, model, metadata, report, addresses = synthetic_task()
        trace['actions'][5, 3, 3] = -1
        trajectory['qpos'][:, addresses[-1]] = trajectory['qpos'][0, addresses[-1]]
        errors = []
        audit_body_task(trajectory, trace, model, metadata, report, errors)
        self.assertTrue(any('one second before' in error for error in errors), errors)
        self.assertTrue(any('0.20 m net' in error for error in errors), errors)

    def test_pushing_payloads_does_not_count_as_supported_carry(self):
        trajectory, trace, model, metadata, report, _ = synthetic_task()
        for obj in metadata['objects']:
            adr = model.joint(obj['name']+'_free').qposadr[0]
            trajectory['qpos'][:, adr+2] = obj['half_height']
        trajectory['object_floor_contacts'][:] = True
        errors = []
        audit_body_task(trajectory, trace, model, metadata, report, errors)
        self.assertTrue(any('physically lifted and gripped' in error for error in errors), errors)

    def test_fourth_neural_action_is_checked_past_chunk_boundary(self):
        torch.manual_seed(42)
        config = PolicyConfig(**CONTRACT, mirror=True)
        actor = make_actor_critic(config)
        rng = np.random.default_rng(10)
        trace = {"local": rng.normal(size=(257, 2, 40)).astype(np.float32),
                 "neighbors": rng.normal(size=(257, 2, 1, 12)).astype(np.float32),
                 "neighbor_mask": np.ones((257, 2, 1), dtype=bool),
                 "active": np.ones((257, 2), dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'weights.npz'
            export_policy(actor, path)
            policy = NumpySwarmPolicy(path)
            trace['actions'] = policy(trace)
            errors = []
            checked = audit_neural_actions(policy, trace, errors)
            self.assertEqual(errors, [])
            self.assertEqual(checked['neural_action_rows_recomputed'], 257)
            trace['actions'][-1, -1, 3] += .02
            audit_neural_actions(policy, trace, errors)
            self.assertTrue(any('Neural actions disagree' in error for error in errors))

    def test_boolean_acceptance_does_not_bypass_training_measurements(self):
        config = vars(PolicyConfig(**CONTRACT))
        n, p, z = 32, .875, 1.96
        lower = (p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
        training = {"weights_sha256": 'abc', "acceptance_passed": True, "gpu": 'NVIDIA A100',
                    "cuda": '12.8', "cuda_optimization": True, "policy_device": 'cuda:0',
                    "backend": 'native', "ppo_updates": 1, "cumulative_ppo_actor_update_l2": .01,
                    "final_stage": 3, "config": config,
                    "heldout": {"complete": True, "episodes": n, "success_rate": p,
                                "success_wilson_lower_95": lower, "stage": 3, "policy": 'learned'},
                    "zero_baseline": {"complete": True, "episodes": n, "success_rate": 0., "stage": 3, "policy": 'zero'},
                    "args": {"env_factory": ENV_FACTORY, "robots": 40, "objects": 2, "cpu_smoke": False}}
        errors = []
        audit_training(training, 'abc', config, errors)
        self.assertEqual(errors, [])
        forged = copy.deepcopy(training)
        forged['heldout']['success_rate'] = .2
        forged['args']['env_factory'] = 'arena_mujoco.swarm_env:SwarmVectorEnv'
        forged['config']['action_dim'] = 3
        errors = []
        audit_training(forged, 'abc', config, errors)
        self.assertTrue(any('80% success' in error for error in errors), errors)
        self.assertTrue(any('magnetic-body task' in error for error in errors), errors)
        self.assertTrue(any('4 actions' in error for error in errors), errors)


class BodyArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='body-proof-tests-')
        cls.root = Path(cls.temp.name)
        cls.fixture = cls.root/'fixture'
        cls.report = record_body(cls.fixture, policy_type='teacher', allow_teacher=True, max_steps=3, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def copy_fixture(self):
        destination = self.root/self.id().rsplit('.', 1)[-1]
        shutil.copytree(self.fixture, destination)
        return destination

    def rewrite(self, directory, filename, change):
        with np.load(directory/filename, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        change(arrays)
        np.savez_compressed(directory/filename, **arrays)
        report = json.loads((directory/'report.json').read_text())
        report['artifacts'][filename] = digest(directory/filename)
        (directory/'report.json').write_text(json.dumps(report))

    def verify_rejected(self, directory, expected):
        result = verify_body(directory, allow_teacher=True)
        self.assertFalse(result['valid'], result)
        self.assertFalse(result['final_neural_proof'])
        self.assertTrue(any(expected in error for error in result['errors']), result['errors'])

    def test_honest_incomplete_teacher_is_diagnostic_not_final_proof(self):
        result = verify_body(self.fixture, allow_teacher=True)
        self.assertTrue(result['valid'], result['errors'])
        self.assertTrue(result['physics_replayed'])
        self.assertFalse(result['task_passed'])
        self.assertFalse(result['final_neural_proof'])
        self.assertEqual(result['checks']['replay_actions_completed'], 3)
        self.assertFalse(verify_body(self.fixture)['valid'])

    def test_report_success_and_final_gate_forgery_rejected(self):
        directory = self.copy_fixture()
        report = json.loads((directory/'report.json').read_text())
        report.update(success=True, final_neural_proof=True, gated=True, checkpoint_acceptance_passed=True)
        (directory/'report.json').write_text(json.dumps(report))
        self.verify_rejected(directory, 'Reported task success disagrees')

    def test_rehashed_middle_pose_forgery_rejected(self):
        directory = self.copy_fixture()
        def alter(arrays): arrays['qpos'][1, 0] += .001
        self.rewrite(directory, 'trajectory.npz', alter)
        self.verify_rejected(directory, 'Native replay mismatch for qpos')

    def test_rehashed_fourth_command_with_same_enable_state_rejected(self):
        directory = self.copy_fixture()
        def alter_action(arrays): arrays['actions'][-1, -1, 3] = .75
        def alter_command(arrays): arrays['magnet_commands'][-1, -1] = .75
        self.rewrite(directory, 'policy_trace.npz', alter_action)
        self.rewrite(directory, 'trajectory.npz', alter_command)
        self.verify_rejected(directory, 'Recomputed teacher action mismatch')

    def test_rehashed_magnetic_graph_forgery_rejected(self):
        directory = self.copy_fixture()
        def alter(arrays): arrays['magnetic_graph'][-1] = False
        self.rewrite(directory, 'trajectory.npz', alter)
        self.verify_rejected(directory, 'measured 2 mN force-bearing graph')

    def test_rehashed_external_payload_force_rejected(self):
        directory = self.copy_fixture()
        model = mujoco.MjModel.from_xml_path(str(directory/'scene.xml'))
        payload = model.body('payload_00').id
        def alter(arrays): arrays['xfrc_applied'][-1, payload, 0] = .001
        self.rewrite(directory, 'trajectory.npz', alter)
        self.verify_rejected(directory, 'external wrench exists on a payload')

    def test_rehashed_missing_global_observation_rejected(self):
        directory = self.copy_fixture()
        def alter(arrays): del arrays['global']
        self.rewrite(directory, 'policy_trace.npz', alter)
        self.verify_rejected(directory, 'Malformed')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
