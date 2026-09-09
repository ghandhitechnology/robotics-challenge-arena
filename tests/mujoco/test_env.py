"""Reference robot actuation and reproducible Gym controller behavior."""
import unittest
import warnings
from xml.etree.ElementTree import Element, SubElement, tostring

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from arena_mujoco.env import ArenaEnv
from arena_mujoco.robot import add_robot


class RobotTests(unittest.TestCase):
    def test_mass_and_forward_drive(self):
        root = Element("mujoco")
        SubElement(root, "option", timestep="0.0002", integrator="implicitfast")
        world = SubElement(root, "worldbody")
        SubElement(world, "geom", type="plane", size="2 2 .1", condim="6",
                   friction=".95 .0002 .00003")
        metadata = add_robot(world, SubElement(root, "actuator"), SubElement(root, "sensor"), {})
        model = mujoco.MjModel.from_xml_string(tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        wheel_dofs = [model.joint(name).dofadr[0] for name in metadata["wheel_joints"]]
        self.assertAlmostEqual(float(model.body_mass.sum()), 1.0)
        for step in range(5000):
            target = metadata["wheel_direction_sign"] * (10 if step >= 500 else 0)
            data.ctrl[:] = np.clip(metadata["velocity_gain"] * (target - data.qvel[wheel_dofs]),
                                   -metadata["max_motor_torque_nm"], metadata["max_motor_torque_nm"])
            mujoco.mj_step(model, data)
        self.assertGreater(data.qpos[1], 0.78 + 0.15)
        self.assertLess(abs(data.qpos[0] - 1.02), 0.005)
        self.assertGreater(data.qpos[3], 0.99)
        self.assertEqual(int(data.warning.number.sum()), 0)

    def test_flat_motor_profile_is_applied_once(self):
        root = Element("mujoco")
        metadata = add_robot(SubElement(root, "worldbody"), SubElement(root, "actuator"),
                             SubElement(root, "sensor"), {"max_motor_torque_nm": 0.10,
                                                          "motor_strength_scale": 0.8})
        model = mujoco.MjModel.from_xml_string(tostring(root, encoding="unicode"))
        self.assertAlmostEqual(metadata["max_motor_torque_nm"], 0.08)
        np.testing.assert_allclose(model.actuator_ctrlrange, [[-0.08, 0.08]] * 2)


class EnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = ArenaEnv(tape_mode=kwargs.pop("tape_mode", "none"), frame_skip=2, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_gym_contract(self):
        env = self.make_env()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Box observation space.*")
            check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=19)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs.shape, (270,))
        self.assertIn("Cylinder_Red", info["target_name"])

    def test_seeded_reset_and_delayed_noisy_action_replay(self):
        env = self.make_env(randomize=True, action_delay_s=0.0008,
                            motor_lag_s=0.002, action_noise_std=0.03)
        first, _ = env.reset(seed=31)
        profile = env.sim.profile
        outputs = [env.step(action) for action in ([0.2, -0.1], [0.4, 0.5], [0.3, 0.2])]
        second, _ = env.reset(seed=31)
        repeated = [env.step(action) for action in ([0.2, -0.1], [0.4, 0.5], [0.3, 0.2])]
        np.testing.assert_array_equal(first, second)
        self.assertEqual(profile, env.sim.profile)
        for left, right in zip(outputs, repeated):
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1:], right[1:])
        checkpoint = env.get_state()
        expected = env.step([0.8, -0.3])
        expected_qpos = env.data.qpos.copy()
        env.set_state(checkpoint)
        actual = env.step([0.8, -0.3])
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected_qpos, env.data.qpos)
        self.assertEqual(expected[1:], actual[1:])

    def test_flex_step_and_checkpoint(self):
        env = self.make_env(tape_mode="flex")
        env.step([0, 0])
        checkpoint = env.get_state()
        expected = env.step([0.1, 0.1])
        env.set_state(checkpoint)
        actual = env.step([0.1, 0.1])
        np.testing.assert_array_equal(expected[0], actual[0])
        self.assertEqual(expected[1:], actual[1:])
        self.assertEqual(actual[4]["tape_damage_fraction"], 0)
        self.assertEqual(int(env.data.warning.number.sum()), 0)

    def test_success_and_time_limit(self):
        env = self.make_env(max_episode_seconds=0.0004)
        _, _, terminated, truncated, info = env.step([0, 0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertIn("time_limit", info["truncation_reasons"])
        with self.assertRaises(RuntimeError):
            env.step([0, 0])
        env = self.make_env()
        _, qpos, dof = env._objects[env.target_index]
        env.data.qpos[qpos:qpos + 2] = env.goal_xy
        env.data.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(env.model, env.data)
        _, reward, terminated, truncated, info = env.step([0, 0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertGreater(reward, 9)


if __name__ == "__main__":
    unittest.main()
