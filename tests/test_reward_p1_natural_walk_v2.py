import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REWARD_PATH = Path(__file__).resolve().parents[1] / "rewards" / "reward_p1_natural_walk_v2.py"
SPEC = importlib.util.spec_from_file_location("reward_p1_natural_walk_v2_test", REWARD_PATH)
REWARD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REWARD)


class FakeState(SimpleNamespace):
    def touching(self, *names):
        return any(name in self.contacts for name in names)

    def joint(self, name):
        return float(self.joints[name])

    def segment_pos(self, name):
        return np.asarray(self.positions[name], dtype=np.float64).copy()

    def segment_vel(self, name):
        return np.asarray(self.velocities[name], dtype=np.float64).copy()


def make_state(**overrides):
    # t=0 -> 왼발 heel-strike: 왼발이 앞, 오른발이 뒤에 있다.
    values = dict(
        time=0.0,
        x_vel=0.75,
        angle=0.0,
        height_ratio=0.93,
        height_vel=0.0,
        angle_vel=0.0,
        energy=0.5,
        joint_vels=np.zeros(6, dtype=np.float64),
        contacts=("foot_l",),
        joints={
            "thigh_l": 0.24,
            "thigh_r": -0.24,
            "shin_l": -0.15,
            "shin_r": -0.15,
        },
        positions={
            "foot_l": np.array([0.12, 0.0, 0.06]),
            "foot_r": np.array([-0.24, 0.0, 0.06]),
        },
        velocities={
            "foot_l": np.array([0.02, 0.0, 0.0]),
            "foot_r": np.array([0.9, 0.0, 0.0]),
        },
    )
    values.update(overrides)
    return FakeState(**values)


class NaturalWalkV2RewardTests(unittest.TestCase):
    def test_role_swapping_gait_beats_fixed_split_toe_tap(self):
        proper = REWARD.compute_reward(make_state(), 0.0, {})
        fixed = REWARD.compute_reward(
            make_state(
                joints={
                    "thigh_l": -0.24,
                    "thigh_r": 0.24,
                    "shin_l": -0.15,
                    "shin_r": -0.15,
                },
                positions={
                    "foot_l": np.array([-0.24, 0.0, 0.06]),
                    "foot_r": np.array([0.12, 0.0, 0.06]),
                },
            ),
            0.0,
            {},
        )
        self.assertGreater(proper, fixed)

    def test_stance_slip_reduces_reward(self):
        stable = REWARD.compute_reward(make_state(), 0.0, {})
        slipping_velocities = dict(make_state().velocities)
        slipping_velocities["foot_l"] = np.array([0.9, 0.0, 0.0])
        slipping = REWARD.compute_reward(
            make_state(velocities=slipping_velocities), 0.0, {}
        )
        self.assertGreater(stable, slipping)

    def test_v1_compatible_extra_observation_shape(self):
        obs = np.asarray(REWARD.extra_observation(make_state()))
        self.assertEqual(obs.shape, (4,))
        self.assertTrue(np.isfinite(obs).all())

    def test_requires_segment_kinematics(self):
        self.assertTrue(REWARD.USES_SEGMENT_KINEMATICS)


if __name__ == "__main__":
    unittest.main()
