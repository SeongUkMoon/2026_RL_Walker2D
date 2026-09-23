import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REWARD_PATH = Path(__file__).resolve().parents[1] / "rewards" / "reward_p1_natural_walk.py"
SPEC = importlib.util.spec_from_file_location("reward_p1_natural_walk_test", REWARD_PATH)
REWARD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REWARD)


class FakeState(SimpleNamespace):
    def touching(self, *names):
        return any(name in self.contacts for name in names)


def make_state(**overrides):
    values = dict(
        time=0.225,
        x_vel=0.8,
        angle=0.0,
        height_ratio=0.93,
        height_vel=0.0,
        angle_vel=0.0,
        energy=0.5,
        joint_vels=np.zeros(6, dtype=np.float64),
        contacts=("foot_l",),
    )
    values.update(overrides)
    return FakeState(**values)


class NaturalWalkRewardTests(unittest.TestCase):
    def test_target_gait_scores_above_fast_airborne_motion(self):
        target = REWARD.compute_reward(make_state(), 0.0, {})
        fast_airborne = REWARD.compute_reward(
            make_state(
                x_vel=3.1,
                angle=0.45,
                height_ratio=1.20,
                height_vel=1.5,
                angle_vel=4.0,
                energy=6.0,
                joint_vels=np.full(6, 8.0),
                contacts=(),
            ),
            0.0,
            {},
        )
        self.assertGreater(target, fast_airborne)

    def test_phase_matched_support_scores_above_wrong_or_airborne(self):
        matched = REWARD.compute_reward(make_state(contacts=("foot_l",)), 0.0, {})
        wrong = REWARD.compute_reward(make_state(contacts=("foot_r",)), 0.0, {})
        airborne = REWARD.compute_reward(make_state(contacts=()), 0.0, {})
        self.assertGreater(matched, wrong)
        self.assertGreater(matched, airborne)

    def test_extra_observation_is_fixed_and_finite(self):
        obs = np.asarray(REWARD.extra_observation(make_state()), dtype=np.float64)
        self.assertEqual(obs.shape, (4,))
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue(math.isclose(obs[2], 1.0))
        self.assertTrue(math.isclose(obs[3], 0.0))

    def test_termination_uses_base_or_torso_contact(self):
        self.assertFalse(REWARD.is_terminated(make_state(), False))
        self.assertTrue(REWARD.is_terminated(make_state(), True))
        self.assertTrue(
            REWARD.is_terminated(make_state(contacts=("torso",)), False)
        )


if __name__ == "__main__":
    unittest.main()
