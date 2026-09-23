import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REWARD_PATH = Path(__file__).resolve().parents[1] / "rewards" / "reward_p1_natural_walk_v3.py"
SPEC = importlib.util.spec_from_file_location("reward_p1_natural_walk_v3_test", REWARD_PATH)
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
    # t=0 -> 왼발 heel-strike. 목표 ankle delta는 약 +0.359 m이다.
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
            "foot_l": np.array([0.18, 0.0, 0.06]),
            "foot_r": np.array([-0.18, 0.0, 0.06]),
        },
        velocities={
            "foot_l": np.array([0.02, 0.0, 0.0]),
            "foot_r": np.array([0.9, 0.0, 0.0]),
        },
    )
    values.update(overrides)
    return FakeState(**values)


class NaturalWalkV3RewardTests(unittest.TestCase):
    def test_target_ankle_delta_is_antisymmetric_over_half_cycle(self):
        values = []
        for time in (0.0, REWARD.GAIT_CYCLE_SEC / 2.0):
            state = make_state(time=time)
            phase_sin, phase_cos = REWARD._phase(state)
            targets = REWARD._target_pose(phase_sin, phase_cos)
            values.append(REWARD._target_ankle_delta(0.0, *targets))
        self.assertGreater(values[0], 0.30)
        self.assertLess(values[1], -0.30)
        self.assertTrue(math.isclose(values[0], -values[1], rel_tol=1e-9))

    def test_matching_lead_distance_beats_oversized_same_sign_stride(self):
        matched = REWARD.compute_reward(make_state(), 0.0, {})
        oversized = REWARD.compute_reward(
            make_state(
                positions={
                    "foot_l": np.array([0.40, 0.0, 0.06]),
                    "foot_r": np.array([-0.40, 0.0, 0.06]),
                }
            ),
            0.0,
            {},
        )
        self.assertGreater(matched, oversized)

    def test_both_legs_must_match_target_pose(self):
        matched = REWARD.compute_reward(make_state(), 0.0, {})
        sacrificed_left = REWARD.compute_reward(
            make_state(
                joints={
                    "thigh_l": -0.70,
                    "thigh_r": -0.24,
                    "shin_l": -1.20,
                    "shin_r": -0.15,
                }
            ),
            0.0,
            {},
        )
        self.assertGreater(matched, sacrificed_left)

    def test_mid_swing_contact_loses_clearance_reward_and_gets_penalty(self):
        time = REWARD.GAIT_CYCLE_SEC / 4.0  # 왼발 stance, 오른발 mid-swing
        targets = REWARD._target_pose(1.0, 0.0)
        joints = {
            "thigh_l": targets[0],
            "thigh_r": targets[1],
            "shin_l": targets[2],
            "shin_r": targets[3],
        }
        target_delta = REWARD._target_ankle_delta(0.0, *targets)
        positions = {
            "foot_l": np.array([target_delta / 2.0, 0.0, 0.06]),
            "foot_r": np.array([-target_delta / 2.0, 0.0, 0.16]),
        }
        clear = REWARD.compute_reward(
            make_state(time=time, joints=joints, positions=positions, contacts=("foot_l",)),
            0.0,
            {},
        )
        dragging = REWARD.compute_reward(
            make_state(
                time=time,
                joints=joints,
                positions=positions,
                contacts=("foot_l", "foot_r"),
            ),
            0.0,
            {},
        )
        self.assertGreater(clear, dragging)

    def test_unplanted_nominal_stance_velocity_is_not_called_slip(self):
        time = REWARD.GAIT_CYCLE_SEC / 4.0
        low = REWARD.compute_reward(
            make_state(time=time, contacts=("foot_r",)), 0.0, {}
        )
        velocities = dict(make_state().velocities)
        velocities["foot_l"] = np.array([1.2, 0.0, 0.0])
        high = REWARD.compute_reward(
            make_state(time=time, contacts=("foot_r",), velocities=velocities), 0.0, {}
        )
        self.assertTrue(math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12))

    def test_v2_compatible_extra_observation_shape(self):
        obs = np.asarray(REWARD.extra_observation(make_state()))
        self.assertEqual(obs.shape, (4,))
        self.assertTrue(np.isfinite(obs).all())

    def test_requires_segment_kinematics(self):
        self.assertTrue(REWARD.USES_SEGMENT_KINEMATICS)


if __name__ == "__main__":
    unittest.main()
