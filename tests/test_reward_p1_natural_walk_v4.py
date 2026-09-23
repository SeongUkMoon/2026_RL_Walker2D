import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REWARD_PATH = Path(__file__).resolve().parents[1] / "rewards" / "reward_p1_natural_walk_v4.py"
SPEC = importlib.util.spec_from_file_location("reward_p1_natural_walk_v4_test", REWARD_PATH)
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


def target_positions(time=0.0, torso_angle=0.0, root_x=1.0):
    phase_angle = 2.0 * math.pi * time / REWARD.GAIT_CYCLE_SEC
    targets = REWARD._target_pose(math.sin(phase_angle), math.cos(phase_angle))
    return targets, {
        "foot_l": np.array([
            root_x + REWARD._target_ankle_relative(torso_angle, targets[0], targets[2]),
            0.0,
            0.06,
        ]),
        "foot_r": np.array([
            root_x + REWARD._target_ankle_relative(torso_angle, targets[1], targets[3]),
            0.0,
            0.06,
        ]),
    }


def make_state(**overrides):
    targets, positions = target_positions()
    values = dict(
        time=0.0,
        x=1.0,
        x_vel=0.75,
        angle=0.0,
        height_ratio=0.93,
        height_vel=0.0,
        angle_vel=0.0,
        energy=0.5,
        joint_vels=np.zeros(6, dtype=np.float64),
        contacts=("foot_l",),
        joints={
            "thigh_l": targets[0],
            "thigh_r": targets[1],
            "shin_l": targets[2],
            "shin_r": targets[3],
        },
        positions=positions,
        velocities={
            "foot_l": np.array([0.02, 0.0, 0.0]),
            "foot_r": np.array([0.9, 0.0, 0.0]),
        },
    )
    values.update(overrides)
    return FakeState(**values)


class NaturalWalkV4RewardTests(unittest.TestCase):
    def test_half_cycle_swaps_individual_ankle_targets(self):
        first, _ = target_positions(0.0)
        second, _ = target_positions(REWARD.GAIT_CYCLE_SEC / 2.0)
        first_l = REWARD._target_ankle_relative(0.0, first[0], first[2])
        first_r = REWARD._target_ankle_relative(0.0, first[1], first[3])
        second_l = REWARD._target_ankle_relative(0.0, second[0], second[2])
        second_r = REWARD._target_ankle_relative(0.0, second[1], second[3])
        self.assertTrue(math.isclose(first_l, second_r, abs_tol=1e-12))
        self.assertTrue(math.isclose(first_r, second_l, abs_tol=1e-12))
        self.assertGreater(first_l, first_r)

    def test_individual_tracking_rejects_common_shift_that_preserves_delta(self):
        matched = REWARD.compute_reward(make_state(), 0.0, {})
        shifted = {
            name: pos + np.array([0.25, 0.0, 0.0])
            for name, pos in make_state().positions.items()
        }
        wrong = REWARD.compute_reward(make_state(positions=shifted), 0.0, {})
        self.assertGreater(matched, wrong)

    def test_farther_pose_error_continues_to_reduce_reward(self):
        near_joints = dict(make_state().joints)
        near_joints["thigh_l"] -= 0.50
        far_joints = dict(make_state().joints)
        far_joints["thigh_l"] -= 0.80
        near = REWARD.compute_reward(make_state(joints=near_joints), 0.0, {})
        far = REWARD.compute_reward(make_state(joints=far_joints), 0.0, {})
        self.assertGreater(near, far)

    def test_contacting_midstance_slip_reduces_reward(self):
        time = REWARD.GAIT_CYCLE_SEC / 4.0
        targets, positions = target_positions(time)
        positions["foot_r"][2] = 0.16
        joints = {
            "thigh_l": targets[0],
            "thigh_r": targets[1],
            "shin_l": targets[2],
            "shin_r": targets[3],
        }
        stable_vel = dict(make_state().velocities)
        stable_vel["foot_l"] = np.array([0.02, 0.0, 0.0])
        slip_vel = dict(stable_vel)
        slip_vel["foot_l"] = np.array([0.8, 0.0, 0.0])
        stable = REWARD.compute_reward(
            make_state(time=time, joints=joints, positions=positions, velocities=stable_vel),
            0.0,
            {},
        )
        slipping = REWARD.compute_reward(
            make_state(time=time, joints=joints, positions=positions, velocities=slip_vel),
            0.0,
            {},
        )
        self.assertGreater(stable, slipping)

    def test_unplanted_nominal_stance_velocity_is_not_penalized(self):
        time = REWARD.GAIT_CYCLE_SEC / 4.0
        slow = REWARD.compute_reward(make_state(time=time, contacts=("foot_r",)), 0.0, {})
        velocities = dict(make_state().velocities)
        velocities["foot_l"] = np.array([1.2, 0.0, 0.0])
        fast = REWARD.compute_reward(
            make_state(time=time, contacts=("foot_r",), velocities=velocities), 0.0, {}
        )
        self.assertTrue(math.isclose(slow, fast, rel_tol=0.0, abs_tol=1e-12))

    def test_extra_observation_shape_is_compatible(self):
        obs = np.asarray(REWARD.extra_observation(make_state()))
        self.assertEqual(obs.shape, (4,))
        self.assertTrue(np.isfinite(obs).all())

    def test_requires_segment_kinematics(self):
        self.assertTrue(REWARD.USES_SEGMENT_KINEMATICS)


if __name__ == "__main__":
    unittest.main()
