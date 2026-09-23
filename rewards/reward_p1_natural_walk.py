"""P1용 자연 보행 reward 1차안.

목표는 최고 속도가 아니라 약 0.8 m/s의 전진, 직립 자세, 낮은 수직 진동,
그리고 좌우 발의 주기적인 교대 접지다. 이전 ``my_reward.py`` 모델과 비교할 수
있도록 별도 reward로 유지한다.

이 reward는 보행 위상과 발 접촉 두 값을 observation에 추가하므로 기존 17차원
모델과 호환되지 않는다. 반드시 ``--from`` 없이 새로 학습한다.
"""
from __future__ import annotations

import math

import numpy as np


TARGET_SPEED = 0.8       # m/s
GAIT_CYCLE_SEC = 0.90    # 왼발 지지 + 오른발 지지 한 주기


def _phase(state) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(state.time) / GAIT_CYCLE_SEC
    return math.sin(angle), math.cos(angle)


def _foot_contacts(state) -> tuple[float, float]:
    return (
        1.0 if state.touching("foot_l") else 0.0,
        1.0 if state.touching("foot_r") else 0.0,
    )


def _gaussian(value: float, center: float, sigma: float) -> float:
    z = (float(value) - center) / sigma
    return math.exp(-0.5 * z * z)


def compute_reward(state, base_reward, info):
    phase_sin, _ = _phase(state)
    contact_l, contact_r = _foot_contacts(state)

    # 반 주기마다 원하는 지지발을 바꾼다. 접촉 이력을 전역 변수에 저장하지 않아
    # 병렬 환경과 episode reset에서도 안전하다.
    desired_l = 1.0 if phase_sin >= 0.0 else 0.0
    desired_r = 1.0 - desired_l
    contact_match = 0.5 * (
        (1.0 - abs(contact_l - desired_l))
        + (1.0 - abs(contact_r - desired_r))
    )

    speed_track = _gaussian(state.x_vel, TARGET_SPEED, 0.30)
    upright = _gaussian(state.angle, 0.0, 0.25)
    height_track = _gaussian(state.height_ratio, 0.93, 0.12)

    vertical_speed = float(np.clip(state.height_vel, -1.5, 1.5)) / 0.75
    angular_speed = float(np.clip(state.angle_vel, -6.0, 6.0)) / 3.0
    joint_speed_sq = min(float(np.mean(np.square(state.joint_vels))), 100.0)

    airborne = contact_l == 0.0 and contact_r == 0.0
    double_support = contact_l == 1.0 and contact_r == 1.0
    bad_ground_contact = any(
        name not in {"foot_l", "foot_r"} for name in state.contacts
    )

    reward = 0.0
    reward += 2.00 * speed_track
    reward += 0.15 * float(np.clip(state.x_vel, -1.0, 1.2))
    reward += 0.20                              # alive
    reward += 0.80 * upright
    reward += 0.50 * height_track
    reward += 0.60 * contact_match
    reward -= 0.60 if airborne else 0.0
    reward -= 0.10 if double_support else 0.0
    reward -= 1.00 if bad_ground_contact else 0.0
    reward -= 0.25 * vertical_speed**2
    reward -= 0.08 * angular_speed**2
    reward -= 0.01 * float(state.energy)
    reward -= 0.003 * joint_speed_sq
    return float(reward)


def is_terminated(state, base_terminated):
    return bool(base_terminated or state.touching("torso"))


def extra_observation(state):
    phase_sin, phase_cos = _phase(state)
    contact_l, contact_r = _foot_contacts(state)
    return np.array(
        [phase_sin, phase_cos, contact_l, contact_r], dtype=np.float64
    )
