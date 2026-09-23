"""P1 자연 보행 reward v2: 접촉뿐 아니라 다리의 앞뒤 역할도 교대한다.

v1은 체공과 속도를 잘 제어했지만 한 발은 계속 앞, 다른 발은 계속 뒤에 둔 채
toe-tap과 지면 미끄럼으로 접촉 위상만 맞추는 해를 찾았다. v2는 다음을 추가한다.

* 좌우 hip/knee의 연속적인 반대 위상 목표
* 발목 world-x 위치의 lead 역할 교대
* swing 발 clearance
* 접지 중인 stance 발목의 world-x 미끄럼 벌점

extra observation은 v1과 같은 4개이므로 v1의 21차원 모델에서 이어 학습할 수 있다.
"""
from __future__ import annotations

import math

import numpy as np


USES_SEGMENT_KINEMATICS = True
TARGET_SPEED = 0.75
GAIT_CYCLE_SEC = 0.95


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
    phase_sin, phase_cos = _phase(state)
    contact_l, contact_r = _foot_contacts(state)

    # sin>0: 왼발 stance/back, 오른발 swing/front. 다음 반 주기는 반대다.
    left_stance = phase_sin >= 0.0
    stance_contact = contact_l if left_stance else contact_r
    swing_contact = contact_r if left_stance else contact_l

    # contact 반주기(sin)와 90도 차이인 cos 궤적으로 lead 역할을 바꾼다.
    # theta=0: 왼발 heel-strike(앞), theta=pi/2: 두 다리 교차,
    # theta=pi: 오른발 heel-strike(앞).
    target_hip_l = 0.24 * phase_cos
    target_hip_r = -0.24 * phase_cos
    target_knee_l = -0.15 - 0.80 * max(-phase_sin, 0.0) ** 2
    target_knee_r = -0.15 - 0.80 * max(phase_sin, 0.0) ** 2

    hip_track = 0.5 * (
        _gaussian(state.joint("thigh_l"), target_hip_l, 0.20)
        + _gaussian(state.joint("thigh_r"), target_hip_r, 0.20)
    )
    knee_track = 0.5 * (
        _gaussian(state.joint("shin_l"), target_knee_l, 0.25)
        + _gaussian(state.joint("shin_r"), target_knee_r, 0.25)
    )

    foot_l_pos = state.segment_pos("foot_l")
    foot_r_pos = state.segment_pos("foot_r")
    foot_l_vel = state.segment_vel("foot_l")
    foot_r_vel = state.segment_vel("foot_r")

    lead_delta = float(foot_l_pos[0] - foot_r_pos[0])
    lead_swap = math.tanh(4.0 * phase_cos) * math.tanh(lead_delta / 0.10)

    # swing 발목은 주기 중앙에서 stance 발목보다 약 10 cm 높고, touchdown과
    # toe-off에서는 같은 높이에 가까워지도록 한다.
    signed_clearance = (
        float(foot_r_pos[2] - foot_l_pos[2])
        if left_stance
        else float(foot_l_pos[2] - foot_r_pos[2])
    )
    target_clearance = 0.10 * abs(phase_sin)
    clearance_track = _gaussian(signed_clearance, target_clearance, 0.05)

    stance_vel_x = float(foot_l_vel[0] if left_stance else foot_r_vel[0])
    slip = float(np.clip((abs(stance_vel_x) - 0.10) / 0.60, 0.0, 1.0))

    speed_track = _gaussian(state.x_vel, TARGET_SPEED, 0.25)
    upright = _gaussian(state.angle, 0.0, 0.25)
    height_track = _gaussian(state.height_ratio, 0.93, 0.12)
    vertical_speed = float(np.clip(state.height_vel, -1.5, 1.5)) / 0.75
    angular_speed = float(np.clip(state.angle_vel, -6.0, 6.0)) / 3.0
    joint_speed_sq = min(float(np.mean(np.square(state.joint_vels))), 100.0)

    airborne = contact_l == 0.0 and contact_r == 0.0
    bad_ground_contact = any(
        name not in {"foot_l", "foot_r"} for name in state.contacts
    )

    reward = 0.0
    reward += 1.50 * speed_track
    reward += 0.15 * float(np.clip(state.x_vel, -1.0, 1.1))
    reward += 0.15                              # alive
    reward += 0.60 * upright
    reward += 0.35 * height_track
    reward += 0.45 * hip_track
    reward += 0.45 * knee_track
    reward += 0.55 * lead_swap
    reward += 0.35 * clearance_track
    reward += 0.25 * stance_contact
    reward -= 0.20 * abs(phase_sin) * swing_contact
    reward -= 0.40 * slip
    reward -= 0.35 if airborne else 0.0
    reward -= 1.00 if bad_ground_contact else 0.0
    reward -= 0.20 * vertical_speed**2
    reward -= 0.08 * angular_speed**2
    reward -= 0.008 * float(state.energy)
    reward -= 0.002 * joint_speed_sq
    return float(reward)


def is_terminated(state, base_terminated):
    return bool(base_terminated or state.touching("torso"))


def extra_observation(state):
    # v1과 동일한 4개를 유지해 저장 모델/VecNormalize의 21차원 관측과 호환한다.
    phase_sin, phase_cos = _phase(state)
    contact_l, contact_r = _foot_contacts(state)
    return np.array(
        [phase_sin, phase_cos, contact_l, contact_r], dtype=np.float64
    )
