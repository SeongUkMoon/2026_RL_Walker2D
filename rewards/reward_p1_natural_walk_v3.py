"""P1 자연 보행 reward v3: 좌우 보폭과 swing 높이를 대칭에 가깝게 만든다.

v2는 실제 앞발 역할 교대와 지지발 미끄럼 감소에는 성공했지만, 한쪽 다리만
크게 휘두르고 다른 쪽 발은 낮게 끄는 비대칭 해를 찾았다. v3는 다음을 고친다.

* 양쪽 pose 점수를 산술평균하지 않고 기하평균하여 한쪽 다리 희생을 억제
* 관절 목표와 일치하는 ankle-x 목표를 사용하여 좌우에 같은 보폭을 요구
* swing 중 접촉을 더 강하게 벌점하고, 실제로 뜬 swing 발에만 clearance 보상
* 실제 접지한 stance 발에만 phase 중앙부 slip 벌점 적용

extra observation은 v1/v2와 같은 4개이므로 기존 21차원 모델에서 이어 학습할
수 있다. 아래 길이는 P1 XML의 hip->knee 및 knee->ankle body offset이다.
"""
from __future__ import annotations

import math

import numpy as np


USES_SEGMENT_KINEMATICS = True
TARGET_SPEED = 0.75
GAIT_CYCLE_SEC = 0.95
THIGH_TO_KNEE = 0.32
KNEE_TO_ANKLE = 0.44


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


def _target_pose(phase_sin: float, phase_cos: float) -> tuple[float, ...]:
    """(hip_l, hip_r, knee_l, knee_r) 목표를 반환한다."""
    return (
        0.24 * phase_cos,
        -0.24 * phase_cos,
        -0.15 - 0.80 * max(-phase_sin, 0.0) ** 2,
        -0.15 - 0.80 * max(phase_sin, 0.0) ** 2,
    )


def _target_ankle_delta(
    torso_angle: float,
    target_hip_l: float,
    target_hip_r: float,
    target_knee_l: float,
    target_knee_r: float,
) -> float:
    """목표 pose가 만드는 world-x ``ankle_l - ankle_r``를 계산한다."""
    phi = float(torso_angle)
    left_x = (
        THIGH_TO_KNEE * math.sin(target_hip_l - phi)
        + KNEE_TO_ANKLE * math.sin(target_hip_l + target_knee_l - phi)
    )
    right_x = (
        THIGH_TO_KNEE * math.sin(target_hip_r - phi)
        + KNEE_TO_ANKLE * math.sin(target_hip_r + target_knee_r - phi)
    )
    return left_x - right_x


def compute_reward(state, base_reward, info):
    phase_sin, phase_cos = _phase(state)
    contact_l, contact_r = _foot_contacts(state)

    # sin>0: 왼발 stance, 오른발 swing. 다음 반 주기는 반대다.
    left_stance = phase_sin >= 0.0
    stance_contact = contact_l if left_stance else contact_r
    swing_contact = contact_r if left_stance else contact_l

    target_hip_l, target_hip_r, target_knee_l, target_knee_r = _target_pose(
        phase_sin, phase_cos
    )

    # 두 다리 중 하나만 잘 맞혀도 절반 점수를 받던 v2의 산술평균을 없앤다.
    # 이는 각 다리 Gaussian 점수의 기하평균과 같다.
    hip_error_sq = (
        ((state.joint("thigh_l") - target_hip_l) / 0.20) ** 2
        + ((state.joint("thigh_r") - target_hip_r) / 0.20) ** 2
    )
    knee_error_sq = (
        ((state.joint("shin_l") - target_knee_l) / 0.25) ** 2
        + ((state.joint("shin_r") - target_knee_r) / 0.25) ** 2
    )
    hip_track = math.exp(-0.25 * hip_error_sq)
    knee_track = math.exp(-0.25 * knee_error_sq)

    foot_l_pos = state.segment_pos("foot_l")
    foot_r_pos = state.segment_pos("foot_r")
    foot_l_vel = state.segment_vel("foot_l")
    foot_r_vel = state.segment_vel("foot_r")

    lead_delta = float(foot_l_pos[0] - foot_r_pos[0])
    target_lead_delta = _target_ankle_delta(
        state.angle,
        target_hip_l,
        target_hip_r,
        target_knee_l,
        target_knee_r,
    )
    lead_track = _gaussian(lead_delta, target_lead_delta, 0.12)

    signed_clearance = (
        float(foot_r_pos[2] - foot_l_pos[2])
        if left_stance
        else float(foot_l_pos[2] - foot_r_pos[2])
    )
    phase_mid = abs(phase_sin)
    target_clearance = 0.10 * phase_mid
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
    reward += 0.55 * lead_track
    reward += 0.35 * phase_mid * (1.0 - swing_contact) * clearance_track
    reward += 0.25 * stance_contact
    reward -= 0.35 * phase_mid * swing_contact
    reward -= 0.40 * phase_mid * stance_contact * slip
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
    # v1/v2와 동일한 4개를 유지해 저장 모델/VecNormalize와 호환한다.
    phase_sin, phase_cos = _phase(state)
    contact_l, contact_r = _foot_contacts(state)
    return np.array(
        [phase_sin, phase_cos, contact_l, contact_r], dtype=np.float64
    )
