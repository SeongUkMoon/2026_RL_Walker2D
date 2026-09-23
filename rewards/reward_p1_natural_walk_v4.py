"""P1 자연 보행 reward v4: 좌우 각 다리의 목표 궤적을 끝까지 추적한다.

v3는 양쪽 swing clearance를 회복했지만, 큰 관절 오차에서 Gaussian 점수와
gradient가 거의 0이 되어 한쪽 hip을 뒤에 고정하는 local optimum이 남았다.
v4는 목표 궤적과 최대 보상은 유지하면서 다음 세 항만 pseudo-Huber 형태로
바꾼다.

* 좌우 hip/knee를 각각 robust하게 추적
* 발 사이 차이 하나가 아니라 torso 기준 좌우 ankle-x를 각각 추적
* 실제 접지한 stance 발의 중간 위상 slip을 비포화 robust loss로 억제

extra observation은 v1~v3와 같은 4개이므로 기존 21차원 모델과 호환된다.
"""
from __future__ import annotations

import math

import numpy as np


USES_SEGMENT_KINEMATICS = True
TARGET_SPEED = 0.75
GAIT_CYCLE_SEC = 0.95
TORSO_TO_HIP = 0.15
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


def _pseudo_huber(error: float, scale: float) -> float:
    """0에서 매끄럽고 큰 오차에서도 gradient가 사라지지 않는 robust loss."""
    z = float(error) / scale
    return math.sqrt(1.0 + z * z) - 1.0


def _target_pose(phase_sin: float, phase_cos: float) -> tuple[float, ...]:
    return (
        0.24 * phase_cos,
        -0.24 * phase_cos,
        -0.15 - 0.80 * max(-phase_sin, 0.0) ** 2,
        -0.15 - 0.80 * max(phase_sin, 0.0) ** 2,
    )


def _target_ankle_relative(
    torso_angle: float,
    target_hip: float,
    target_knee: float,
) -> float:
    """torso root 기준 목표 ankle body-origin의 world-x 위치를 반환한다."""
    phi = float(torso_angle)
    return (
        -TORSO_TO_HIP * math.sin(phi)
        + THIGH_TO_KNEE * math.sin(target_hip - phi)
        + KNEE_TO_ANKLE * math.sin(target_hip + target_knee - phi)
    )


def _slip_loss(velocity_x: float) -> float:
    # ankle body-origin의 자연스러운 작은 rolling motion은 허용한다.
    excess = max(abs(float(velocity_x)) - 0.08, 0.0)
    return _pseudo_huber(excess, 0.15)


def compute_reward(state, base_reward, info):
    phase_sin, phase_cos = _phase(state)
    contact_l, contact_r = _foot_contacts(state)

    left_stance = phase_sin >= 0.0
    stance_contact = contact_l if left_stance else contact_r
    swing_contact = contact_r if left_stance else contact_l

    target_hip_l, target_hip_r, target_knee_l, target_knee_r = _target_pose(
        phase_sin, phase_cos
    )

    hip_loss = (
        _pseudo_huber(state.joint("thigh_l") - target_hip_l, 0.20)
        + _pseudo_huber(state.joint("thigh_r") - target_hip_r, 0.20)
    )
    knee_loss = (
        _pseudo_huber(state.joint("shin_l") - target_knee_l, 0.25)
        + _pseudo_huber(state.joint("shin_r") - target_knee_r, 0.25)
    )

    foot_l_pos = state.segment_pos("foot_l")
    foot_r_pos = state.segment_pos("foot_r")
    foot_l_vel = state.segment_vel("foot_l")
    foot_r_vel = state.segment_vel("foot_r")

    target_ankle_l = _target_ankle_relative(
        state.angle, target_hip_l, target_knee_l
    )
    target_ankle_r = _target_ankle_relative(
        state.angle, target_hip_r, target_knee_r
    )
    ankle_loss = (
        _pseudo_huber(float(foot_l_pos[0]) - state.x - target_ankle_l, 0.12)
        + _pseudo_huber(float(foot_r_pos[0]) - state.x - target_ankle_r, 0.12)
    )

    signed_clearance = (
        float(foot_r_pos[2] - foot_l_pos[2])
        if left_stance
        else float(foot_l_pos[2] - foot_r_pos[2])
    )
    phase_mid = abs(phase_sin)
    target_clearance = 0.10 * phase_mid
    clearance_track = _gaussian(signed_clearance, target_clearance, 0.05)

    # sin의 부호가 지정한 stance 발이 실제 접지했을 때만 벌점을 준다.
    # phase 경계에서는 gate가 0이 되어 정상적인 발 교환 충격을 제외한다.
    stance_slip = (
        max(phase_sin, 0.0) * contact_l * _slip_loss(foot_l_vel[0])
        + max(-phase_sin, 0.0) * contact_r * _slip_loss(foot_r_vel[0])
    )

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
    reward += 0.45 - 0.15 * hip_loss
    reward += 0.45 - 0.15 * knee_loss
    reward += 0.55 - 0.18 * ankle_loss
    reward += 0.35 * phase_mid * (1.0 - swing_contact) * clearance_track
    reward += 0.25 * stance_contact
    reward -= 0.35 * phase_mid * swing_contact
    reward -= 0.45 * stance_slip
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
    phase_sin, phase_cos = _phase(state)
    contact_l, contact_r = _foot_contacts(state)
    return np.array(
        [phase_sin, phase_cos, contact_l, contact_r], dtype=np.float64
    )
