"""
예시 reward A - 천천히(약 1 m/s), 꼿꼿하게 걷기

사용:  python 4_train.py --character dog --reward rewards/reward_slow_upright.py --minutes 10
"""
import numpy as np

TARGET_SPEED = 1.0   # 목표 속도 (m/s)


def compute_reward(state, base_reward, info):
    r = 0.0
    r += 1.5 * max(0.0, 1.0 - abs(state.x_vel - TARGET_SPEED))   # 목표 속도에 가까울수록 (최대 1.5)
    r += 1.0                                                      # 살아있기
    r -= 1.0 * abs(state.angle)                                   # 몸통 기울기 벌점 (rad)
    r += 0.5 * max(0.0, 1.0 - abs(1.0 - state.height_ratio))     # 처음 높이 유지
    r -= 0.002 * state.energy                                     # 힘 낭비 벌점
    return r


def is_terminated(state, base_terminated):
    return base_terminated


def extra_observation(state):
    return []
