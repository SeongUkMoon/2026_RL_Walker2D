"""
예시 reward C - 제자리에서 넘어지지 않고 서 있기 (1~2분이면 배웁니다)

사용:  python 4_train.py --character ostrich --reward rewards/reward_stand_still.py --minutes 3
"""
import numpy as np


def compute_reward(state, base_reward, info):
    r = 0.0
    r += 1.0                                                   # 살아있기
    r += 1.0 * max(0.0, 1.0 - abs(1.0 - state.height_ratio))  # 처음 높이 유지
    r -= 1.0 * abs(state.angle)                                # 기울기 벌점
    r -= 0.5 * abs(state.x_vel)                                # 앞뒤로 움직이면 벌점
    r -= 0.01 * state.energy                                   # 힘 낭비 벌점 (크게: 조용히 서 있기)
    return r


def is_terminated(state, base_terminated):
    return base_terminated


def extra_observation(state):
    return []
