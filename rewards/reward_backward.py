"""
예시 reward D - 뒤로 걷기

사용:  python 4_train.py --character biped --reward rewards/reward_backward.py --minutes 10
"""
import numpy as np


def compute_reward(state, base_reward, info):
    r = 0.0
    r += 1.0 * (-state.x_vel)      # 뒤로(왼쪽으로) 가는 속도
    r += 1.0                        # 살아있기
    r -= 0.001 * state.energy
    return r


def is_terminated(state, base_terminated):
    return base_terminated


def extra_observation(state):
    return []
