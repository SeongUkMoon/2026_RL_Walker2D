"""
예시 reward B - 앞으로 뛰어가기 (점프)

사용:  python 4_train.py --character dog --reward rewards/reward_jump.py --minutes 10
"""
import numpy as np


def compute_reward(state, base_reward, info):
    r = 0.0
    r += 1.0 * state.x_vel                        # 전진
    r += 1.5 * max(0.0, state.height_vel)         # 위로 솟는 속도 보너스
    r += 1.0 * max(0.0, state.height_ratio - 1.0) * 5.0   # 처음 높이보다 높이 떠 있으면 보너스
    r += 0.5                                      # 살아있기 (작게: 가만히 있지 못하게)
    r -= 0.001 * state.energy
    return r


def is_terminated(state, base_terminated):
    return base_terminated


def extra_observation(state):
    return []
