"""
예시 reward E - 기본 reward 그대로 + '머리가 땅에 닿으면 실패' (termination 조건 바꾸기 예시)
dog / ostrich 처럼 head 마디가 있는 캐릭터용.

사용:  python 4_train.py --character dog --reward rewards/reward_head_safe.py --minutes 10
"""
import numpy as np


def compute_reward(state, base_reward, info):
    r = 1.0 * state.x_vel + 1.0 - 0.001 * state.energy
    if state.touching("head"):
        r -= 2.0                    # 머리가 닿은 스텝은 큰 벌점
    return r


def is_terminated(state, base_terminated):
    return base_terminated or state.touching("head")


def extra_observation(state):
    return []
