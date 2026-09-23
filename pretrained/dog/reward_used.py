"""
my_reward.py  -  2교시(reward tuning)에서 수정하는 파일입니다.

캐릭터가 "무엇을 잘하면 점수를 받는지"를 여기서 정합니다. 학습(4_train.py)은 이 점수(reward)의 합을
최대로 만드는 controller 를 찾습니다. 세 개의 함수가 있고, 보통은 compute_reward() 와 위의 W(가중치)만 바꿉니다.

  compute_reward(state, base_reward, info) -> float   매 스텝의 점수
  is_terminated(state, base_terminated)    -> bool    에피소드를 여기서 끝낼지 (넘어짐 판정)
  extra_observation(state)                 -> list    (고급, 선택) observation 에 값 추가

state 에서 쓸 수 있는 값 (자세한 설명은 walker_rl/character_env.py 의 State 참고)
  state.x_vel          전진 속도 (m/s), 앞으로 가면 +
  state.height         몸통 높이 (m)             state.height_ratio  처음 높이에 대한 비율 (1.0 = 처음 높이)
  state.height_vel     위로 움직이는 속도 (m/s)
  state.angle          몸통 기울기 (rad)          state.angle_deg     같은 값 (deg)
  state.angle_vel      몸통 회전 속도 (rad/s)
  state.energy         action 제곱합 (힘을 얼마나 쓰는지)
  state.action         action 배열 (-1~1)         state.joint_angles / state.joint_vels  관절 각도/각속도 배열
  state.joint("thigh_r")   이름으로 관절 각도(rad) 얻기
  state.contacts       바닥에 닿은 segment 이름들   state.touching("head", "torso")  둘 중 하나라도 닿았으면 True
  state.time / state.step   에피소드 안에서 흐른 시간(s) / 스텝 수
  base_reward          Gymnasium 기본 reward (전진속도 + 1.0 - 0.001*energy). 그대로 return 해도 됩니다.

팁
  - 값이 갑자기 커지는 항(예: 속도 * 100)은 학습을 불안정하게 합니다. 각 항이 대략 -3 ~ +3 범위가 되게 하세요.
  - "매 스텝 +1" (alive) 은 '넘어지지 말라'는 뜻입니다. 이게 너무 크면 가만히 서 있기만 합니다.
  - 바꾼 뒤에는  python 4_train.py --character <이름> --minutes 10  처럼 다시 학습하고 6_plot.py 로 비교하세요.
  - 말로 설명한 보상을 코드로 바꾸고 싶으면 prompts/02_reward_from_text.md 의 프롬프트를 AI 에게 주세요.
"""
import numpy as np

# ---------------------------------------------------------------------------
# 가중치(weight). 0 이면 그 항은 꺼진 것입니다.
# ---------------------------------------------------------------------------
W = {
    "forward": 1.0,    # 전진 속도(m/s) x 이 값.  빠르게 앞으로 갈수록 좋음
    "alive": 1.0,      # 넘어지지 않고 있는 매 스텝마다 받는 보너스
    "control": 0.001,  # 힘(action) 낭비 벌점  (energy x 이 값)
    "posture": 0.0,    # 몸통 기울기 벌점      (|angle(rad)| x 이 값)   예: 0.5
    "height": 0.0,     # 처음 높이 유지 보너스 (높이가 처음과 같을 때 최대 1)  예: 0.5
    "jump": 0.0,       # 위로 뛰는 속도 보너스 (height_vel 이 + 일 때)       예: 1.0
    "smooth": 0.0,     # 관절이 너무 빨리 움직이면 벌점 (부드러운 동작)     예: 0.01
}

# 목표 속도 (m/s). None 이면 "빠를수록 좋음". 숫자(예: 1.0)를 넣으면 그 속도에 가까울수록 좋음 (천천히 걷기)
TARGET_SPEED = None


def compute_reward(state, base_reward, info):
    """매 스텝의 reward. float 하나를 return 합니다."""
    r = 0.0

    # 1) 전진
    if TARGET_SPEED is None:
        r += W["forward"] * state.x_vel
    else:
        r += W["forward"] * max(0.0, 1.0 - abs(state.x_vel - TARGET_SPEED))

    # 2) 살아있기(넘어지지 않기)
    r += W["alive"]

    # 3) 벌점들
    r -= W["control"] * state.energy
    r -= W["posture"] * abs(state.angle)
    r -= W["smooth"] * float(np.mean(np.square(state.joint_vels)))

    # 4) 자세/점프 보너스
    r += W["height"] * max(0.0, 1.0 - abs(1.0 - state.height_ratio))
    r += W["jump"] * max(0.0, state.height_vel)

    return r


def is_terminated(state, base_terminated):
    """
    True 를 return 하면 에피소드가 끝납니다(넘어짐 판정).
    base_terminated: 기본 판정 (몸통 높이가 처음의 55% 아래로 내려가거나, 60도 이상 기울면 True)
    예) 머리가 땅에 닿으면 끝:   return base_terminated or state.touching("head")
    예) 절대 끝내지 않기:        return False   (넘어진 채로 기어가는 법을 배울 수도 있음)
    """
    return base_terminated


def extra_observation(state):
    """
    (고급, 선택) observation 에 추가할 숫자 목록. 기본은 빈 목록.
    예) 발이 땅에 닿았는지 알려주기:  return [1.0 if state.touching("foot_l") else 0.0, 1.0 if state.touching("foot_r") else 0.0]
    주의: 이 목록의 길이를 바꾸면 이전에 학습한 모델과 호환되지 않습니다 (새로 학습해야 함).
    """
    return []


# ---------------------------------------------------------------------------
# 참고용 예시 (compute_reward 안의 내용을 이런 식으로 바꿔 볼 수 있습니다)
# ---------------------------------------------------------------------------
# 예시 A. 천천히, 꼿꼿하게 걷기
#   TARGET_SPEED = 0.8 ; W["posture"] = 1.0 ; W["height"] = 0.5
#
# 예시 B. 앞으로 뛰어가기 (점프)
#   W["jump"] = 1.0 ; W["forward"] = 1.0 ; W["alive"] = 0.5
#
# 예시 C. 제자리에서 균형 잡고 서 있기 (1~2분이면 배웁니다)
#   W["forward"] = 0.0 ; W["alive"] = 1.0 ; W["height"] = 1.0 ; W["posture"] = 1.0 ; W["control"] = 0.01
#
# 예시 D. 뒤로 걷기
#   compute_reward 안의  state.x_vel  을  -state.x_vel  로
