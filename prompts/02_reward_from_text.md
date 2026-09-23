# 프롬프트 카드 02 — 말로 쓴 보상(reward) → my_reward.py

## 1) 먼저 원하는 움직임을 한 문장으로 씁니다

예)
- "천천히(약 1 m/s) 걷되 몸통을 꼿꼿하게 세우고 머리를 흔들지 말 것"
- "최대한 높이, 자주 점프하면서 앞으로 갈 것"
- "뒤로 걸을 것"
- "제자리에서 넘어지지 않고 서 있을 것. 앞뒤로 움직이면 벌점"
- "발을 크게 들어 올리며 걸을 것 (행진하듯)"
- "머리가 땅에 닿으면 실패로 끝낼 것"

## 2) AI 에게 보낼 프롬프트

`my_reward.py` 파일 내용을 전부 복사해서 `{ }` 자리에 붙입니다.

```
아래는 2D 캐릭터 강화학습(Stable-Baselines3 PPO)에서 매 스텝 reward 를 계산하는 Python 파일이다.
state 객체에서 쓸 수 있는 값은 파일 맨 위 설명에 있다. 허용되는 외부 라이브러리는 numpy뿐이다.

내가 원하는 움직임: "{여기에 한 문장}"

요청:
1. compute_reward() (필요하면 is_terminated() 도) 를 수정해서 위 움직임이 최대 점수를 받게 만들어라.
2. 각 항의 크기는 대략 -3 ~ +3 범위가 되게 가중치를 잡고, 매 스텝 살아있기 보너스(alive)는 0.5~1.0 으로 둔다.
3. compute_reward 는 반드시 float 하나를 return 한다. 기본적으로 extra_observation 은 그대로 둔다. gait phase나 contact를 observation에 꼭 추가해야 한다면 항상 같은 길이의 숫자 배열로 수정하고, 변경 이유와 새 길이를 주석에 적는다.
4. 각 항에 왜 넣었는지 한 줄 한글 주석을 단다.
5. 함수는 같은 입력에 항상 같은 결과를 내는 stateless 함수로 작성한다. 전역 counter/cache/list/dict를 수정하거나 random/time에 의존하지 않는다.
6. import는 `import numpy as np`만 허용한다. `open`, `eval`, `exec`를 쓰지 말고, 파일·환경변수·네트워크·프로세스에 접근하지 않는다. 특히 os, sys, pathlib, subprocess, socket, requests는 사용하지 않는다.
7. state나 info 안의 객체를 수정하지 않는다. 모든 계산 결과는 NaN/inf가 아닌 유한한 값이어야 한다.
8. 파일 전체를 출력한다 (설명 없이 코드만).

[현재 파일]
{my_reward.py 전체 붙여 넣기}
```

## 3) 저장하고 학습

- 받은 코드를 `my_reward.py` 에 덮어쓰거나, 비교하려면 `my_reward_v2.py` 같은 새 파일로 저장합니다 (프로젝트 폴더 안).
- **주의:** reward 파일은 일반 Python 코드이며 현재 사용자 권한으로 실행됩니다. 자동 sandbox가 아니므로, 저장 전에 위 제한 밖의 import나 파일·네트워크·프로세스 접근 코드가 없는지 직접 확인하세요.
- 학습:
  ```
  python 4_train.py --character my_character --minutes 10
  python 4_train.py --character my_character --reward my_reward_v2 --minutes 10     (새 파일로 저장한 경우)
  python 4_train.py --character dog --from pretrained/dog --minutes 4              (미리 학습된 모델에서 이어서, 빨리 보고 싶을 때)
  ```
- 시작 직후 `reward 함수 점검 중... OK` 가 나와야 합니다. 오류가 나면 메시지를 AI 에게 붙여 고칩니다.
- `extra_observation()` 길이를 바꾼 reward는 기존 모델과 observation 차원이 달라집니다. 이 경우 `--from`을 쓰지 말고 새로 학습하세요. 길이가 같은 reward끼리만 이어 학습할 수 있습니다.
- 각 run에는 `reward_used.py`와 character XML/meta/source JSON snapshot(가능한 경우)이 저장됩니다. `python 5_watch.py --run runs/<run 이름>`은 저장된 reward와 character snapshot을 재사용하고 character 파일의 SHA-256을 확인하므로, 원본 파일을 나중에 바꿔도 기존 run 재생은 바뀌지 않습니다.
- 비교:
  ```
  python 6_plot.py --character my_character
  python 5_watch.py --character my_character
  ```

### 고급: segment 위치와 속도 사용

발의 앞뒤 위치나 지지발 속도가 꼭 필요한 reward에서는 파일 상단에 아래 선언을 추가할 수 있습니다.

```python
USES_SEGMENT_KINEMATICS = True
```

그 다음 `state.segment_pos("foot_l")`, `state.segment_vel("foot_l")`처럼 사용합니다. 반환값은 해당 segment **body 원점**의 world `[x, y, z]` 위치(m)와 속도(m/s)입니다. 계산 비용이 추가되므로 필요한 reward에서만 선언하고, 존재하는 segment 이름만 사용하세요.

## 4) 결과를 보고 다시 말로 고치기

학습 결과는 보상을 **정직하게** 최적화한 결과입니다. 이상해 보이면 보상에 빈틈이 있다는 뜻입니다.

| 보이는 현상 | 보상에 추가할 말 |
|---|---|
| 몸을 눕혀서 미끄러져 간다 | "몸통 기울기가 30도를 넘으면 벌점" / "높이가 처음의 80% 아래면 벌점" |
| 가만히 서 있기만 한다 | "전진 속도 보상을 키우고 살아있기 보너스를 줄여라" |
| 다리를 미친 듯이 떤다 | "관절 속도가 크면 벌점(smooth)" / "힘 사용량 벌점을 10배" |
| 한 발로만 뛴다 | "두 발이 번갈아 바닥에 닿으면 보너스" (state.touching 사용) |
| 너무 빨라서 넘어진다 | "목표 속도 1 m/s 에 가까울수록 보상 (TARGET_SPEED)" |

## 5) 가중치만 바꾸는 가장 쉬운 방법 (AI 없이)

`my_reward.py` 맨 위의 `W = {...}` 숫자를 바꾸면 됩니다. 파일 아래쪽에 예시 A~D 가 있습니다.
