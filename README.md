# Walker2D 강화학습 실습

종이에 그린 2D 캐릭터를 MuJoCo 시뮬레이션에 불러오고, PPO로 움직임을 학습하는 2시간 실습입니다. 1교시에는 손그림을 캐릭터 데이터로 만들고, 2교시에는 보상을 바꾸며 움직임이 어떻게 달라지는지 비교합니다.

`손그림 → 캐릭터 JSON → 물리 시뮬레이션 → 강화학습 → 결과 비교`

- 수업 시간: 1교시 60분 + 2교시 60분
- 실행 환경: Windows, Python 3.11, CPU 사용(GPU 불필요)
- 프로그래밍 경험이 없어도 참여할 수 있습니다.

> 사진 속 그림을 그대로 3D 모델로 만드는 과정은 아닙니다. 몸통과 팔다리의 마디, 관절, 연결 관계를 읽어 2D 관절 구조로 바꿉니다. 옆모습 구조도에 영문 라벨을 붙여 그리면 결과가 가장 안정적입니다.

---

## 1. 수업 전 준비

1. [Python 3.11.9](https://www.python.org/downloads/release/python-3119/)의 **Windows installer (64-bit)** 를 설치합니다. 설치 첫 화면에서 **Add python.exe to PATH**를 체크해 두는 것이 좋습니다.
2. 실습 폴더를 ZIP으로 받아 `C:\Walker2D`처럼 한글이 없는 경로에 풉니다.
3. `setup.bat`을 더블클릭합니다. 가상환경과 필요한 패키지가 설치되며, 마지막에 `모든 점검 통과!`가 나오면 준비가 끝납니다.
4. 손그림을 AI로 변환할 경우, 이미지를 첨부할 수 있는 AI 채팅에 미리 로그인합니다.

### 터미널 열기

`open_terminal.bat`을 더블클릭하면 실습용 가상환경이 적용된 터미널이 열립니다. 이 README의 명령은 모두 그 터미널에서 실행합니다.

설치 상태를 다시 확인하려면 다음 명령을 사용합니다.

```text
python 0_check.py
```

각 스크립트의 전체 옵션은 `python 파일이름.py --help`로 볼 수 있습니다.

---

## 2. 1교시: 캐릭터 만들기

| 시간 | 진행 내용 | 명령 |
|---|---|---|
| 0–5분 | 설치 확인 | `python 0_check.py` |
| 5–15분 | 기본 Walker2D를 무작위 동작, 무동작, 관절 흔들기로 비교 | `python 1_run_walker.py`<br>`python 1_run_walker.py --mode zero`<br>`python 1_run_walker.py --mode wiggle` |
| 15–25분 | 캐릭터의 마디, 관절, 모터 구조 살펴보기 | `python 3_play.py dog --mode wiggle`<br>`python 3_play.py ostrich` |
| 25–35분 | 종이에 캐릭터 그리기 | — |
| 35–45분 | 사진과 `prompts/01_sketch_to_json.md`를 AI에 전달하고 결과를 `characters/my_character.json`에 저장 | 편집기 사용 |
| 45–55분 | JSON 검사, 캐릭터 빌드, 낙하·관절 동작 확인 | `python 2_build_character.py --validate-only`<br>`python 2_build_character.py`<br>`python 3_play.py my_character` |
| 55–60분 | 결과 공유 또는 영상 저장 | `python 3_play.py my_character --record` |

### 그림 그릴 때 표시할 것

- 캐릭터를 옆모습으로 그리고 전진 방향을 화살표로 표시합니다.
- 몸통과 팔다리의 각 마디에 서로 다른 영문 소문자 이름을 붙입니다.
- 관절 위치는 동그라미로, 바닥에 닿는 부분은 별표로 표시합니다.
- 부모와 자식 마디가 어디에서 연결되는지 분명하게 그립니다.
- 전체 높이를 m 단위로 적습니다.

그림이 모호해 AI가 확인 질문을 하면, 답변한 뒤 JSON을 확정합니다. AI를 사용하지 않아도 `characters/my_character.json` 템플릿의 길이, 두께, 관절 범위만 바꿔 실습할 수 있습니다.

JSON 형식은 [`characters/README.md`](characters/README.md)에 정리되어 있습니다.

---

## 3. 2교시: 강화학습과 보상 조정

| 시간 | 진행 내용 | 명령 |
|---|---|---|
| 0–10분 | 미리 학습된 캐릭터의 움직임 확인 | `python 5_watch.py --pretrained walker`<br>`python 5_watch.py --pretrained dog` |
| 10–20분 | 상태(state), 행동(action), 보상(reward) 이해하기 | `my_reward.py` 열기 |
| 20–25분 | 내 캐릭터 학습 시작 | `python 4_train.py --character my_character --minutes 12` |
| 25–40분 | `my_reward.py`의 가중치를 바꾸거나 다른 보상으로 학습 | `python 4_train.py --character my_character --reward rewards/reward_jump.py --minutes 10` |
| 40–50분 | 학습 결과와 곡선 비교 | `python 5_watch.py --character my_character`<br>`python 6_plot.py --character my_character` |
| 50–60분 | 결과 발표, 보상과 움직임의 관계 토론 | `python 5_watch.py --character my_character --terrain bumps` |

`my_reward.py`의 `W` 값을 바꾸면 보상의 비중을 쉽게 조절할 수 있습니다.

- 더 빨리 전진시키기: `forward`
- 넘어지지 않게 하기: `alive`, `height`
- 몸통을 세우기: `posture`
- 힘을 덜 쓰게 하기: `control` (힘 사용 벌점)

전진 보상만 지나치게 크면 걷는 대신 미끄러지거나 점프하는 동작을 학습할 수 있습니다. 자세와 높이, 에너지 보상을 함께 조정해 결과를 비교해 보세요.

학습 속도와 결과는 컴퓨터와 캐릭터 구조에 따라 달라집니다. 짧은 수업에서는 완성도 높은 보행보다, 보상을 바꿨을 때 움직임이 어떻게 달라지는지 확인하는 데 초점을 둡니다.

미리 학습된 모델에서 이어서 학습하면 짧은 시간에 보상 변화의 효과를 볼 수 있습니다.

```text
python 4_train.py --character dog --from pretrained/dog --minutes 4
```

이어 학습하려면 캐릭터의 모터 수와 observation 구조가 기존 모델과 같아야 합니다. 구조가 달라졌다면 `--from` 없이 새로 학습합니다.

---

## 4. 자주 쓰는 명령

| 명령 | 용도 |
|---|---|
| `python 0_check.py` | 설치와 실행 환경 확인 |
| `python 1_run_walker.py` | 기본 Walker2D 움직여 보기 |
| `python 2_build_character.py --validate-only` | 캐릭터 JSON 검사 |
| `python 2_build_character.py` | JSON을 XML로 변환하고 낙하 테스트 |
| `python 3_play.py my_character` | 내 캐릭터의 관절 움직임 확인 |
| `python 4_train.py --character my_character --minutes 10` | PPO 학습 |
| `python 5_watch.py --character my_character` | 최근 학습 결과 재생 |
| `python 6_plot.py --character my_character` | 학습 곡선 비교 |

자주 사용하는 옵션은 다음과 같습니다.

- `--record`: 화면 대신 MP4와 장면 이미지를 저장 (`1_run_walker.py`, `3_play.py`, `5_watch.py`)
- `--mode random|wiggle|zero`, `--strength 0.3`: 동작 방식과 관절 힘 조절 (`1_run_walker.py`, `3_play.py`)
- `--terrain flat|bumps|bumps_hard`: 평지 또는 장애물 지형 선택 (`1_run_walker.py`, `3_play.py`, `4_train.py`, `5_watch.py`)
- `--minutes 10`, `--steps 500000`: 학습 시간 또는 학습 단계 수의 상한 지정 (`4_train.py`)
- `--reward 경로`: 사용할 보상 파일 선택 (`4_train.py`)
- `--from 경로`: 기존 모델에서 이어서 학습 (`4_train.py`)

---

## 5. 주요 파일과 폴더

| 파일 또는 폴더 | 용도 |
|---|---|
| `setup.bat` | 가상환경 생성, 패키지 설치, 실행 점검 |
| `open_terminal.bat` | 실습용 터미널 열기 |
| `my_reward.py` | 2교시에서 수정하는 기본 보상 파일 |
| `characters/` | 캐릭터 JSON과 생성된 XML |
| `prompts/` | 손그림 변환, 보상 작성, 오류 해결용 프롬프트 |
| `rewards/` | 바로 사용할 수 있는 보상 예시 |
| `pretrained/` | 미리 학습된 예제 모델 |
| `runs/` | 학습 모델과 진행 기록 |
| `videos/` | 녹화 영상과 그래프 |

---

## 6. 문제 해결

| 증상 | 확인할 내용 |
|---|---|
| `python`을 찾을 수 없음 | Python 설치 프로그램을 다시 실행해 **Add python.exe to PATH**를 적용한 뒤 `setup.bat`을 다시 실행합니다. |
| `.venv`가 있지만 Python을 실행할 수 없음 | 기존 Python이 삭제되거나 이동된 경우입니다. `.venv`를 `.venv_old`로 바꾸고 `setup.bat`을 다시 실행합니다. 정상 동작을 확인한 뒤 이전 폴더를 삭제합니다. |
| `setup.bat`의 한글이 깨지거나 명령 오류가 연속으로 나옴 | BAT 파일은 CP949 인코딩과 CRLF 줄바꿈을 사용합니다. 편집기로 다시 저장했다면 인코딩을 `Korean (Windows 949)`, 줄바꿈을 `CRLF`로 맞춥니다. |
| 패키지 설치 실패 | 인터넷 연결과 프록시를 확인합니다. 계속 실패하면 `.venv\Scripts\python -m pip install -r requirements.txt`를 실행해 오류를 확인합니다. |
| XML 파일을 열 수 없음 | 프로젝트를 `C:\Walker2D`처럼 한글이 없는 짧은 경로로 옮깁니다. |
| 창이 열리지 않거나 OpenGL 오류 발생 | 그래픽 드라이버를 업데이트합니다. 창 표시만 문제라면 `--record`로 영상 저장을 시도합니다. |
| `2_build_character.py`에서 캐릭터 구조(spec) 오류 발생 | 오류에 표시된 마디와 항목을 `characters/my_character.json`에서 고친 뒤 `--validate-only`로 다시 검사합니다. |
| 캐릭터가 바로 넘어지거나 튕김 | 겹친 마디, 큰 관절 범위, 강한 모터가 원인일 수 있습니다. `joint_range` 또는 `options.strength`를 줄입니다. |
| 학습 결과가 미끄러지거나 이상한 자세로 이동함 | 전진 보상만 크지 않은지 확인하고 `posture`, `height`, `control` 비중을 함께 조정합니다. |
| 관측값(observation)/행동(action) 차원 오류 | 캐릭터나 보상 구조가 학습 당시와 달라진 경우입니다. 같은 구조의 모델을 사용하거나 새로 학습합니다. |
| 학습이 너무 느림 | 노트북 전원을 연결하고 다른 프로그램을 종료합니다. 필요하면 `--n-envs 4`로 줄입니다. |

---

## 7. 세부 자료

- 캐릭터 JSON 형식: [`characters/README.md`](characters/README.md)
- 손그림·보상 작성 프롬프트: [`prompts/`](prompts/)
- P1~P3 손그림 파일럿과 자연 보행 보상 실험: [`pilot/pilot_log.md`](pilot/pilot_log.md)

사용 라이브러리: [MuJoCo](https://mujoco.readthedocs.io/) · [Gymnasium Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/) · [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/)
