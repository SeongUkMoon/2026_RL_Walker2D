# Walker2D 실습 — 그려서 만들고, 강화학습으로 움직이기

종이에 그린 2D 캐릭터를 물리 시뮬레이션(MuJoCo) 안에 세우고, 강화학습(PPO)으로 "스스로 걷는 컨트롤러"를 학습시키는 실습입니다.
코드는 거의 쓰지 않습니다. 그림 → AI → JSON → 시뮬레이션, 말로 쓴 보상 → AI → 코드 → 학습의 흐름으로 진행합니다.

```
 [1교시] 종이에 그리기 ──(사진+프롬프트)──> AI ──> characters/my_character.json
             │                                             │
             └──────────── 2_build_character.py ◀──────────┘   JSON → XML, 낙하 테스트
                                     │
                             3_play.py  (무작위 action 으로 움직여 보기)

 [2교시] "이렇게 움직였으면…" ──(프롬프트)──> AI ──> my_reward.py
                                     │
                              4_train.py  (PPO 학습, 10~15분, CPU)
                                     │
                     5_watch.py (재생 / MP4)   6_plot.py (학습 곡선 비교)
```

- 대상: 게임·애니메이션·웹툰 등 콘텐츠 분야 실무자 (프로그래밍 경험 없어도 됩니다)
- 환경: Windows 노트북, Python 3.11, GPU 불필요 (CPU 로 5~15분 학습)
- 구성: 1교시 60분(물리 시뮬레이션 + 캐릭터 만들기), 2교시 60분(강화학습 + reward tuning)

---

## 1. 사전 준비 (수업 전에 각자 해 오기, 15분)

> 아래 문구를 사전 공지에 그대로 쓰셔도 됩니다.

1. **Python 3.11 설치**: https://www.python.org/downloads/release/python-3119/ 에서 *Windows installer (64-bit)* 를 내려받아 설치합니다.
   설치 첫 화면에서 **"Add python.exe to PATH" 체크**를 꼭 해 주세요.
2. **실습 폴더 내려받기**: 이 폴더(Walker2D)를 zip 으로 받아 `C:\Walker2D` 처럼 **경로에 한글이 없는 곳**에 풉니다.
3. **`setup.bat` 더블클릭**: 가상환경(.venv)을 만들고 패키지를 설치합니다 (인터넷 필요, 5~10분).
   마지막에 시뮬레이션 창이 3초 떴다가 닫히고 `모든 점검 통과!` 가 나오면 준비 완료입니다.
4. (선택) 사진을 넣을 수 있는 AI 채팅(ChatGPT, Claude, Gemini 등) 계정을 노트북에서 로그인해 둡니다.

문제가 생기면 setup.bat 창에 나온 내용을 캡처해서 보내 주세요. 대부분 "Python 이 PATH 에 없음" 또는 "경로에 한글" 문제입니다.

### 실습 중 명령 실행 방법
`open_terminal.bat` 을 더블클릭하면 가상환경이 켜진 터미널이 열립니다. 거기에 아래 명령들을 입력합니다.
(직접 열 경우: 터미널에서 `cd C:\Walker2D` → `.venv\Scripts\activate`)
모든 스크립트는 `python 파일이름.py --help` 로 옵션 설명을 볼 수 있습니다.

---

## 2. 1교시 — 물리 시뮬레이션과 내 캐릭터 만들기 (60분)

| 시간 | 내용 | 명령 |
|---|---|---|
| 0–5 | 설치 확인 | `python 0_check.py` |
| 5–15 | **기본 Walker2D 보기.** 무작위 action 이면 어떻게 되는지, 힘을 안 주면 어떻게 되는지, 관절을 하나씩 흔들면 어떻게 되는지 | `python 1_run_walker.py` <br> `python 1_run_walker.py --mode zero` <br> `python 1_run_walker.py --mode wiggle` |
| 15–25 | 캐릭터가 어떻게 정의되는지 설명: 마디(segment, 캡슐) + 관절(hinge joint) + 모터(actuator). `characters/dog.json` 을 열어 보고 `python 3_play.py dog` | `python 3_play.py dog --mode wiggle` <br> `python 3_play.py ostrich` |
| 25–35 | **종이에 내 캐릭터 그리기** (옆모습, 마디마다 영문 이름, 관절에 동그라미, 발 표시, 키 m 단위) | — |
| 35–45 | **AI 로 JSON 만들기**: `prompts/01_sketch_to_json.md` 의 프롬프트 + 사진 → 받은 JSON 을 `characters/my_character.json` 에 덮어쓰기 | 메모장 / VS Code |
| 45–55 | **빌드 + 낙하 테스트 + 움직여 보기.** 오류가 나면 메시지를 AI 에게 그대로 붙여서 고치기 | `python 2_build_character.py` <br> `python 3_play.py my_character` |
| 55–60 | 서로의 캐릭터 구경, `--record` 로 영상 저장 | `python 3_play.py my_character --record` |

**AI 없이도 됩니다.** `characters/my_character.json` 은 이미 동작하는 템플릿(이족 보행)이라, 숫자(길이·두께·관절 범위)만 바꿔도 새 캐릭터가 됩니다.
`characters/README.md` 에 JSON 형식 설명이 있습니다.

> 강사 팁: 1교시가 끝날 때 `python 4_train.py --character my_character --minutes 20` 을 켜 놓고 쉬는 시간을 보내면 2교시 시작 때 자기 캐릭터의 첫 학습 결과를 볼 수 있습니다.

---

## 3. 2교시 — 강화학습으로 컨트롤러 학습 + reward tuning (60분)

| 시간 | 내용 | 명령 |
|---|---|---|
| 0–10 | **학습된 결과 먼저 보기.** 미리 학습해 둔 walker / dog / ostrich 재생. "무작위 → 걷기" 사이에 무슨 일이 있었나 | `python 5_watch.py --pretrained walker` <br> `python 5_watch.py --pretrained dog` |
| 10–20 | 강화학습 개념 3가지: state(상태) / action(행동) / reward(보상). `my_reward.py` 를 열어 지금 보상이 무엇인지 읽기 | `my_reward.py` |
| 20–25 | **내 캐릭터 학습 시작** (10~15분, 백그라운드로 돌려 놓기). 쉬는 시간에 이미 돌렸다면 결과 보기 | `python 4_train.py --character my_character --minutes 12` |
| 25–40 | **reward tuning.** 원하는 움직임을 말로 쓰고 (`prompts/02_reward_from_text.md`) AI 가 만든 `my_reward.py` 로 교체. 또는 가중치 W 만 바꾸기, 또는 `rewards/` 의 예시 사용. 두 번째 터미널을 열어 다른 이름으로 학습 | `python 4_train.py --character my_character --reward my_reward_v2 --minutes 10` <br> `python 4_train.py --character my_character --reward rewards/reward_jump.py --minutes 10` |
| 40–50 | 결과 비교: 재생 + 학습 곡선 | `python 5_watch.py --character my_character` <br> `python 6_plot.py --character my_character` |
| 50–60 | 발표/토론: 어떤 보상이 어떤 움직임을 만들었나. 지형 옵션도 시도 | `python 5_watch.py --character my_character --terrain bumps` |

**시간이 부족할 때의 지름길**: 미리 학습된 모델에서 이어서(fine-tuning) 학습하면 reward 변경 효과를 3~5분 만에 볼 수 있습니다.
```
python 4_train.py --character dog --from pretrained/dog --minutes 4        (my_reward.py 를 바꾼 뒤)
```

**기대치**: 2코어 CPU 기준으로 dog 는 1~2분, walker 는 5분이면 앞으로 걷기 시작합니다. 노트북(4~8코어)은 이보다 빠릅니다.
사람이 보기 좋은 "자연스러운" 걸음은 훨씬 오래 걸리며, 보상 설계가 곧 움직임의 스타일을 정한다는 것을 보는 것이 목표입니다.

---

## 4. 파일 안내

| 파일 | 하는 일 |
|---|---|
| `setup.bat` | Python 3.11 확인 → `.venv` 생성 → 패키지 설치 → 점검 |
| `open_terminal.bat` | 가상환경이 켜진 터미널 열기 |
| `0_check.py` | 설치 점검 (라이브러리, 물리 시뮬, 화면 창) |
| `1_run_walker.py` | 기본 Walker2D 를 무작위/사인파/무동작으로 움직여 보기 |
| `2_build_character.py` | `characters/*.json` → XML 변환 + 낙하 테스트 (videos/*_drop.mp4) |
| `3_play.py` | 내 캐릭터 움직여 보기, 상태값(state) 콘솔 출력 |
| `4_train.py` | PPO 학습. `runs/<run>/` 에 model.zip, 진행 기록, 결과 영상 저장 |
| `5_watch.py` | 학습 결과 재생 (`--pretrained`, `--character`, `--run`, `--record`) |
| `6_plot.py` | 학습 곡선 비교 그래프 (videos/learning_curves.png) |
| `my_reward.py` | **2교시에 수정하는 파일.** reward / termination / (선택) 추가 observation |
| `rewards/` | 바로 써 볼 수 있는 reward 예시 5개 (천천히 꼿꼿하게 / 점프 / 제자리 서기 / 뒤로 걷기 / 머리 닿으면 실패). `--reward rewards/reward_jump.py` |
| `characters/` | 캐릭터 JSON spec 과 생성된 XML. `walker`(기본), `biped`, `dog`, `ostrich`, `my_character`(템플릿) |
| `prompts/` | AI 에게 줄 프롬프트 카드 (스케치→JSON, 말→reward 코드, 오류 해결) |
| `pretrained/` | 미리 학습된 모델 (walker, biped, dog, ostrich). `5_watch.py --pretrained 이름`, `4_train.py --from pretrained/이름` |
| `walker_rl/` | 내부 코드 (JSON→XML 변환기, 환경, 학습 도우미). 읽어 보기만 하면 됩니다 |
| `runs/`, `videos/` | 학습 결과와 영상이 저장되는 곳 (자동 생성) |

### 자주 쓰는 옵션
- `--record` : 창 대신 `videos/` 에 MP4 + 장면 PNG 저장 (1_run_walker, 3_play, 5_watch)
- `--mode random|wiggle|zero`, `--strength 0.3` : 움직임 종류와 세기 (1_run_walker, 3_play)
- `--terrain flat|bumps|bumps_hard` : 평지 / 장애물 지형 (모든 스크립트)
- `--minutes 10` 또는 `--steps 500000` : 학습 예산 (4_train)
- `--reward my_reward_v2` : 다른 reward 파일로 학습 (4_train). 파일은 프로젝트 폴더에 두세요
- `--from pretrained/dog` : 이어서 학습 (4_train)

---

## 5. 캐릭터 JSON 형식 (요약)

```json
{
  "name": "dog",
  "segments": [
    {"name": "torso", "parent": null,    "start": [-0.4, 0.6], "end": [0.4, 0.6], "radius": 0.10},
    {"name": "head",  "parent": "torso", "start": [0.4, 0.6],  "end": [0.6, 0.8], "radius": 0.07, "joint_range": [-20, 20], "motor": false},
    {"name": "front_upper", "parent": "torso",       "start": [0.35, 0.55], "end": [0.30, 0.30], "radius": 0.04, "joint_range": [-70, 70]},
    {"name": "front_lower", "parent": "front_upper", "start": [0.30, 0.30], "end": [0.35, 0.05], "radius": 0.035, "joint_range": [-90, 30], "friction": 1.9}
  ]
}
```
- 옆에서 본 그림. **x 오른쪽 = 앞(전진 방향), z 위**, 단위 m. 바닥 높이는 자동으로 맞춰집니다.
- 마디(segment)는 캡슐. `start` = 부모에 붙는 관절 위치, `end` = 끝점. `parent: null` 인 몸통(torso)이 정확히 하나.
- `joint_range` 는 관절 회전 범위(deg), `motor: false` 는 힘 없이 흔들리는 관절, 발에는 `friction: 1.9`.
- 자세한 설명과 흔한 실수: `characters/README.md`

## 6. reward 수정 (요약)

`my_reward.py` 의 가중치 `W` 를 바꾸거나 `compute_reward()` 를 고칩니다. 쓸 수 있는 값은 `state.x_vel`(전진 속도), `state.height_ratio`(높이 비율), `state.angle`(기울기), `state.energy`(힘 사용량), `state.touching("head")`(바닥 접촉) 등이며 파일 상단에 정리되어 있습니다.
말로 쓴 보상을 코드로 바꾸는 프롬프트는 `prompts/02_reward_from_text.md` 에 있습니다.

---

## 7. 자주 나오는 문제

| 증상 | 해결 |
|---|---|
| `python` 을 찾을 수 없다 / 'python'은 내부 또는 외부 명령이 아닙니다 | Python 설치 시 "Add to PATH" 를 안 한 것. 재설치(Modify → Add to PATH) 후 setup.bat 다시 실행 |
| setup.bat 의 한글이 깨져 보인다 / `'defined'은(는) 내부 또는 외부 명령...` 같은 오류가 여러 줄 나온다 | bat 파일은 한국어 Windows 콘솔 인코딩(CP949)으로 저장되어 있습니다. 메모장 등으로 열어 **UTF-8 로 다시 저장하면 이 오류가 납니다** — GitHub 에서 받은 원본을 그대로 쓰세요. 시스템 로캘을 'Beta: UTF-8' 로 바꿔 둔 PC 에서는 글자만 깨지고 동작은 정상입니다 |
| setup.bat 에서 패키지 설치 실패 | 인터넷/회사 프록시 문제. 핫스팟으로 다시 시도. 그래도 안 되면 `.venv\Scripts\python -m pip install -r requirements.txt` 를 터미널에서 직접 실행해 오류 확인 |
| XML 을 열 수 없다는 오류 (경로 관련) | 폴더 경로에 한글/공백이 있는 경우. `C:\Walker2D` 로 옮기기 |
| 창이 안 뜬다 / OpenGL 오류 | 그래픽 드라이버 업데이트. 안 되면 `--record` 로 영상 저장 방식으로 진행 (실습 가능) |
| `2_build_character.py` 에서 `[실패] spec 오류` | 메시지가 어느 마디의 무엇이 잘못됐는지 알려 줍니다. 메시지 + JSON 을 AI 에게 붙여 "고쳐 달라" 하면 됩니다 |
| 캐릭터가 바로 넘어지거나 튕겨 나간다 | 마디가 서로 겹치게 그려졌거나 관절 범위가 너무 큼. `--mode zero` 로 떨어뜨려 보고 `joint_range` 를 좁히거나 `options.strength` 를 0.5 로 |
| 학습해도 가만히 서 있기만 한다 | `W["alive"]` 가 상대적으로 큼. `W["forward"]` 를 키우거나 `alive` 를 0.5 로. 또는 더 오래 학습 |
| 학습 결과가 이상한 자세로 미끄러져 간다 | 보상이 "앞으로만 가면 됨" 이라서 정직하게 최적화한 결과입니다. `posture`, `height` 항을 켜서 자세를 요구하세요 (2교시의 핵심 토론 주제) |
| `5_watch.py` 에서 observation 차원이 다르다는 오류 | 학습 후에 캐릭터 JSON 이나 `extra_observation` 이 바뀜. 다시 학습하거나 원래대로 되돌리기 |
| 학습이 너무 느리다 (steps/s 가 1000 미만) | 노트북 전원 연결(고성능 모드), 다른 프로그램 종료. `--n-envs 4` 로 줄여 보기 |

---

## 8. 참고
- 물리 엔진: [MuJoCo](https://mujoco.readthedocs.io/) · 환경: [Gymnasium Walker2d-v5](https://gymnasium.farama.org/environments/mujoco/walker2d/) (`xml_file` 인자로 임의의 캐릭터 XML 을 받습니다) · 강화학습: [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/)
- 이전 실습 프로젝트(2025 DS RL, Walker2D)의 wrapper 구조와 bump terrain 아이디어를 참고했습니다.
