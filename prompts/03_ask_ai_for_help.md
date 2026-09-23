# 프롬프트 카드 03 — 오류가 났을 때 AI 에게 묻는 법

오류 메시지는 무섭게 보이지만, 그대로 복사해서 AI 에게 주면 대부분 해결됩니다. 아래 순서로 정보를 주면 정확도가 올라갑니다.

```
Windows 11, Python 3.11, 가상환경(.venv) 에서 MuJoCo + Gymnasium + Stable-Baselines3 로 2D 캐릭터 강화학습 실습을 하고 있다.
아래 명령을 실행했더니 오류가 났다. 원인과 해결 방법을 단계별로 알려 줘. 파일을 고쳐야 하면 고친 전체 내용을 보여 줘.

[실행한 명령]
python 2_build_character.py

[터미널에 나온 내용 전체]
{터미널 창 내용을 처음부터 끝까지 복사해서 붙이기}

[관련 파일 내용]  (JSON 오류면 characters/my_character.json, reward 오류면 my_reward.py)
{파일 내용 붙이기}
```

## 터미널 내용 복사하는 법
터미널 창 안에서 마우스로 드래그해 선택하고 `Enter` (또는 마우스 오른쪽 클릭) → 복사됩니다. 전체를 선택하려면 창 제목을 오른쪽 클릭 → 편집 → 모두 선택.

## 자주 나오는 오류와 뜻

| 오류 문구 | 뜻 | 보통의 해결 |
|---|---|---|
| `JSON 문법 오류` / `Expecting ',' delimiter` | JSON 에 쉼표·따옴표·괄호가 빠지거나 남음 | 메시지의 line/column 근처를 확인. AI 에게 파일 전체를 주고 고쳐 달라고 |
| `parent "xxx" 라는 segment 가 없습니다` | 이름 오타 | 마디 이름과 parent 이름을 똑같이 |
| `ModuleNotFoundError: No module named 'mujoco'` | 가상환경이 안 켜졌거나 설치가 안 됨 | `open_terminal.bat` 으로 터미널을 열었는지 확인, 아니면 `setup.bat` 다시 |
| `compute_reward 가 숫자가 아닌 값을 돌려줬습니다` | reward 함수가 return 을 빼먹었거나 배열을 돌려줌 | `return float(r)` 로 끝나는지 확인 |
| `observation 차원이 다릅니다` | 관측 길이가 다른 reward/model을 `--from`으로 연결했거나 snapshot이 없는 구형 run과 현재 파일이 다름 | `--from`은 observation/action 차원이 같을 때만 사용. 다르면 새로 학습하고, 재생은 가능하면 `--run`으로 저장 snapshot 사용 |
| 창이 안 뜨고 `GLFW`, `OpenGL` 오류 | 그래픽 드라이버/원격 데스크톱 문제 | `--record` 옵션으로 영상 저장 방식 사용 |
