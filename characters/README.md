# characters/ — 캐릭터 JSON spec 설명

| 파일 | 설명 |
|---|---|
| `walker.xml`, `walker.meta.json` | Gymnasium 기본 Walker2d (XML 원본). `1_run_walker.py` 가 사용 |
| `biped.json` | 기본 Walker2d 를 JSON spec 으로 다시 그린 이족 보행 예제 |
| `dog.json` | 사족(옆모습이라 다리 2개) 예제. 모터 4개 |
| `ostrich.json` | 타조. 3마디 다리, 목/머리는 모터 없음. 모터 6개 |
| `my_character.json` | **내 캐릭터 템플릿** (biped 와 같은 구조, 설명 주석 포함). 이 파일을 수정하거나 AI 가 만든 JSON 으로 덮어씁니다 |
| `pilot_p1.json`, `pilot_p2.json`, `pilot_p3.json` | 손그림 파일럿에서 확정한 기준형 이족·동물형·조류형 캐릭터. 판독과 검증 기록은 `../pilot/pilot_log.md` |
| `*.xml`, `*.meta.json` | `2_build_character.py` 가 만드는 생성물 (직접 수정하지 않습니다) |
| `_generated/` | 지형(terrain) 옵션용으로 자동 생성되는 XML |

## 좌표계

```
        z (위)
        ▲
        │      ●  head
        │     ╱
        │  ━━━━━━━━  torso         옆에서 본 그림.
        │  ┃      ┃                x 오른쪽(+) = 캐릭터의 앞 = 전진 방향
        │  ┃      ┃  thigh         z 위(+),  단위 m
        │  ┃      ┃  shin          바닥 z=0 은 자동으로 맞춰지므로 대충 그려도 됩니다.
   ─────┴──┻━━━━━━┻━━━▶ x (앞)
          foot (friction 1.9)
```

## segment(마디) 필드

| 필드 | 필수 | 뜻 | 예 |
|---|---|---|---|
| `name` | O | 영문 이름 (소문자/숫자/밑줄). 고유해야 함 | `"thigh_r"` |
| `parent` | O | 붙을 부모 마디 이름. 몸통(root)만 `null` (정확히 1개) | `"torso"` |
| `start` | O | `[x, z]` 시작점 = 부모와 연결되는 관절(hinge joint) 위치. 부모 마디 위의 점이어야 자연스럽습니다 | `[0.0, 1.05]` |
| `end` | O | `[x, z]` 끝점 | `[0.0, 0.60]` |
| `radius` | O | 캡슐 두께(반지름, m). 0.03~0.12 | `0.05` |
| `joint_range` | | 관절 회전 범위 `[min, max]` (deg). 기본 `[-90, 90]`. 무릎처럼 한쪽으로만: `[-150, 0]` | `[-150, 0]` |
| `motor` | | 이 관절에 모터(actuator)를 달지. 기본 `true`. 머리/꼬리는 `false` | `false` |
| `friction` | | 바닥 마찰. 기본 0.9. 발은 1.5~2.0 | `1.9` |
| `color` | | `brown` `purple` `blue` `red` `green` `gray` `yellow` `white` `black` 또는 `"r g b a"` | `"blue"` |

`"_설명"` 처럼 `_` 로 시작하는 항목은 무시되므로 메모용으로 자유롭게 쓸 수 있습니다.

## options (선택)

```json
"options": {"strength": 1.0, "fall_height_ratio": 0.55, "fall_angle_deg": 60}
```
- `strength` : 모터 힘 배율. 힘이 남아 튕기면 0.5, 다리가 무거워 못 들면 1.5
- `fall_height_ratio` : 몸통 높이가 처음의 이 비율 아래로 내려가면 '넘어짐'(에피소드 종료). 낮게 기는 캐릭터는 0.4
- `fall_angle_deg` : 몸통이 이 각도(deg) 이상 기울면 '넘어짐'. 구르는 캐릭터를 원하면 크게

## 변환기가 자동으로 해 주는 것

- 가장 낮은 마디가 바닥(z=0)에 닿도록 전체를 위/아래로 이동
- 몸통에 root joint 3개(rootx, rootz, rooty) 추가 → Gymnasium `Walker2d-v5` 에 `xml_file` 로 바로 로드 가능
- 모터 힘(gear)을 캐릭터 총 질량에 비례해 설정 (기본 Walker2d 23.7 kg ↔ gear 100)
- 마디 사이 충돌은 꺼져 있고(겹쳐도 됨), 바닥/장애물과만 충돌
- 카메라가 몸통을 따라감

## 흔한 실수

| 증상 | 원인 → 해결 |
|---|---|
| `parent 가 null 인 segment 가 정확히 1개` 오류 | 몸통이 없거나 두 개. 하나만 `null` |
| 마디가 허공에 떠서 붙어 있음 (경고) | 자식의 `start` 가 부모 마디 위에 있지 않음. 부모의 `end` 나 중간 점으로 옮기기 |
| 힘을 안 줘도 바로 튕겨 나감 | 마디 두께가 너무 커서 바닥에 파고들었거나 관절 범위가 초기 자세를 벗어남. `joint_range` 가 0을 포함하게 |
| 학습이 되어도 못 일어남 | 다리 대비 몸통이 너무 무거움. 몸통 `radius` 줄이기, 발 `friction` 1.9 |
| 관절이 반대로 꺾임 | `joint_range` 의 부호를 뒤집기 (`[-150, 0]` ↔ `[0, 150]`) |
