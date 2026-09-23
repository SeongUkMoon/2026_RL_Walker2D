# 손그림 변환 파일럿 로그

원본 변환 JSON을 먼저 보존하고, 수정하지 않은 상태로 `--validate-only`를 실행한다.
검증 실패 시 수정본을 별도 파일로 남기고 수정 항목 수를 기록한다.

> `runs/`와 `videos/demo/` 밖의 영상은 로컬 실험 산출물이며 `.gitignore`에 따라 GitHub 저장소에는 포함되지 않는다. 아래 경로는 같은 작업 PC에서 결과를 추적하기 위한 기록이다.

## P1 — 기준형 이족 캐릭터

- 상태: 정적 변환 완료 / 물리 낙하·관절 구동 테스트 통과 / 사용자 시각 확인 완료
- 원본: `p1_source.png`
- 원본 크기: 2252 × 4000, PNG RGBA
- SHA-256: `DC5EF90D402F28A147D984994C6D8A04DB26C69B4EEC3E3FA9E74D0199CDA651`
- 촬영 품질: 통과 — 전체 그림, 라벨, 전진 방향, 높이 표시를 판독할 수 있음
- 판독한 마디: `torso`, `thigh_l`, `shin_l`, `foot_l`, `thigh_r`, `shin_r`, `foot_r`
- 판독한 구조: root 1개, 다리 2개, 마디 7개, 모터 후보 관절 6개, 발 2개
- 전진 방향 / 전체 높이: 오른쪽 / 1.2 m
- 명확화 질문 수: 3 (변환 전 2개, 변환 후 구조 확인 1개)
- 질문 1: 무릎의 양방향 화살표가 양방향 회전인지, 사람 무릎처럼 한 방향 굽힘인지
- 질문 2: `foot_l`이 뒤쪽(왼쪽), `foot_r`이 앞쪽(오른쪽)을 향하는 것이 의도인지, 두 발 모두 앞쪽인지
- 질문 3: 종이에서 좌우로 벌어진 두 다리를 시뮬레이터의 앞뒤 초기 보폭으로 유지할지, 옆모습처럼 같은 x 위치에 겹칠지
- 원본 JSON: `p1_raw.json`
- 원본 JSON SHA-256: `19E5FD65E9239D85F09778D347486868AB5D65E77F2265E22ACFFFBA99EF2DCA`
- 원본 JSON 검증: 통과 — 경고 0개, segment 7개, motor 6개
- 자동 수정 수: 0
- 사용자 의도 반영: 1건 — 선택 B에 따라 좌우 다리를 동일한 x 위치에 겹침
- 최종 JSON: `p1_final.json` → `../characters/pilot_p1.json`
- 최종 JSON SHA-256: `1ECC4B7D305623A1258FEB9F86ED82DAF0859253387D788A4DC1C1F84C365D23`
- 최종 JSON 검증: 통과 — 경고 0개, segment 7개, motor 6개
- 구조 충실도 확인: 완료 — 무릎 한 방향, 두 발 +x 방향, 좌우 다리 옆모습 중첩
- 물리 낙하 테스트: 통과 — Python 3.11.9 / MuJoCo 3.14.0, observation 17 / action 6, 무동작 낙상 0.82초(무렌더)·0.88초(영상 기록)
- 시각 판정: 마디 분리, 바닥 관통, 폭발적 튐 없음; 좌우 다리 중첩 유지
- 산출물: `../videos/pilot_p1_drop.mp4`, `../videos/pilot_p1_drop.png`
- 저강도 관절 구동: 통과 — `wiggle`, strength 0.25, 6초, seed 0; 첫 에피소드 0.98초·0.65 m, 자동 재시작 포함 낙상 11회
- 관절 판정: 좌우 다리 모두 구동됨; knee `[-140, 0]` 단방향 동일, hip `[-60, 60]`·ankle `[-45, 45]` 양방향; 분리·관통·폭발 없음
- 구동 산출물: `../videos/pilot_p1_wiggle.mp4`, `../videos/pilot_p1_wiggle.png`
- 사용자 시각 확인: 완료 — 낙하 및 관절 구동 영상 승인

## P2 — 중간 난도 동물형

- 상태: 정적 변환 완료 / 물리 낙하·관절 구동 테스트 통과 / 사용자 시각 확인 완료
- 원본: `p2_source.png`
- 원본 크기: 2252 × 4000, PNG RGBA
- SHA-256: `8CCA3026C846CE008A61BBFE8E871A09BEC793FF966FE8D1414BD356FBA95653`
- 촬영 품질: 통과 — 전체 그림, 라벨, free/별표, 전진 방향, 높이 표시를 판독할 수 있음
- 판독한 마디: `torso`, `head`, `tail`, `front_upper`, `front_lower`, `back_upper`, `back_lower`
- 판독한 구조: root 1개, free 마디 2개, 모터 후보 관절 4개, 접지 마디 2개
- 전진 방향 / 전체 높이: 오른쪽 / 0.8 m
- 명확화 질문 수: 3
- 질문 1: torso-upper hip은 양방향이고 upper-lower knee만 한 방향인지, 그림의 네 다리 관절을 모두 단방향으로 의도했는지
- 질문 2: 앞·뒤 knee가 같은 방향으로만 굽는지, 서로 반대 방향으로 굽는지
- 질문 3: 별표가 lower 마디 자체의 접지 표시인지, 별도 foot 마디를 추가하라는 표시인지
- 원본 JSON: `p2_raw.json`
- 원본 JSON SHA-256: `C127629CB7F30A708B5E1C3322439F2B07FE2394C548560E198FAF2815D3EFD9`
- 원본 JSON 검증: 통과 — 경고 0개, segment 7개, motor 4개
- 자동 수정 수: 0
- 사용자 의도 반영: 질문 답변만 반영, JSON 생성 후 변경 0건
- 수치 확인: 외곽 높이 0.795 m, 최저점 0.000 m
- 최종 JSON: `p2_final.json` → `../characters/pilot_p2.json`
- 최종 JSON SHA-256: `C127629CB7F30A708B5E1C3322439F2B07FE2394C548560E198FAF2815D3EFD9`
- 최종 JSON 검증: 통과 — 경고 0개, segment 7개, motor 4개
- 구조 충실도 확인: 완료 — head/tail 수동, hip 양방향, knee 같은 단방향, lower 자체 접지
- 물리 낙하 테스트: 통과 — observation 17 / action 4, 무동작 낙상 0.46초
- 시각 판정: head/tail 연결 유지, front/back lower 직접 접지, 마디 분리·바닥 관통·폭발적 튐 없음
- 산출물: `../videos/pilot_p2_drop.mp4`, `../videos/pilot_p2_drop.png`
- 낙하 사용자 시각 확인: 완료
- 저강도 관절 구동: 통과 — `wiggle`, strength 0.25, 6초, seed 0; 첫 에피소드 0.99초·-0.50 m, 자동 재시작 포함 낙상 6회
- 관절 판정: front/back 체인 모두 구동됨; knee `[-110, 0]` 단방향 동일, hip `[-70, 70]` 양방향; head/tail은 actuator 없이 수동 추종; 분리·관통·폭발 없음
- 구동 산출물: `../videos/pilot_p2_wiggle.mp4`, `../videos/pilot_p2_wiggle.png`
- 관절 구동 사용자 시각 확인: 완료

## P3 — 도전형 조류 캐릭터

- 상태: 정적 변환 완료 / 물리 낙하·관절 구동 테스트 통과 / 사용자 시각 확인 완료
- 원본: `p3_source.png`
- 원본 크기: 1400 × 786, PNG RGB
- SHA-256: `DAFD44CE48FB33B3A0A9301E84FC108DFAC4D9FD613DE92C25C195C3C6A0241B`
- 촬영 품질: 통과 — 전체 그림, 라벨, free/별표, x2, 전진 방향, 높이 표시를 판독할 수 있음
- 판독한 원본 마디: `torso`, `neck`, `head`, `upper_leg`, `lower_leg`, `foot`
- x2 확장 후 구조: segment 9개, motor 6개, 수동 마디 2개, 접지 마디 2개
- 전진 방향 / 전체 높이: 오른쪽 / 1.5 m
- 명확화 질문 수: 0
- 원본 JSON: `p3_raw.json`
- 원본 JSON SHA-256: `2389D9FBC1087F648992ACBF09C38B2354B2B59819D3923AC69077F0E9F251DB`
- 원본 JSON 검증: 통과 — 경고 0개, segment 9개, motor 6개
- 자동 수정 수: 0
- 수치 확인: 외곽 높이 1.490 m, 최저점 0.000 m
- 복제 확인: 좌우 start/end/radius/joint_range/motor/friction 동일, 각 parent 체인 정상
- 독립 감사: 치명적 불일치 없음
- 최종 JSON: `p3_final.json` → `../characters/pilot_p3.json`
- 최종 JSON SHA-256: `2389D9FBC1087F648992ACBF09C38B2354B2B59819D3923AC69077F0E9F251DB`
- 최종 JSON 검증: 통과 — 경고 0개, segment 9개, motor 6개
- 구조 충실도 확인: 완료 — neck/head 수동, hip/ankle 양방향, knee 단방향, 동일 위치 좌우 복제
- 물리 낙하 테스트: 통과 — observation 21 / action 6, 무동작 낙상 0.54초
- 시각 판정: 좌우 다리 동일 위치 중첩, 두 foot 전진 방향 접지, neck/head 연결 유지, 마디 분리·바닥 관통·폭발적 튐 없음
- 경미한 관찰사항: lower-leg 끝과 ankle 중심이 0.08 m 어긋나지만 캡슐 반경 합 0.105 m 안에서 겹쳐 연결·물리에 이상 없음; 현 단계 수정 보류
- 산출물: `../videos/pilot_p3_drop.mp4`, `../videos/pilot_p3_drop.png`
- 낙하 사용자 시각 확인: 완료
- 저강도 관절 구동: 통과 — `wiggle`, strength 0.25, 6초, seed 0; 첫 에피소드 1.14초·-0.91 m, 자동 재시작 포함 낙상 5회
- 관절 판정: 좌우 다리 6개 actuator 모두 구동됨; knee `[-140, 0]` 단방향 동일, hip `[-70, 70]`·ankle `[-45, 45]` 양방향; neck/head은 actuator 없이 수동 추종; 분리·관통·폭발 없음
- 구동 산출물: `../videos/pilot_p3_wiggle.mp4`, `../videos/pilot_p3_wiggle.png`
- 관절 구동 사용자 시각 확인: 완료

## 정적 단계 요약

| 항목 | P1 | P2 | P3 | 합계 |
|---|---:|---:|---:|---:|
| 사진 판독 통과 | 1 | 1 | 1 | 3/3 |
| 명확화 질문 수 | 3 | 3 | 0 | 6 |
| 최초 JSON 검증 통과 | 1 | 1 | 1 | 3/3 |
| 검증기 자동 수정 | 0 | 0 | 0 | 0 |
| 생성 후 사용자 의도 변경 | 1 | 0 | 0 | 1 |
| 1차 물리 낙하 테스트 통과 | 1 | 1 | 1 | 3/3 |
| 저강도 관절 구동 테스트 통과 | 1 | 1 | 1 | 3/3 |

- 질문 없이 바로 변환 가능: 1/3
- 질문 후 구조 확정 및 JSON 생성 성공: 3/3
- 최초 JSON의 정적 검증 무수정 통과: 3/3
- MuJoCo 환경 자가점검 통과: Python 3.11.9, MuJoCo 3.14.0, Gymnasium 1.3.0, Stable-Baselines3 2.9.0
- P1은 낙하 안정성·낙하 시간·양쪽 다리 구동·관절 제한 방향을 확인함
- P2는 낙하 안정성·lower 직접 접지·head/tail 수동 추종·앞뒤 다리 관절 제한 방향을 확인함
- P3는 낙하 안정성·동일 위치 좌우 복제·neck/head 수동 추종·양쪽 다리 관절 제한 방향을 확인함

## 파일럿 최종 결론

| 캐릭터 | 정적 변환 | 무동작 낙하 | 저강도 관절 구동 | 사용자 영상 확인 | 최종 판정 |
|---|---|---:|---|---|---|
| P1 기준형 이족 | 통과 | 0.82~0.88초 | 통과 | 완료 | 통과 |
| P2 동물형 | 통과 | 0.46초 | 통과 | 완료 | 통과 |
| P3 조류형 | 통과 | 0.54초 | 통과 | 완료 | 통과 |

- 파일럿 성공 범위: 라벨, 부모-자식 연결, 전진 방향, 전체 높이, 관절 의도를 표시한 구조화된 손그림 3장 모두 JSON 변환·검증·MuJoCo 빌드·낙하·저강도 구동에 성공했다.
- 판정 강도: 이번 통과는 짧은 실행과 영상·사용자 육안 확인에 기반한 기본 실행 가능성/구조 일관성 판정이다. 사전에 고정한 정량 성능 기준에 따른 보행 성능 검증은 아니다.
- 사람 확인 필요성: P1/P2는 각각 3개의 명확화 답변이 필요했고 P1은 생성 후 중첩 구조를 1회 수정했다. 따라서 모호한 화살표·좌우 배치·접지 표시는 자동 변환만으로 확정하지 않는다.
- 아직 입증하지 않은 범위: 라벨 없는 자유화, 심한 원근·가림·흐림, 복수 시점 그림, 3차원 구조, 다중 seed·다른 지형에서의 안정적인 보행 일반화는 이번 결과에 포함되지 않는다.
- 물리 모델 한계: 질량·반경·모터 힘은 비례 추정치이고 self-collision이 비활성화되어 있다. 겹친 다리 구현에는 적합하지만 실제 기구 충돌 재현으로 해석하면 안 된다.
- 학습 위험: P1/P3의 좌우 다리가 동일 좌표에 겹쳐 있어 두 actuator 체인이 물리적으로 중복된다. P1 단일 **학습 seed**에서는 전진 gait를 학습했지만 대칭성·접촉·제어 효율의 다중 학습 seed 강건성과 P3 학습은 아직 확인하지 않았다.
- 보류 사항: P3의 lower-leg 끝과 ankle 중심 0.08 m 차이는 현재 물리·영상에 문제가 없어 유지하되, 향후 자동 연결 오차 임계값 회귀검사 대상으로 둔다.
- 당시 다음 권장 검증(아래에서 완료): P1의 빠른 달리기 gait를 사용자 영상으로 확인한 뒤, 목표 속도·자세 가중치를 조정한 별도 run을 만들고 여러 평가 seed로 비교한다.

## P1 RL 학습 스모크 테스트

- 상태: 학습·저장·재로드 파이프라인 통과 / 보행 성능 미확보
- 실행 예산: PPO 16,384 steps, 최대 2분, 병렬 환경 8개, seed 0, flat terrain
- run: `../runs/pilot_p1__my_reward__smoke__20260924-000317`
- 실행 결과: 6.2초, 16,384 steps 완료, 마지막 평균 reward 41.1, 평균 에피소드 51.9 steps, 평균 전진 -0.08 m
- 저장 검증: `model.zip`, `vecnormalize.pkl`, `config.json`, `result.json`, `progress.csv`, reward 및 character XML/meta/source snapshot 생성 완료
- 재로드 검증: 학습 당시 XML/meta SHA-256 확인, observation/action 17/6 일치, 결정론적 5초 rollout 및 MP4/PNG 생성 성공
- 재생 결과: 첫 에피소드 0.20초·0.09 m·reward 36.4, 5초 동안 낙상 25회
- 수치 검증: 정책 파라미터의 NaN/inf 0건, 재로드 예측 action 6개 모두 유한
- 산출물: `../videos/pilot_p1__my_reward__smoke__20260924-000317_watch.mp4`, `../videos/pilot_p1__my_reward__smoke__20260924-000317_watch.png`
- 판정: 코드·환경·PPO·정규화·저장·재로드·재생 경로는 정상이다. 16,384 steps 정책은 서 있거나 걷는 수준이 아니며, 보행 가능성은 이 시험으로 입증되지 않았다.

## P1 5분 학습 테스트

- 상태: 기술 게이트 통과 / 단일 seed·평지 평가에서 전진 gait 확보
- 실행 조건: 처음부터 새 PPO 학습, 5분, 병렬 환경 8개, seed 0, flat terrain, `my_reward`, `from=null`
- run: `../runs/pilot_p1__my_reward__five_min__20260924-000822`
- 학습 결과: 300.0초, 373,312 steps, 마지막 평균 reward 1,719.6, 평균 에피소드 474.6 steps, 평균 전진 9.96 m
- 학습 추세: 30초의 reward 109.4 / 92 steps / 0.14 m에서 300초의 1,719.6 / 474.6 steps / 9.96 m로 상승
- 저장 검증: 최종 `model.zip`, `vecnormalize.pkl`, config/result/progress, reward·character snapshot 및 145,480-step 중간 체크포인트 쌍 생성 완료
- 자동 평가(seed 1, deterministic): 첫 에피소드 1,000 steps·8.0초(TimeLimit)·24.80 m·reward 4,097.8, 10초 동안 낙상 0회
- 재로드 검증: 새 프로세스에서 최종 model/normalizer와 character snapshot SHA를 불러와 자동 평가와 동일한 결과를 재현
- 수치 검증: 정책 파라미터 NaN/inf 0건, 예측 action 6개 모두 유한; observation/action 17/6 일치
- 영상 판정: 좌우 다리 반복 교대, 몸통 건강 높이·각도 유지, 기어가기·고정 활주·폭발적 점프 없음. 느린 보행보다는 빠른 달리기/바운딩 gait에 가까움
- 산출물: `../runs/pilot_p1__my_reward__five_min__20260924-000822/final.mp4`, `../runs/pilot_p1__my_reward__five_min__20260924-000822/final.png`, `../videos/pilot_p1__my_reward__five_min__20260924-000822_watch.mp4`, `../videos/pilot_p1__my_reward__five_min__20260924-000822_watch.png`
- 한계: 단일 학습 seed, 평지, 단일 평가 seed, 10초만 확인했다. 현재 reward는 목표 속도·posture·height 항이 꺼져 있어 최대 전진 속도를 선호하므로, 안정적인 일반 보행이나 다른 seed·지형에 대한 강건성은 아직 입증되지 않았다.

## P1 자연 보행 reward 튜닝

- 목표: 빠른 바운딩/점프형 전진을 약 0.75 m/s의 양발 교대 보행으로 바꾸고, 체공·발 끌림·접지 미끄럼·좌우 비대칭을 순차적으로 줄인다.
- 공통 조건: flat terrain, PPO, 병렬 환경 8개, 학습 seed 0, 이전 버전 체크포인트에서 5분 fine-tuning, 결정론적 8초 에피소드 평가.
- 관측 확장: gait phase `sin/cos`와 좌우 foot contact 4개를 추가하여 observation 21 / action 6으로 유지.
- 환경 확장: reward가 `USES_SEGMENT_KINEMATICS = True`를 선언할 때만 각 segment body-origin의 world position/velocity를 `State`에 제공. 기존 reward와 17차원 모델에는 영향 없음.
- 안전 검증: 전체 `unittest` 54개 통과, 각 최종 모델 저장·재로드 및 character snapshot SHA 확인, 모델 파라미터·정규화·action NaN/inf 0건.

| 버전 | 핵심 변경 | seed 1 평가 | 정량/영상 판정 |
|---|---|---|---|
| 기존 `my_reward` | 전진속도 중심 | 8초 24.80 m, 낙상 0 | 빠른 달리기/바운딩. 체공률 57.9%, 자연 보행 아님 |
| v1 | 목표속도·자세·높이·phase contact·체공 억제 | 8초 6.27 m, 낙상 0 | 속도 0.784 m/s, 체공 0.2%, 주 접지 교대 100%. 그러나 한 발이 99% 계속 앞에 있고 toe-tap/활주로 contact만 교대 |
| v2 | world ankle lead 교대·swing clearance·stance slip | 8초 5.55 m, 낙상 0 | lead 17회 교대, 접촉점 slip 평균 36% 감소. 그러나 한쪽 lead가 과도하고 반대쪽 swing이 평균 0.7 cm·접촉 17%인 비대칭 |
| v3 | 목표 pose와 일치하는 ankle lead 궤적, 좌우 pose 기하평균, swing-contact 강화 | 8초 5.77 m, 낙상 0 | 양발 mid-swing 9.18/8.79 cm, 접촉 0/0%, lead 17회 교대. 잔여 보폭 비대칭과 ankle-origin 움직임은 있으나 전체 영상 품질 최상 |
| v4 | 개별 torso-relative ankle·hip·knee robust 추적, stance slip 강화 | 8초 5.77 m, 낙상 0 | 실제 접촉점 slip은 0.097→0.070 m/s로 개선했지만 한쪽 swing 9.18→4.62 cm, lead 대칭비 0.296→0.257로 퇴보. 미채택 |

### 최종 채택 — v3

- reward: `../rewards/reward_p1_natural_walk_v3.py`
- test: `../tests/test_reward_p1_natural_walk_v3.py`
- run: `../runs/pilot_p1__reward_p1_natural_walk_v3__natural_v3_5min__20260924-014146`
- 재생 영상: `../videos/pilot_p1__reward_p1_natural_walk_v3__natural_v3_5min__20260924-014146_watch.mp4`
- 학습: 300초, 이번 run 524,512 steps, 누적 1,592,224 steps.
- 평가 seed 0~4: 모두 8초 1,000 steps 완주, 10초 낙상 0회, 전진거리 5.77~5.90 m.
- seed 1 gait: 평균속도 0.722 m/s, 체공률 0.2%, 양발 lead 17회 교대, 주 접지 교대 100%, 발 이외 접촉 0.
- 양발 mid-swing: 상대 높이 9.18/8.79 cm, p10 2.17/5.38 cm, swing 접촉률 0/0%.
- 안정성: 몸통 높이 표준편차 0.0305 m, 수직속도 RMS 0.206 m/s, 몸통 각도 평균/최대 3.42/8.94°, 평균 energy 1.166.
- 남은 한계: 좌우 lead p90 0.127/0.430 m(대칭비 0.296)로 보폭 비대칭이 남고, stance ankle 속도는 실제 접촉점 slip과 발 구르기 성분을 함께 포함한다. 따라서 실제 사람 보행 재현이나 완전한 대칭 gait로 해석하지 않는다.

### 보존한 비교 실험 — v4

- reward/run: `../rewards/reward_p1_natural_walk_v4.py`, `../runs/pilot_p1__reward_p1_natural_walk_v4__natural_v4_5min__20260924-020210`
- 재생 영상: `../videos/pilot_p1__reward_p1_natural_walk_v4__natural_v4_5min__20260924-020210_watch.mp4`
- 평가 seed 0~4 모두 낙상 없이 5.77~5.87 m 전진했으나, 미끄럼 개선과 맞바꾼 좌우 swing/보폭 회귀 때문에 최종 기준으로 채택하지 않았다.
- 중단 기준: v4가 사전 비회귀 기준을 넘지 못했으므로 자동 reward 반복 튜닝을 여기서 끝낸다. 다음 변경은 v3/v4 영상을 사람이 비교해 스타일 우선순위를 정한 뒤 별도 실험으로 수행한다.
