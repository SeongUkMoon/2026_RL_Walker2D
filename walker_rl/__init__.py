"""
walker_rl - 실습용 내부 패키지 (참가자가 직접 수정할 필요는 없습니다)

- spec2mjcf.py     : 캐릭터 JSON spec -> MuJoCo XML(MJCF) 변환, 검증, terrain 추가
- character_env.py : Gymnasium Walker2d-v5 위에 캐릭터를 올리고 reward/termination hook 을 연결하는 Env
- runner.py        : 에피소드 실행(창 띄우기 / MP4·PNG 저장), 무작위·사인파 policy
- training.py      : PPO 학습 설정, 시간 예산 callback, run 폴더 저장/불러오기
- utils.py         : 경로, 로그, 공용 함수
"""
import sys as _sys

# Windows 콘솔(cp949 등)에서 출력 못 하는 글자가 있어도 프로그램이 멈추지 않게 함
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

from .utils import ROOT, CHARACTERS_DIR, RUNS_DIR, VIDEOS_DIR, PRETRAINED_DIR  # noqa: F401,E402
