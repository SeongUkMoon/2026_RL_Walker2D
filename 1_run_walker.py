"""
1교시 - 기본 Walker2D 를 움직여 보기

Gymnasium 에 들어 있는 기본 2D 캐릭터(Walker2d)를 무작위 action 으로 움직여 봅니다.
"강화학습을 하기 전의 컨트롤러는 이렇게 아무것도 모른다" 를 눈으로 확인하는 단계입니다.

사용 예
  python 1_run_walker.py                      무작위 action, 20초, 창으로 보기
  python 1_run_walker.py --mode wiggle        모터를 사인파로 흔들어 각 joint 움직임 보기
  python 1_run_walker.py --mode zero          힘을 주지 않고 중력에 맡기기
  python 1_run_walker.py --strength 1.0       action 세기 최대
  python 1_run_walker.py --record             창 대신 videos/walker_random.mp4 로 저장
  python 1_run_walker.py --terrain bumps      장애물 지형
"""
from walker_rl.play_cli import build_parser, run

if __name__ == "__main__":
    args = build_parser(default_character="walker", description=__doc__).parse_args()
    run("walker", args)
