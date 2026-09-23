"""
1교시 - 내가 만든 캐릭터를 움직여 보기

2_build_character.py 로 만든 캐릭터(또는 dog / ostrich / biped 예제)를 무작위 action 등으로 움직여 봅니다.
콘솔에 1초마다 캐릭터 상태(x 위치, 속도, 높이, 기울기, 바닥에 닿은 마디)가 출력됩니다.
이 값들이 2교시 reward 함수에서 쓰는 state 입니다.

사용 예
  python 3_play.py                          characters/my_character.json 을 무작위 action 으로
  python 3_play.py dog                      예제 캐릭터 dog
  python 3_play.py dog --mode wiggle        모터를 사인파로 흔들기
  python 3_play.py ostrich --strength 1.0   action 세기 최대
  python 3_play.py dog --record             창 대신 videos/dog_random.mp4 + png 저장
  python 3_play.py dog --terrain bumps      장애물 지형에서
"""
from walker_rl.play_cli import build_parser, run

if __name__ == "__main__":
    args = build_parser(default_character=None, description=__doc__).parse_args()
    run(args.character, args)
