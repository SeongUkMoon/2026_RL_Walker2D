"""
1교시 - 내 캐릭터 만들기: JSON spec -> MuJoCo XML

종이에 그린 캐릭터를 AI 에게 보여주고 (prompts/01_sketch_to_json.md 참고) 받은 JSON 을
characters/my_character.json 에 저장한 뒤 이 스크립트를 실행합니다.

하는 일
  1) JSON 검사 (문제가 있으면 한글로 알려 줍니다)
  2) characters/<이름>.xml + <이름>.meta.json 생성
  3) 낙하 테스트: 힘을 주지 않고 3초 동안 떨어뜨려서 제대로 서는지 확인
     -> videos/<이름>_drop.mp4 / _drop.png 저장 (--show 를 붙이면 창으로 바로 보기)

사용 예
  python 2_build_character.py                              characters/my_character.json
  python 2_build_character.py characters/my_character.json
  python 2_build_character.py dog --show                   예제 dog 를 빌드하고 창으로 보기
  python 2_build_character.py my_character --no-video      영상 저장 생략 (빠름)
"""
import argparse
import sys
from pathlib import Path

from walker_rl.character_env import make_env
from walker_rl.runner import run_live, run_record, zero_policy
from walker_rl.spec2mjcf import SpecError, build_character, load_spec, validate_spec
from walker_rl.utils import CHARACTERS_DIR, VIDEOS_DIR, banner, kv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("spec", nargs="?", default="my_character", help="캐릭터 이름 또는 json 경로 (기본 my_character)")
    p.add_argument("--show", action="store_true", help="낙하 테스트를 창으로 보기")
    p.add_argument("--seconds", type=float, default=3.0, help="낙하 테스트 시간(초). 기본 3")
    p.add_argument("--no-video", action="store_true", help="낙하 테스트 영상 저장 생략")
    args = p.parse_args()

    spec_path = Path(args.spec)
    if spec_path.suffix != ".json":
        spec_path = CHARACTERS_DIR / f"{spec_path.stem}.json"
    spec_path = spec_path.resolve()
    # characters/ 폴더 안의 파일이면 이름으로, 밖의 파일이면 경로로 부릅니다
    ref = spec_path.stem if spec_path.parent == CHARACTERS_DIR.resolve() else str(spec_path)

    banner(f"캐릭터 빌드: {spec_path.name}")
    try:
        spec = load_spec(spec_path)
        warnings = validate_spec(spec)
        xml_path, meta = build_character(spec_path)
    except SpecError as e:
        print("\n[실패] " + str(e))
        print("\n힌트: 이 메시지와 JSON 파일 내용을 AI 에게 함께 주고 '고쳐 달라'고 하면 대부분 해결됩니다.")
        sys.exit(1)

    for w in warnings:
        print(f"  [경고] {w}")
    print()
    kv("이름", meta["name"])
    kv("segment(마디)", ", ".join(meta["segments"]))
    kv("motor 가 달린 joint", ", ".join(j.replace("_joint", "") for j in meta["motor_joints"]) or "(없음)")
    kv("총 질량", f'{meta["total_mass"]:.1f} kg')
    kv("전체 높이 / 몸통 높이", f'{meta["height"]:.2f} m / {meta["torso_z0"]:.2f} m')
    kv("모터 힘(gear)", meta["gear"])
    kv("넘어짐 판정", f'몸통 높이 < {meta["healthy_z_range"][0]:.2f} m  또는  기울기 > {abs(meta["healthy_angle_range"][1]) * 57.3:.0f} deg')
    kv("생성 파일", f"{xml_path.name}, {xml_path.stem}.meta.json")

    # ---- 낙하 테스트 ----
    print("\n낙하 테스트 (힘 없이 떨어뜨리기)...")
    if args.show:
        env = make_env(ref, render_mode="human")
        kv("observation / action 차원", f"{env.observation_space.shape[0]} / {env.action_space.shape[0]}")
        run_live(env, zero_policy(env), seconds=max(args.seconds, 3.0), print_every=0.5)
    else:
        env = make_env(ref, render_mode="rgb_array")
        kv("observation / action 차원", f"{env.observation_space.shape[0]} / {env.action_space.shape[0]}")
        out_mp4 = None if args.no_video else VIDEOS_DIR / f"{spec_path.stem}_drop.mp4"
        res = run_record(env, zero_policy(env), seconds=args.seconds, stop_on_fall=True,
                         out_mp4=out_mp4, out_png=VIDEOS_DIR / f"{spec_path.stem}_drop.png", verbose=True)
        if res["falls"] > 0:
            print(f"  -> 힘을 주지 않으면 {res['episode_seconds']}초 만에 넘어집니다. (정상입니다. 학습으로 균형을 잡게 됩니다)")
            print("     다만 0.3초도 안 돼서 넘어지면 다리가 너무 짧거나 몸통이 너무 위에 있는지 그림을 확인해 보세요.")
        else:
            print(f"  -> {args.seconds}초 동안 넘어지지 않고 서 있습니다.")
    print(f"\n다음 단계:  python 3_play.py {ref}            (무작위로 움직여 보기)")
    print(f"            python 4_train.py --character {ref} --minutes 10   (2교시: 학습)")
    if ref != spec_path.stem:
        print("  (JSON 을 characters/ 폴더에 두면 이름만으로 부를 수 있습니다)")


if __name__ == "__main__":
    main()
