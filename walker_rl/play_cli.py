"""1_run_walker.py 와 3_play.py 가 공유하는 실행 로직 (캐릭터를 무작위/사인파/무동작으로 움직여 보기)."""
from __future__ import annotations

import argparse

from .character_env import make_env
from .runner import describe_result, random_policy, run_live, run_record, wiggle_policy, zero_policy
from .utils import VIDEOS_DIR, TERRAINS, banner, kv

MODES = {
    "random": "무작위 action (강화학습 전의 '아무것도 모르는' 컨트롤러)",
    "wiggle": "각 모터를 서로 다른 박자의 사인파로 흔들기 (어느 joint 가 어떻게 움직이는지 보기)",
    "zero": "힘을 주지 않음 (중력에만 맡김)",
}


def build_parser(default_character: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    if default_character is None:
        p.add_argument("character", nargs="?", default="my_character",
                       help="캐릭터 이름 (characters/<이름>.json) 또는 json 경로. 기본: my_character")
    p.add_argument("--mode", choices=list(MODES), default="random",
                   help="\n".join(f"{k:7s}: {v}" for k, v in MODES.items()))
    p.add_argument("--strength", type=float, default=0.5, help="action 세기 0~1 (기본 0.5)")
    p.add_argument("--seconds", type=float, default=20.0, help="실행 시간(시뮬레이션 초). 기본 20")
    p.add_argument("--terrain", choices=TERRAINS, default="flat", help="지형: flat(평지) / bumps / bumps_hard")
    p.add_argument("--record", action="store_true", help="창 대신 videos/ 폴더에 MP4 + PNG 로 저장")
    p.add_argument("--seed", type=int, default=None)
    return p


def make_policy(env, mode: str, strength: float):
    if mode == "random":
        return random_policy(env, strength)
    if mode == "wiggle":
        return wiggle_policy(env, strength)
    return zero_policy(env)


def run(character: str, args) -> None:
    banner(f"캐릭터 실행: {character}  |  mode = {args.mode}  |  terrain = {args.terrain}")
    render_mode = "rgb_array" if args.record else "human"
    env = make_env(character, render_mode=render_mode, terrain=args.terrain)
    meta = env.meta
    kv("segment(마디) 수", len(meta["segments"]))
    kv("motor(모터) 수 = action 차원", meta["n_motors"])
    kv("observation 차원", env.observation_space.shape[0])
    kv("몸통 처음 높이", f'{meta["torso_z0"]:.2f} m')
    kv("mode", MODES[args.mode])
    print()
    policy = make_policy(env, args.mode, args.strength)
    if args.record:
        name = str(character).replace("/", "_").replace("\\", "_").replace(".json", "")
        stem = f"{name}_{args.mode}" + (f"_{args.terrain}" if args.terrain != "flat" else "")
        res = run_record(env, policy, seconds=args.seconds, seed=args.seed,
                         out_mp4=VIDEOS_DIR / f"{stem}.mp4", out_png=VIDEOS_DIR / f"{stem}.png")
        print("  " + describe_result(res))
    else:
        print("  창이 열립니다. Space = 일시정지 / Tab = 카메라 전환 / 마우스 드래그 = 시점 회전 / 휠 = 확대")
        print("  넘어지면 자동으로 처음 자세로 돌아갑니다. 끝내려면 창을 닫거나 Ctrl+C.\n")
        res = run_live(env, policy, seconds=args.seconds, seed=args.seed)
        print(f"\n  총 {res['episodes']}개 에피소드, 넘어진 횟수 {res['falls']}, 최대 전진 거리 {res['max_x']:.2f} m")
