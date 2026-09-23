"""
2교시 - 학습된 컨트롤러 재생

사용 예
  python 5_watch.py --pretrained walker         미리 학습해 둔 walker 보기 (dog, ostrich 도 있음)
  python 5_watch.py --character dog             dog 로 가장 최근에 학습한 결과 보기
  python 5_watch.py --run runs/dog__my_reward__20260923-1530   특정 run 보기
  python 5_watch.py --character dog --record    창 대신 videos/ 에 MP4 저장
  python 5_watch.py --pretrained dog --terrain bumps   다른 지형에서 재생해 보기 (학습 때와 다르면 잘 못 걷습니다)
  python 5_watch.py --list                      저장된 run 목록 보기
"""
import argparse
import math
import sys
from pathlib import Path

from walker_rl.character_env import make_env
from walker_rl.runner import describe_result, run_live, run_record
from walker_rl.training import load_normalizer, load_run, make_policy_fn
from walker_rl.utils import (PRETRAINED_DIR, TERRAINS, VIDEOS_DIR, banner, kv, latest_run,
                             list_runs, read_json, resolve_run_artifact)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pretrained", "-p", default=None, help="pretrained/<이름> 의 모델 (walker / dog / ostrich)")
    g.add_argument("--character", "-c", default=None, help="이 캐릭터의 가장 최근 run")
    g.add_argument("--run", default=None, help="run 폴더 경로")
    g.add_argument("--list", action="store_true", help="저장된 run 목록 출력")
    p.add_argument("--seconds", type=float, default=15.0, help="재생 시간(초). 기본 15")
    p.add_argument("--terrain", choices=TERRAINS, default=None, help="지형 (기본: 학습 때와 같은 지형)")
    p.add_argument("--record", action="store_true", help="창 대신 videos/ 에 MP4 + PNG 저장")
    p.add_argument("--stochastic", action="store_true", help="action 에 무작위성 유지 (기본은 deterministic)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if not math.isfinite(args.seconds) or args.seconds <= 0:
        p.error("--seconds 는 유한한 0보다 큰 수여야 합니다.")

    if args.list:
        banner("저장된 run 목록 (오래된 순)")
        for r in list_runs():
            cfg = read_json(r / "config.json")
            res = read_json(r / "result.json") if (r / "result.json").exists() else {}
            print(f"  runs/{r.name}   reward={cfg.get('reward')}  terrain={cfg.get('terrain')}  "
                  f"평균reward={res.get('mean_reward_last')}  전진={res.get('eval_distance')} m")
        for r in sorted(PRETRAINED_DIR.glob("*/config.json")):
            print(f"  pretrained/{r.parent.name}")
        return

    if args.run:
        run_dir = Path(args.run)
    elif args.pretrained:
        run_dir = PRETRAINED_DIR / args.pretrained
    else:
        character = args.character or "my_character"
        run_dir = latest_run(character)
        if run_dir is None:
            print(f"[실패] '{character}' 로 학습한 run 이 runs/ 에 없습니다. 먼저  python 4_train.py --character {character}  를 실행하세요.")
            print("       미리 학습된 모델을 보려면:  python 5_watch.py --pretrained walker")
            sys.exit(1)

    try:
        model, stats_path, cfg = load_run(run_dir)
    except FileNotFoundError as e:
        print("[실패] " + str(e))
        sys.exit(1)

    character_ref = cfg.get("character_ref") or cfg["character"]
    trained_terrain = cfg.get("terrain", "flat")
    terrain = args.terrain or trained_terrain
    using_snapshot = False
    try:
        if terrain == trained_terrain:
            snapshot_xml = resolve_run_artifact(run_dir, cfg, "xml")
            snapshot_meta = resolve_run_artifact(run_dir, cfg, "meta")
            if bool(snapshot_xml) != bool(snapshot_meta):
                raise ValueError("캐릭터 XML/meta snapshot 중 하나만 기록되어 있습니다.")
            if snapshot_xml is not None:
                character = str(snapshot_xml)
                using_snapshot = True
            else:
                character = character_ref
        else:
            source_json = resolve_run_artifact(run_dir, cfg, "source_json")
            if source_json is not None:
                character = str(source_json)
            elif "character_artifacts" in cfg:
                raise ValueError(
                    "이 run에는 다른 지형을 다시 만들 원본 character JSON snapshot이 없습니다. "
                    f"학습 지형({trained_terrain})으로 재생하거나 원본 JSON 캐릭터로 다시 학습하세요."
                )
            else:
                character = character_ref
                print("  [주의] 예전 형식의 run이라 캐릭터 snapshot이 없습니다. 현재 characters/ 파일로 다른 지형을 만듭니다.")
    except (FileNotFoundError, ValueError) as e:
        print("[실패] " + str(e))
        sys.exit(1)
    reward = cfg.get("reward", "my_reward")
    # 학습 때 쓴 reward 파일이 run 폴더에 있으면 그것을 사용 (termination 조건을 같게 하기 위해)
    reward_used = run_dir / "reward_used.py"
    reward_arg = str(reward_used) if reward_used.exists() else None

    banner(f"재생: {run_dir}")
    kv("character / terrain / reward", f'{cfg["character"]} / {terrain} / {reward}')
    if using_snapshot:
        kv("캐릭터", "학습 당시 XML/meta snapshot (SHA-256 확인 완료)")
    res_file = run_dir / "result.json"
    if res_file.exists():
        res = read_json(res_file)
        total_steps = res.get("timesteps_total")
        total_steps_text = f"{total_steps:,}" if isinstance(total_steps, int) else "?"
        kv("학습 step 수 / 마지막 평균 reward", f"{total_steps_text} / {res.get('mean_reward_last')}")

    render_mode = "rgb_array" if args.record else "human"
    env = make_env(character, render_mode=render_mode, reward_module=reward_arg, terrain=terrain)
    env_obs_shape = getattr(env.observation_space, "shape", None)
    model_obs_shape = getattr(model.observation_space, "shape", None)
    env_action_shape = getattr(env.action_space, "shape", None)
    model_action_shape = getattr(model.action_space, "shape", None)
    if env_obs_shape != model_obs_shape or env_action_shape != model_action_shape:
        print("[실패] 환경과 모델의 공간 차원이 다릅니다.")
        print(f"       observation: 환경 {env_obs_shape}, 모델 {model_obs_shape}")
        print(f"       action:      환경 {env_action_shape}, 모델 {model_action_shape}")
        print("       캐릭터나 extra_observation 이 학습 이후에 바뀌었는지 확인하세요.")
        env.close()
        sys.exit(1)
    policy = make_policy_fn(model, load_normalizer(stats_path), deterministic=not args.stochastic)

    if args.record:
        stem = f"{run_dir.name}_watch" + (f"_{terrain}" if terrain != cfg.get("terrain", "flat") else "")
        res = run_record(env, policy, seconds=args.seconds, seed=args.seed,
                         out_mp4=VIDEOS_DIR / f"{stem}.mp4", out_png=VIDEOS_DIR / f"{stem}.png")
        print("  " + describe_result(res))
    else:
        print("  창이 열립니다. Space = 일시정지 / Tab = 카메라 전환 / 창 닫기 또는 Ctrl+C = 종료\n")
        r = run_live(env, policy, seconds=args.seconds, seed=args.seed)
        print(f"\n  {r['episodes']}개 에피소드, 넘어진 횟수 {r['falls']}, 최대 전진 거리 {r['max_x']:.2f} m")


if __name__ == "__main__":
    main()
