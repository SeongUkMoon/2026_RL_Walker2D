"""
2교시 - 강화학습(PPO)으로 캐릭터 컨트롤러 학습

my_reward.py 의 reward 를 최대로 만드는 controller(policy)를 학습합니다. 결과는 runs/<run이름>/ 에 저장되고,
학습이 끝나면 자동으로 10초짜리 결과 영상(final.mp4)을 만듭니다.

사용 예
  python 4_train.py --character dog --minutes 10              dog 를 10분 학습
  python 4_train.py --character my_character --minutes 15     내 캐릭터
  python 4_train.py --character walker --minutes 5 --from pretrained/walker
        -> 미리 학습된 walker 에서 이어서(fine-tuning) 5분. reward 를 바꾼 뒤 변화를 빨리 보고 싶을 때
  python 4_train.py --character dog --reward my_reward_v2     다른 reward 파일로 학습 (비교용)
  python 4_train.py --character dog --terrain bumps           장애물 지형에서 학습
  python 4_train.py --character dog --steps 500000            시간 대신 step 수로 예산 지정

학습 중 Ctrl+C 를 누르면 그 시점까지의 모델을 저장하고 끝냅니다.
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from walker_rl.character_env import load_reward_module, make_env, reward_module_name, sanity_check
from walker_rl.runner import describe_result, run_record
from walker_rl.spec2mjcf import SpecError, resolve_character
from walker_rl.training import (TimeBudgetCallback, create_model, default_n_envs, load_model_for_training,
                                load_normalizer, load_run, make_policy_fn, make_vec_env, save_run, snapshot_reward_file)
from walker_rl.utils import RUNS_DIR, TERRAINS, VIDEOS_DIR, banner, fmt_minutes, kv, timestamp, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--character", "-c", default="my_character", help="캐릭터 이름 (기본 my_character)")
    p.add_argument("--reward", "-r", default="my_reward", help="reward 모듈 이름 또는 .py 경로 (기본 my_reward)")
    p.add_argument("--minutes", "-m", type=float, default=15.0, help="학습 시간 예산(분). 기본 15")
    p.add_argument("--steps", type=int, default=None, help="시간 대신 step 수 예산 (둘 다 주면 먼저 도달하는 쪽)")
    p.add_argument("--terrain", choices=TERRAINS, default="flat")
    p.add_argument("--from", dest="from_dir", default=None,
                   help="이어서 학습할 run 폴더 (예: pretrained/walker, runs/dog__my_reward__...)")
    p.add_argument("--n-envs", type=int, default=default_n_envs(), help="병렬 환경 수 (기본 8)")
    p.add_argument("--subproc", action="store_true", help="환경을 별도 프로세스로 (코어가 많은 PC 에서 더 빠를 수 있음)")
    p.add_argument("--name", default=None, help="run 폴더 이름에 붙일 메모 (예: slow_walk)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-video", action="store_true", help="학습 후 결과 영상 만들지 않기")
    args = p.parse_args()

    banner(f"학습 시작: character = {args.character} | reward = {args.reward} | terrain = {args.terrain}")

    # 1) 캐릭터 / reward 확인 및 빠른 점검 --------------------------------------------------
    try:
        xml_path, meta = resolve_character(args.character, terrain=args.terrain)
    except SpecError as e:
        print("[실패] " + str(e))
        sys.exit(1)
    try:
        reward_mod = load_reward_module(args.reward)
    except FileNotFoundError as e:
        print("[실패] " + str(e))
        sys.exit(1)

    print("  reward 함수 점검 중...", end=" ", flush=True)
    try:
        test_env = make_env(args.character, reward_module=reward_mod, terrain=args.terrain)
        sanity_check(test_env, n_steps=100)
        obs_dim, act_dim = test_env.observation_space.shape[0], test_env.action_space.shape[0]
        test_env.close()
    except Exception as e:
        print("\n[실패] reward 함수(또는 캐릭터)를 실행하는 중 오류가 났습니다:\n  " + repr(e))
        print("  힌트: 오류 메시지와 my_reward.py 내용을 AI 에게 함께 주고 고쳐 달라고 하세요.")
        sys.exit(1)
    print("OK")
    kv("observation 차원 / action 차원", f"{obs_dim} / {act_dim}")
    kv("병렬 환경 수", args.n_envs)
    kv("시간 예산", fmt_minutes(args.minutes * 60) if args.steps is None else f"{args.steps:,d} steps (최대 {fmt_minutes(args.minutes * 60)})")
    kv("CPU 코어", os.cpu_count())

    # 2) run 폴더 -------------------------------------------------------------------------
    rname = reward_module_name(reward_mod)
    run_name = f"{Path(args.character).stem}__{rname}"
    if args.terrain != "flat":
        run_name += f"__{args.terrain}"
    if args.name:
        run_name += f"__{args.name}"
    run_name += f"__{timestamp()}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_reward_file(reward_mod, run_dir)
    kv("저장 폴더", f"runs/{run_name}")

    # 3) 환경 + 모델 ----------------------------------------------------------------------
    start_steps = 0
    if args.from_dir:
        _, prev_stats, prev_cfg = load_run(args.from_dir)
        if prev_cfg.get("character") not in (Path(args.character).stem, None):
            print(f"  [주의] 이어서 학습할 모델은 '{prev_cfg.get('character')}' 로 학습된 것입니다. 캐릭터가 다르면 오류가 납니다.")
        venv = make_vec_env(args.character, args.reward, args.terrain, n_envs=args.n_envs,
                            subproc=args.subproc, seed=args.seed, normalize_stats=prev_stats)
        model = load_model_for_training(args.from_dir, venv)
        start_steps = int(model.num_timesteps)
        kv("이어서 학습", f"{args.from_dir} (지금까지 {start_steps:,d} steps)")
    else:
        venv = make_vec_env(args.character, args.reward, args.terrain, n_envs=args.n_envs,
                            subproc=args.subproc, seed=args.seed)
        model = create_model(venv, seed=args.seed)

    config = {
        "character": Path(args.character).stem,
        "character_ref": args.character if Path(args.character).suffix else Path(args.character).stem,
        "reward": rname,
        "terrain": args.terrain,
        "from": args.from_dir,
        "minutes": args.minutes,
        "steps_budget": args.steps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_dir / "config.json", config)

    # 4) 학습 -----------------------------------------------------------------------------
    print_every = 30.0 if args.minutes >= 5 else 10.0   # 짧은 학습은 더 자주 기록
    cb = TimeBudgetCallback(run_dir, budget_minutes=args.minutes, max_steps=args.steps, start_steps=start_steps,
                            print_every=print_every)
    t0 = time.time()
    try:
        model.learn(total_timesteps=10 ** 10, callback=cb, reset_num_timesteps=not bool(args.from_dir))
    except KeyboardInterrupt:
        print("\n  Ctrl+C - 학습을 중단하고 지금까지의 모델을 저장합니다.")
    elapsed = time.time() - t0

    # 5) 저장 -----------------------------------------------------------------------------
    save_run(model, venv, run_dir, config)
    mean_r, mean_l, mean_d = cb._stats()
    result = {
        "elapsed_sec": round(elapsed, 1),
        "timesteps_total": int(model.num_timesteps),
        "timesteps_this_run": int(model.num_timesteps) - start_steps,
        "mean_reward_last": None if mean_r is None else round(mean_r, 1),
        "mean_ep_len_last": None if mean_l is None else round(mean_l, 1),
        "mean_distance_last": None if mean_d is None else round(mean_d, 2),
    }
    venv.close()

    banner("학습 종료")
    kv("걸린 시간", fmt_minutes(elapsed))
    kv("학습 step 수", f"{result['timesteps_this_run']:,d} (누적 {result['timesteps_total']:,d})")
    if mean_r is not None:
        kv("마지막 평균 reward / 에피소드 길이 / 전진거리", f"{mean_r:.1f} / {mean_l:.0f} steps / {mean_d:.2f} m")
    kv("모델 저장", f"runs/{run_name}/model.zip")

    # 6) 결과 영상 -------------------------------------------------------------------------
    if not args.no_video:
        print("\n  결과 영상 만드는 중 (10초)...")
        env = make_env(args.character, render_mode="rgb_array", reward_module=reward_mod, terrain=args.terrain)
        policy = make_policy_fn(model, load_normalizer(run_dir / "vecnormalize.pkl"))
        res = run_record(env, policy, seconds=10.0, seed=1, out_mp4=run_dir / "final.mp4", out_png=run_dir / "final.png")
        result.update({"eval_" + k: v for k, v in res.items()})
        VIDEOS_DIR.mkdir(exist_ok=True)
        shutil.copy(run_dir / "final.mp4", VIDEOS_DIR / f"{run_name}.mp4")
        print("  " + describe_result(res))
        kv("영상", f"videos/{run_name}.mp4")
    write_json(run_dir / "result.json", result)

    print(f"\n다음 단계:  python 5_watch.py --run runs/{run_name}         (창으로 보기)")
    print(f"            python 6_plot.py --character {Path(args.character).stem}    (학습 곡선 비교)")
    print(f"            my_reward.py 를 수정하고 다시  python 4_train.py --character {Path(args.character).stem} --minutes {int(args.minutes)}")


if __name__ == "__main__":
    main()
