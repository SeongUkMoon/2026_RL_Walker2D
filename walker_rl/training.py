"""
PPO 학습 도우미 (Stable-Baselines3)

  - make_vec_env()      : 병렬 환경 + VecMonitor + VecNormalize 만들기
  - create_model()      : PPO 모델 만들기 (CPU)
  - TimeBudgetCallback  : 시간 예산(분) 안에서 학습, 진행 상황 한글 출력, 체크포인트 저장
  - save_run()/load_run : runs/<run>/ 폴더에 model.zip, vecnormalize.pkl, config.json 저장/불러오기
  - load_normalizer()   : 학습 때 쓴 observation 정규화 통계 불러오기 (재생할 때 필요)

관측/보상 정규화(VecNormalize)를 켜는 것이 같은 시간 안에 걷기까지 가느냐를 좌우합니다.
(2코어 CPU 기준: 정규화 없이 6분 -> 걷기 시작 단계, 정규화 켜고 5분 -> 안정적으로 걷기)
"""
from __future__ import annotations

import csv
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from .character_env import make_env
from .utils import fmt_minutes, read_json, write_json

# PPO 하이퍼파라미터 (CPU 에서 빠르게 배우도록 맞춘 값. 바꾸지 않아도 됩니다)
PPO_KWARGS = dict(
    n_steps=512,           # 환경 하나당 rollout 길이 (8개 env 면 한 번에 4096 step 수집)
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
)


def default_n_envs() -> int:
    return 8


def make_vec_env(character: str, reward_module: str | None, terrain: str = "flat",
                 n_envs: int = 8, subproc: bool = False, seed: int = 0,
                 normalize_stats: str | Path | None = None, training: bool = True,
                 reset_reward_stats: bool = False):
    """
    학습용 벡터 환경을 만듭니다.
    normalize_stats 를 주면 저장된 VecNormalize 관측 통계를 이어서 사용합니다 (이어 학습 / fine-tuning).
    reward 함수를 바꿔 fine-tuning 할 때는 reset_reward_stats=True 로 이전 return 통계를 버릴 수 있습니다.
    """
    def _factory(rank: int):
        def _init():
            env = make_env(character, render_mode=None, reward_module=reward_module, terrain=terrain)
            env.reset(seed=seed + rank)
            return env
        return _init

    fns = [_factory(i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if subproc else DummyVecEnv(fns)
    venv = VecMonitor(venv, info_keywords=("x_position",))
    if normalize_stats is not None:
        venv = VecNormalize.load(str(normalize_stats), venv)
        venv.training = training
        venv.norm_reward = True
        if reset_reward_stats:
            venv.ret_rms = RunningMeanStd(shape=())
            venv.returns = np.zeros(venv.num_envs, dtype=np.float64)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=PPO_KWARGS["gamma"])
    return venv


def create_model(venv, seed: int = 0, verbose: int = 0, **overrides) -> PPO:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    kwargs = {**PPO_KWARGS, **overrides}
    return PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=verbose, **kwargs)


class TimeBudgetCallback(BaseCallback):
    """
    - budget_minutes 가 지나거나 max_steps 에 도달하면 학습을 멈춥니다.
    - print_every 초마다 진행 상황을 한글로 출력하고 progress.csv 에 기록합니다.
    - ckpt_every_minutes 마다 checkpoints/ 에 중간 저장합니다.
    """

    def __init__(self, run_dir: Path, budget_minutes: float | None, max_steps: int | None = None,
                 print_every: float = 30.0, ckpt_every_minutes: float = 3.0, start_steps: int = 0):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.budget = None if budget_minutes is None else budget_minutes * 60.0
        self.max_steps = max_steps
        self.print_every = print_every
        self.ckpt_every = ckpt_every_minutes * 60.0
        self.start_steps = start_steps
        self.t0 = None
        self.next_print = print_every
        self.next_ckpt = self.ckpt_every
        self.rows: list[dict] = []
        self.best_mean = -np.inf

    def _on_training_start(self) -> None:
        self.t0 = time.time()
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self._csv = open(self.run_dir / "progress.csv", "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv)
        if self._csv.tell() == 0:
            self._writer.writerow(["elapsed_sec", "timesteps", "steps_per_sec", "mean_reward", "mean_ep_len", "mean_distance"])
        print(f"  학습 시작. 진행 상황은 {int(self.print_every)}초마다 출력됩니다. (Ctrl+C 로 중단하면 지금까지의 결과를 저장합니다)")

    def _stats(self):
        buf = self.model.ep_info_buffer
        if not buf:
            return None, None, None
        rews = [e["r"] for e in buf]
        lens = [e["l"] for e in buf]
        dists = [e.get("x_position", np.nan) for e in buf]
        return float(np.mean(rews)), float(np.mean(lens)), float(np.nanmean(dists)) if len(dists) else float("nan")

    def _on_step(self) -> bool:
        elapsed = time.time() - self.t0
        steps = self.num_timesteps - self.start_steps
        if elapsed >= self.next_print:
            self.next_print += self.print_every
            mean_r, mean_l, mean_d = self._stats()
            sps = steps / max(elapsed, 1e-6)
            if mean_r is None:
                print(f"  [{fmt_minutes(elapsed)}] {steps:>9,d} steps  ({sps:,.0f} steps/s)  아직 끝난 에피소드가 없습니다")
            else:
                dt = self.training_env.get_attr("dt")[0] if hasattr(self.training_env, "get_attr") else 0.008
                print(f"  [{fmt_minutes(elapsed)}] {steps:>9,d} steps ({sps:,.0f} steps/s) | "
                      f"평균 reward {mean_r:8.1f} | 평균 에피소드 {mean_l:5.0f} steps ({mean_l * dt:4.1f}s) | 평균 전진 {mean_d:5.2f} m")
                self._write_row(elapsed)
        if elapsed >= self.next_ckpt:
            self.next_ckpt += self.ckpt_every
            tag = f"{self.num_timesteps:09d}"
            self.model.save(self.run_dir / "checkpoints" / f"ckpt_{tag}.zip")
            self.training_env.save(str(self.run_dir / "checkpoints" / f"vecnormalize_{tag}.pkl"))
        if self.budget is not None and elapsed >= self.budget:
            print(f"  시간 예산({fmt_minutes(self.budget)})이 끝나 학습을 마칩니다.")
            return False
        if self.max_steps is not None and steps >= self.max_steps:
            print(f"  목표 step 수({self.max_steps:,d})에 도달해 학습을 마칩니다.")
            return False
        return True

    def _write_row(self, elapsed: float) -> None:
        mean_r, mean_l, mean_d = self._stats()
        if mean_r is None:
            return
        steps = self.num_timesteps - self.start_steps
        self._writer.writerow([round(elapsed, 1), self.num_timesteps, round(steps / max(elapsed, 1e-6)),
                               round(mean_r, 2), round(mean_l, 1), round(mean_d, 3)])
        self._csv.flush()

    def _on_training_end(self) -> None:
        # 마지막 상태를 한 줄 더 기록 (짧은 학습이라도 그래프에 점이 하나는 생기게)
        try:
            self._write_row(time.time() - self.t0)
            self._csv.close()
        except Exception:
            pass


def save_run(model: PPO, venv: VecNormalize, run_dir: Path, config: dict) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model.zip")
    venv.save(str(run_dir / "vecnormalize.pkl"))
    write_json(run_dir / "config.json", config)


def load_run(run_dir: str | Path) -> tuple[PPO, Path, dict]:
    """runs/<run>/ 또는 pretrained/<name>/ 폴더에서 (model, vecnormalize.pkl 경로, config) 를 불러옵니다."""
    run_dir = Path(run_dir)
    model_path = run_dir / "model.zip"
    stats_path = run_dir / "vecnormalize.pkl"
    cfg_path = run_dir / "config.json"
    for p in (model_path, stats_path, cfg_path):
        if not p.exists():
            raise FileNotFoundError(f"{run_dir} 안에 {p.name} 이 없습니다. 학습이 끝난 run 폴더(또는 pretrained/<이름>)를 지정하세요.")
    model = PPO.load(str(model_path), device="cpu")
    return model, stats_path, read_json(cfg_path)


def load_model_for_training(run_dir: str | Path, venv) -> PPO:
    """저장된 모델을 새 벡터 환경에 붙여서 이어 학습할 수 있게 불러옵니다 (n_envs 가 달라도 됨)."""
    model_path = Path(run_dir) / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"{run_dir} 안에 model.zip 이 없습니다.")
    return PPO.load(str(model_path), env=venv, device="cpu")


def load_normalizer(stats_path: str | Path):
    """저장된 VecNormalize 통계로 observation 을 정규화하는 함수를 돌려줍니다."""
    with open(stats_path, "rb") as f:
        vn = pickle.load(f)
    vn.training = False

    def normalize(obs: np.ndarray) -> np.ndarray:
        return vn.normalize_obs(np.asarray(obs, dtype=np.float64))
    return normalize


def make_policy_fn(model: PPO, normalize, deterministic: bool = True):
    """(정규화 -> 모델 예측) 을 묶은 policy 함수."""
    def fn(obs):
        action, _ = model.predict(normalize(obs), deterministic=deterministic)
        return action
    return fn


def snapshot_reward_file(reward_module, run_dir: Path) -> None:
    """학습에 사용한 reward 파일을 run 폴더에 복사해 둡니다 (나중에 비교용)."""
    src = getattr(reward_module, "__file__", None)
    if src and Path(src).exists():
        shutil.copy(src, Path(run_dir) / "reward_used.py")
