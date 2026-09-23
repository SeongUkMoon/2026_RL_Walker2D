"""
캐릭터 환경(Environment)

Gymnasium 의 Walker2d-v5 환경은 xml_file 인자로 임의의 MJCF 를 받을 수 있습니다.
여기서는 그 위에 CharacterEnv(gym.Wrapper) 를 한 겹 씌워서
  1) reward 함수를 다루기 쉬운 State 객체로 계산할 수 있게 하고     (my_reward.compute_reward)
  2) termination(에피소드 종료) 조건을 바꿀 수 있게 하고           (my_reward.is_terminated)
  3) observation 에 값을 추가할 수 있게 합니다 (선택, 고급)         (my_reward.extra_observation)

참가자는 보통 my_reward.py 만 수정하면 되고, 이 파일은 읽어보기만 하면 됩니다.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import gymnasium as gym
import mujoco
import numpy as np

from .spec2mjcf import resolve_character
from .utils import ROOT, setup_gl_backend

# 프로젝트 루트를 import 경로에 넣어 my_reward 같은 모듈을 어디서 실행해도 찾을 수 있게 함
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------------------
# State: reward 함수에서 사용하는 "지금 캐릭터 상태" 요약
# --------------------------------------------------------------------------------------
@dataclass
class State:
    """
    한 스텝의 캐릭터 상태. reward 함수(my_reward.py)에서 state.x_vel 처럼 사용합니다.

    위치/속도 (몸통 = torso, root segment 기준)
      time          : 에피소드 시작 후 흐른 시간 (s)
      step          : 에피소드 안에서 몇 번째 action 인지
      x             : 몸통 x 위치 (m). 앞(오른쪽)으로 가면 증가
      x_vel         : 전진 속도 (m/s). 앞으로 가면 +, 뒤로 가면 -
      height        : 몸통 높이 (m)
      height_vel    : 높이 변화 속도 (m/s). 위로 뛰면 +
      height_init   : 에피소드 처음(서 있을 때) 몸통 높이 (m)
      angle         : 몸통 기울기 (rad). 0 = 처음 자세. +/- 방향은 캐릭터마다 다르니 abs() 로 쓰는 것을 추천
      angle_vel     : 몸통 회전 속도 (rad/s)
    관절
      joint_angles  : 각 joint 의 각도 (rad) 배열. 순서는 joint_names
      joint_vels    : 각 joint 의 각속도 (rad/s) 배열
      joint_names   : joint 이름 순서 (예: "thigh_r_joint")
    행동/접촉
      action        : 이번 스텝의 action 배열 (-1 ~ 1). 순서는 motor_joints
      contacts      : 지금 바닥(또는 장애물)에 닿아 있는 segment 이름들 (예: ("foot_l", "foot_r"))
      segment_names : 모든 segment 이름
    """

    time: float
    step: int
    x: float
    x_vel: float
    height: float
    height_vel: float
    height_init: float
    angle: float
    angle_vel: float
    joint_angles: np.ndarray
    joint_vels: np.ndarray
    joint_names: tuple
    action: np.ndarray
    motor_joints: tuple
    contacts: tuple
    segment_names: tuple

    # ---- 편의 속성 ----
    @property
    def height_ratio(self) -> float:
        """처음 높이에 대한 비율. 1.0 = 처음 높이, 0.5 = 절반으로 낮아짐"""
        return self.height / max(self.height_init, 1e-6)

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle)

    @property
    def energy(self) -> float:
        """action 크기의 제곱합 (힘을 얼마나 많이 쓰는지)"""
        return float(np.sum(np.square(self.action)))

    @property
    def n_contacts(self) -> int:
        return len(self.contacts)

    def joint(self, name: str) -> float:
        """이름으로 joint 각도(rad) 얻기. 'thigh_r' 또는 'thigh_r_joint' 모두 가능"""
        key = name if name.endswith("_joint") else name + "_joint"
        return float(self.joint_angles[self.joint_names.index(key)])

    def joint_vel(self, name: str) -> float:
        key = name if name.endswith("_joint") else name + "_joint"
        return float(self.joint_vels[self.joint_names.index(key)])

    def touching(self, *names: str) -> bool:
        """주어진 segment 중 하나라도 바닥에 닿아 있으면 True"""
        return any(n in self.contacts for n in names)

    def as_dict(self) -> dict:
        return {
            "time": round(self.time, 2),
            "x": round(self.x, 2),
            "x_vel": round(self.x_vel, 2),
            "height": round(self.height, 2),
            "height_ratio": round(self.height_ratio, 2),
            "angle_deg": round(self.angle_deg, 1),
            "contacts": list(self.contacts),
            "energy": round(self.energy, 2),
        }


# --------------------------------------------------------------------------------------
# reward 모듈 불러오기
# --------------------------------------------------------------------------------------
def load_reward_module(name_or_path: str | ModuleType | None) -> ModuleType | None:
    """'my_reward' 같은 모듈 이름 또는 .py 경로를 받아 import 합니다. None 이면 None."""
    if name_or_path is None or isinstance(name_or_path, ModuleType):
        return name_or_path
    p = Path(name_or_path)
    if p.suffix == ".py":
        if not p.exists():
            raise FileNotFoundError(f"reward 파일을 찾을 수 없습니다: {p}")
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    name = str(name_or_path)
    if name.endswith(".py"):
        name = name[:-3]
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise FileNotFoundError(f'reward 모듈 "{name}" 을(를) 찾을 수 없습니다. 프로젝트 폴더에 {name}.py 가 있어야 합니다. ({e})')


def reward_module_name(mod: ModuleType | None) -> str:
    if mod is None:
        return "default"
    return Path(getattr(mod, "__file__", mod.__name__)).stem


# --------------------------------------------------------------------------------------
# CharacterEnv
# --------------------------------------------------------------------------------------
class CharacterEnv(gym.Wrapper):
    """Walker2d-v5(+custom xml) 위에 State 계산과 reward/termination hook 을 얹은 Wrapper."""

    def __init__(self, env: gym.Env, meta: dict, reward_module: ModuleType | None = None):
        super().__init__(env)
        self.meta = meta
        self.reward_module = reward_module
        self.model: mujoco.MjModel = env.unwrapped.model
        self.data: mujoco.MjData = env.unwrapped.data
        self.dt: float = env.unwrapped.dt

        self.joint_names = tuple(meta["joint_names"])
        self.motor_joints = tuple(meta["motor_joints"])
        self.segment_names = tuple(meta["segments"])

        # geom id -> segment(body) 이름, 바닥/장애물 geom (world body 에 붙은 geom) 찾기
        self._geom_body_name = {}
        self._ground_geoms = set()
        for g in range(self.model.ngeom):
            b = int(self.model.geom_bodyid[g])
            if b == 0:
                self._ground_geoms.add(g)
            else:
                self._geom_body_name[g] = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)

        self._step = 0
        self._height_init = float(meta["torso_z0"])
        self._n_extra = 0

        # extra_observation 이 있으면 길이를 알아내기 위해 한 번 reset 해 본다
        obs, _ = self.env.reset()
        st = self._make_state(np.zeros(len(self.motor_joints), dtype=np.float32))
        extra = self._extra_obs(st)
        self._n_extra = len(extra)
        n = int(np.prod(self.env.observation_space.shape)) + self._n_extra
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float64)

    # ---- 내부 ----
    def _contacts(self) -> tuple:
        names = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self._ground_geoms and g2 in self._geom_body_name:
                names.add(self._geom_body_name[g2])
            elif g2 in self._ground_geoms and g1 in self._geom_body_name:
                names.add(self._geom_body_name[g1])
        return tuple(sorted(names))

    def _make_state(self, action: np.ndarray) -> State:
        q, v = self.data.qpos, self.data.qvel
        return State(
            time=float(self.data.time),
            step=self._step,
            x=float(q[0]),
            x_vel=float(v[0]),
            height=float(q[1]),
            height_vel=float(v[1]),
            height_init=self._height_init,
            angle=float(q[2]),
            angle_vel=float(v[2]),
            joint_angles=np.array(q[3:], dtype=np.float64),
            joint_vels=np.array(v[3:], dtype=np.float64),
            joint_names=self.joint_names,
            action=np.asarray(action, dtype=np.float64),
            motor_joints=self.motor_joints,
            contacts=self._contacts(),
            segment_names=self.segment_names,
        )

    def _extra_obs(self, state: State) -> np.ndarray:
        fn = getattr(self.reward_module, "extra_observation", None)
        if fn is None:
            return np.zeros(0)
        extra = np.asarray(fn(state), dtype=np.float64).ravel()
        return np.nan_to_num(extra)

    def _augment(self, obs: np.ndarray, state: State) -> np.ndarray:
        if self._n_extra == 0:
            return obs
        return np.concatenate([obs, self._extra_obs(state)])

    # ---- gym API ----
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step = 0
        self._height_init = float(self.data.qpos[1])
        state = self._make_state(np.zeros(len(self.motor_joints)))
        info["state"] = state
        return self._augment(obs, state), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        state = self._make_state(action)
        reward = float(base_reward)
        if self.reward_module is not None:
            fn = getattr(self.reward_module, "compute_reward", None)
            if fn is not None:
                reward = fn(state, float(base_reward), info)
                if not isinstance(reward, (int, float, np.floating)) or not math.isfinite(float(reward)):
                    raise ValueError(
                        f"compute_reward 가 숫자가 아닌 값을 돌려줬습니다: {reward!r}\n"
                        "  my_reward.py 의 compute_reward 는 반드시 float 하나를 return 해야 합니다."
                    )
                reward = float(reward)
            fn_t = getattr(self.reward_module, "is_terminated", None)
            if fn_t is not None:
                terminated = bool(fn_t(state, bool(terminated)))
        info["state"] = state
        info["base_reward"] = float(base_reward)
        return self._augment(obs, state), reward, terminated, truncated, info


# --------------------------------------------------------------------------------------
# 환경 만들기
# --------------------------------------------------------------------------------------
def make_env(
    character: str = "walker",
    render_mode: str | None = None,
    reward_module: str | ModuleType | None = None,
    terrain: str = "flat",
    width: int = 1024,
    height: int = 576,
    frame_skip: int | None = None,
    max_episode_steps: int = 1000,
) -> CharacterEnv:
    """
    캐릭터 이름(characters/<name>.json 또는 walker) 으로 환경을 만듭니다.

    render_mode : None(학습용) / "human"(창 열기) / "rgb_array"(영상 저장용)
    reward_module : "my_reward" 같은 모듈 이름, .py 경로, 또는 None(Gymnasium 기본 reward)
    terrain : "flat" / "bumps" / "bumps_hard"
    """
    setup_gl_backend()
    xml_path, meta = resolve_character(character, terrain=terrain)
    cam = {
        "trackbodyid": 1,  # 몸통(torso) 을 따라가는 카메라
        "distance": float(meta["camera"]["distance"]),
        "lookat": np.array([0.0, 0.0, float(meta["camera"]["lookat_z"])]),
        "elevation": -12.0,
    }
    env = gym.make(
        "Walker2d-v5",
        xml_file=str(xml_path),
        render_mode=render_mode,
        frame_skip=int(frame_skip or meta.get("frame_skip", 4)),
        healthy_z_range=tuple(meta["healthy_z_range"]),
        healthy_angle_range=tuple(meta["healthy_angle_range"]),
        default_camera_config=cam,
        width=width,
        height=height,
        max_episode_steps=max_episode_steps,
    )
    mod = load_reward_module(reward_module)
    return CharacterEnv(env, meta, mod)


def sanity_check(env: CharacterEnv, n_steps: int = 50) -> None:
    """reward 함수가 에러 없이 도는지 무작위 action 으로 짧게 확인합니다 (학습 시작 전 빠른 점검)."""
    env.reset(seed=0)
    for _ in range(n_steps):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if not np.all(np.isfinite(obs)):
            raise ValueError("observation 에 NaN/inf 가 있습니다. extra_observation 을 확인하세요.")
        if term or trunc:
            env.reset()
