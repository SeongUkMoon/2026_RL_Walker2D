"""
에피소드 실행 도우미
  - run_live()   : 창(render_mode="human")을 띄워 실시간으로 보여주며 상태를 콘솔에 출력
  - run_record() : 화면을 캡처해 MP4 와 PNG(장면 모음) 로 저장
두 함수 모두 policy_fn(obs) -> action 을 받습니다. (무작위, 0, 학습된 모델 등)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

PolicyFn = Callable[[np.ndarray], np.ndarray]


def _close_env(env) -> None:
    """실행 경로와 오류 종류에 관계없이 환경 정리를 시도합니다."""
    try:
        env.close()
    except Exception:
        pass


def _state_line(info: dict, reward_sum: float) -> str:
    st = info.get("state")
    if st is None:
        return ""
    return (f"t={st.time:5.1f}s  x={st.x:6.2f}m  속도={st.x_vel:5.2f}m/s  높이={st.height:4.2f}m "
            f"({st.height_ratio * 100:3.0f}%)  기울기={st.angle_deg:6.1f}deg  바닥접촉={list(st.contacts)}  누적reward={reward_sum:7.1f}")


def run_live(env, policy_fn: PolicyFn, seconds: float = 20.0, print_every: float = 1.0, seed: int | None = None) -> dict:
    """
    창을 띄워 seconds(시뮬레이션 시간) 동안 실행합니다. 넘어지면 자동으로 reset.
    창에서 Space = 일시정지, Tab = 카메라 전환, 마우스 드래그/휠 = 시점 이동.
    """
    if not np.isfinite(seconds) or seconds <= 0:
        _close_env(env)
        raise ValueError(f"seconds는 0보다 큰 유한한 값이어야 합니다: {seconds!r}")
    if not np.isfinite(print_every) or print_every <= 0:
        _close_env(env)
        raise ValueError(f"print_every는 0보다 큰 유한한 값이어야 합니다: {print_every!r}")

    reward_sum, falls, episodes = 0.0, 0, 1
    x_start = max_x = 0.0
    t0 = time.time()
    try:
        obs, info = env.reset(seed=seed)
        dt = float(env.unwrapped.dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"환경의 dt는 0보다 큰 유한한 값이어야 합니다: {dt!r}")
        n_steps = max(1, int(np.ceil(seconds / dt)))
        print_every_steps = max(1, int(round(print_every / dt)))
        x_start = info["state"].x if "state" in info else 0.0
        max_x = x_start
        for i in range(n_steps):
            action = policy_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            reward_sum += r
            if "state" in info:
                max_x = max(max_x, info["state"].x)
            if i % print_every_steps == 0:
                print("  " + _state_line(info, reward_sum))
            if term or trunc:
                if term:
                    falls += 1
                    print(f"  >> 넘어짐(terminated) - 에피소드 {episodes} 종료, 누적 reward {reward_sum:.1f}. 다시 시작합니다.")
                else:
                    print(f"  >> 시간 제한(truncated) - 에피소드 {episodes} 종료, 누적 reward {reward_sum:.1f}. 다시 시작합니다.")
                episodes += 1
                reward_sum = 0.0
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\n  Ctrl+C 로 중단했습니다.")
    except Exception as e:  # 창을 닫은 경우 등
        msg = str(e)
        if "window" in msg.lower() or "glfw" in msg.lower() or "context" in msg.lower():
            print("\n  창이 닫혀서 종료합니다.")
        else:
            raise
    finally:
        _close_env(env)
    return {"episodes": episodes, "falls": falls, "max_x": max_x - x_start, "wall_time": time.time() - t0}


def run_record(
    env,
    policy_fn: PolicyFn,
    seconds: float = 10.0,
    out_mp4: str | Path | None = None,
    out_png: str | Path | None = None,
    fps: int = 30,
    seed: int | None = None,
    stop_on_fall: bool = False,
    verbose: bool = True,
) -> dict:
    """
    환경을 seconds(시뮬레이션 시간) 동안 실행하며 선택적으로 화면을 저장합니다.

    MP4 프레임은 파일에 바로 기록하고, PNG 장면 모음용 프레임만 최대 6개
    보관합니다. 따라서 긴 영상을 만들 때도 모든 원본 프레임을 메모리에 쌓지
    않습니다. out_mp4 와 out_png 가 모두 None 이면 render()와 imageio를 전혀
    사용하지 않으므로 render_mode=None 환경에서 빠르게 검사할 수 있습니다.

    반환: {"distance": 첫 에피소드 전진 거리(m), "falls": 넘어진 횟수, "reward": 첫 에피소드 누적 reward, ...}
    """
    if not np.isfinite(seconds) or seconds <= 0:
        _close_env(env)
        raise ValueError(f"seconds는 0보다 큰 유한한 값이어야 합니다: {seconds!r}")
    if not np.isfinite(fps) or fps <= 0:
        _close_env(env)
        raise ValueError(f"fps는 0보다 큰 유한한 값이어야 합니다: {fps!r}")

    out_mp4 = Path(out_mp4) if out_mp4 is not None else None
    out_png = Path(out_png) if out_png is not None else None
    should_render = out_mp4 is not None or out_png is not None
    imageio = None
    writer = None
    sheet_frames: list[np.ndarray] = []
    sheet_steps: set[int] = set()
    captured_sheet_steps: set[int] = set()
    completed_steps = 0
    reward_sum, falls, episodes = 0.0, 0, 1
    first_ep_reward = first_ep_dist = first_ep_len = None

    try:
        obs, info = env.reset(seed=seed)
        dt = float(env.unwrapped.dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"환경의 dt는 0보다 큰 유한한 값이어야 합니다: {dt!r}")
        n_steps = max(1, int(np.ceil(seconds / dt)))
        every = max(1, int(round(1.0 / (dt * float(fps)))))
        if out_png is not None:
            sheet_steps = set(
                int(i) for i in np.linspace(0, n_steps, num=min(6, n_steps + 1), dtype=int)
            )

        if should_render:
            import imageio.v2 as imageio

        if out_mp4 is not None:
            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(out_mp4), fps=fps, codec="libx264", quality=8,
                pixelformat="yuv420p", macro_block_size=16,
            )
        if out_png is not None:
            out_png.parent.mkdir(parents=True, exist_ok=True)

        def capture(step: int, *, final: bool = False) -> None:
            """필요한 한 프레임만 렌더링해 writer/장면 모음으로 보냅니다."""
            video_due = writer is not None and (final or step % every == 0)
            sheet_due = out_png is not None and (step in sheet_steps or final)
            if not video_due and not sheet_due:
                return
            frame = env.render()
            if frame is None:
                raise RuntimeError(
                    "env.render()가 프레임을 반환하지 않았습니다. "
                    "영상을 저장할 때는 render_mode='rgb_array' 환경이 필요합니다."
                )
            frame = np.asarray(frame)
            if video_due:
                writer.append_data(frame)
            if sheet_due and step not in captured_sheet_steps:
                # 장면 모음만 절반 크기 복사본으로 보관: 최대 6개 + 조기 종료 시 마지막 1개
                if len(sheet_frames) >= 6:
                    sheet_frames[-1] = frame[::2, ::2].copy()
                else:
                    sheet_frames.append(frame[::2, ::2].copy())
                captured_sheet_steps.add(step)

        x_start = info["state"].x if "state" in info else float(env.unwrapped.data.qpos[0])
        steps_in_ep = 0
        for i in range(n_steps):
            capture(i)
            action = policy_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            completed_steps = i + 1
            reward_sum += r
            steps_in_ep += 1
            if term or trunc:
                if first_ep_reward is None:
                    first_ep_reward = reward_sum
                    first_ep_dist = float(env.unwrapped.data.qpos[0]) - x_start
                    first_ep_len = steps_in_ep
                if term:
                    falls += 1
                if stop_on_fall:
                    break
                episodes += 1
                reward_sum, steps_in_ep = 0.0, 0
                obs, info = env.reset()
                x_start = float(env.unwrapped.data.qpos[0])

        if first_ep_reward is None:  # 한 번도 안 넘어짐
            first_ep_reward = reward_sum
            first_ep_dist = float(env.unwrapped.data.qpos[0]) - x_start
            first_ep_len = steps_in_ep

        capture(completed_steps, final=True)

        if out_png is not None:
            if not sheet_frames:
                raise RuntimeError("장면 모음에 사용할 렌더링 프레임이 없습니다.")
            sheet = np.concatenate(sheet_frames, axis=1)
            imageio.imwrite(str(out_png), sheet)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            _close_env(env)

    if out_mp4 is not None and verbose:
        print(f"  영상 저장: {out_mp4}")
    if out_png is not None and verbose:
        print(f"  장면 모음 저장: {out_png}")
    return {
        "distance": round(float(first_ep_dist), 2),
        "reward": round(float(first_ep_reward), 1),
        "episode_steps": int(first_ep_len),
        "episode_seconds": round(first_ep_len * dt, 2),
        "falls": falls,
        "episodes": episodes,
        "seconds": seconds,
    }


def random_policy(env, strength: float = 0.5) -> PolicyFn:
    """무작위 action. strength 로 세기를 조절 (0~1)."""
    def fn(obs):
        return env.action_space.sample() * strength
    return fn


def zero_policy(env) -> PolicyFn:
    """아무 힘도 주지 않음 (중력에 맡김)."""
    def fn(obs):
        return np.zeros(env.action_space.shape, dtype=np.float32)
    return fn


def wiggle_policy(env, strength: float = 0.5, hz: float = 1.0) -> PolicyFn:
    """각 모터를 서로 다른 위상의 사인파로 흔듭니다. 어느 joint 가 어떻게 움직이는지 보기 좋음."""
    n = env.action_space.shape[0]
    phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dt = env.unwrapped.dt
    counter = {"t": 0.0}

    def fn(obs):
        counter["t"] += dt
        return (strength * np.sin(2 * np.pi * hz * counter["t"] + phases)).astype(np.float32)
    return fn


def describe_result(res: dict) -> str:
    return (f"첫 에피소드: {res['episode_seconds']}초 동안 {res['distance']:.2f} m 전진, 누적 reward {res['reward']:.1f} | "
            f"{res['seconds']}초 동안 넘어진 횟수 {res['falls']}")
