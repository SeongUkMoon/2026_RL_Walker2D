"""MuJoCo 없이 실행 가능한 runner 회귀 테스트."""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from walker_rl.runner import run_record


class _FakeEnv:
    def __init__(self, *, dt: float = 0.1, terminate_at: int | None = None):
        self.unwrapped = self
        self.dt = dt
        self.data = types.SimpleNamespace(qpos=np.array([0.0]))
        self.terminate_at = terminate_at
        self.steps = 0
        self.render_calls = 0
        self.closed = False

    def reset(self, seed=None):
        self.steps = 0
        self.data.qpos[0] = 0.0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        self.data.qpos[0] += 0.1
        terminated = self.terminate_at is not None and self.steps >= self.terminate_at
        return np.zeros(2, dtype=np.float32), 1.0, terminated, False, {}

    def render(self):
        self.render_calls += 1
        return np.full((4, 4, 3), self.render_calls, dtype=np.uint8)

    def close(self):
        self.closed = True


class _CountingWriter:
    def __init__(self):
        self.append_count = 0
        self.closed = False

    def append_data(self, frame):
        self.append_count += 1

    def close(self):
        self.closed = True


class RunnerTests(unittest.TestCase):
    def test_headless_mode_never_renders_or_imports_imageio(self):
        env = _FakeEnv()
        with patch.dict(sys.modules, {"imageio": None, "imageio.v2": None}):
            result = run_record(
                env,
                lambda obs: np.zeros(1, dtype=np.float32),
                seconds=0.5,
                out_mp4=None,
                out_png=None,
                verbose=False,
            )
        self.assertEqual(env.render_calls, 0)
        self.assertTrue(env.closed)
        self.assertEqual(result["episode_steps"], 5)
        self.assertAlmostEqual(result["distance"], 0.5)

    def test_video_is_streamed_and_contact_sheet_is_bounded(self):
        env = _FakeEnv(dt=0.01)
        writer = _CountingWriter()
        saved_sheets = []
        imageio_package = types.ModuleType("imageio")
        imageio_v2 = types.ModuleType("imageio.v2")
        imageio_v2.get_writer = lambda *args, **kwargs: writer
        imageio_v2.imwrite = lambda path, data: saved_sheets.append(np.asarray(data))
        imageio_package.v2 = imageio_v2

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"imageio": imageio_package, "imageio.v2": imageio_v2}
        ):
            run_record(
                env,
                lambda obs: np.zeros(1, dtype=np.float32),
                seconds=1.0,
                fps=10,
                out_mp4=Path(tmp) / "test.mp4",
                out_png=Path(tmp) / "test.png",
                verbose=False,
            )

        self.assertTrue(writer.closed)
        self.assertEqual(writer.append_count, 11)  # t=0..0.9와 마지막 t=1.0
        self.assertEqual(len(saved_sheets), 1)
        self.assertLessEqual(saved_sheets[0].shape[1], 12)  # 절반 너비 2 * 최대 6장
        self.assertLess(env.render_calls, 20)  # 101장을 모두 보관/렌더링하던 회귀 방지

    def test_environment_closes_when_policy_raises(self):
        env = _FakeEnv()

        def broken_policy(obs):
            raise RuntimeError("policy failed")

        with self.assertRaisesRegex(RuntimeError, "policy failed"):
            run_record(env, broken_policy, seconds=0.2, verbose=False)
        self.assertTrue(env.closed)

    def test_environment_closes_when_video_finalization_raises(self):
        env = _FakeEnv()

        class BrokenCloseWriter(_CountingWriter):
            def close(self):
                raise RuntimeError("encoder close failed")

        imageio_package = types.ModuleType("imageio")
        imageio_v2 = types.ModuleType("imageio.v2")
        imageio_v2.get_writer = lambda *args, **kwargs: BrokenCloseWriter()
        imageio_package.v2 = imageio_v2
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"imageio": imageio_package, "imageio.v2": imageio_v2}
        ):
            with self.assertRaisesRegex(RuntimeError, "encoder close failed"):
                run_record(
                    env,
                    lambda obs: np.zeros(1, dtype=np.float32),
                    seconds=0.1,
                    out_mp4=Path(tmp) / "test.mp4",
                    verbose=False,
                )
        self.assertTrue(env.closed)

    def test_invalid_duration_is_rejected_before_reset(self):
        env = _FakeEnv()
        with self.assertRaisesRegex(ValueError, "seconds"):
            run_record(env, lambda obs: obs, seconds=0)
        self.assertEqual(env.steps, 0)
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
