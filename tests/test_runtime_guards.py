"""MuJoCo/Gymnasium 설치 전에도 실행할 수 있는 런타임 방어 회귀 테스트."""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _import_character_env_without_optional_dependencies():
    """테스트 대상의 순수 Python 방어 로직만 검사하도록 선택 의존성을 최소 stub으로 대체합니다."""
    gym = types.ModuleType("gymnasium")
    gym.Wrapper = object
    gym.Env = object
    gym.spaces = types.SimpleNamespace(Box=object)
    mujoco = types.ModuleType("mujoco")
    with patch.dict(sys.modules, {"gymnasium": gym, "mujoco": mujoco}):
        sys.modules.pop("walker_rl.character_env", None)
        return importlib.import_module("walker_rl.character_env")


character_env = _import_character_env_without_optional_dependencies()


class ExtraObservationGuardTests(unittest.TestCase):
    @staticmethod
    def _bare_env(fn):
        env = object.__new__(character_env.CharacterEnv)
        env.reward_module = types.SimpleNamespace(extra_observation=fn)
        env._n_extra = None
        env._has_extra_observation = True
        return env

    def test_non_finite_extra_observation_is_rejected(self):
        env = self._bare_env(lambda state: [0.0, np.nan, np.inf])
        with self.assertRaisesRegex(ValueError, "NaN/inf"):
            env._extra_obs(object())

    def test_extra_observation_length_is_fixed_after_first_call(self):
        values = iter(([1.0, 2.0], [3.0]))
        env = self._bare_env(lambda state: next(values))
        self.assertEqual(env._extra_obs(object()).size, 2)
        with self.assertRaisesRegex(ValueError, "길이가 실행 중 바뀌었습니다"):
            env._extra_obs(object())

    def test_non_finite_base_and_custom_rewards_are_rejected(self):
        class FakeInnerEnv:
            def __init__(self, reward):
                self.reward = reward

            def step(self, action):
                return np.zeros(1), self.reward, False, False, {}

        def make_env(base_reward, compute_reward=None):
            env = object.__new__(character_env.CharacterEnv)
            env.env = FakeInnerEnv(base_reward)
            env._step = 0
            env._make_state = lambda action: object()
            env._augment = lambda obs, state: obs
            env.reward_module = (
                types.SimpleNamespace(compute_reward=compute_reward)
                if compute_reward is not None
                else None
            )
            return env

        with self.assertRaisesRegex(ValueError, "기본 환경 reward"):
            make_env(np.inf).step(np.zeros(1))
        with self.assertRaisesRegex(ValueError, "유한한 숫자"):
            make_env(1.0, lambda state, base, info: np.nan).step(np.zeros(1))


class RewardModuleIsolationTests(unittest.TestCase):
    def test_named_reward_module_is_loaded_as_a_fresh_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "temporary_reward_for_test.py"
            path.write_text("values = []\n", encoding="utf-8")
            sys.path.insert(0, tmp)
            try:
                importlib.invalidate_caches()
                first = character_env.load_reward_module("temporary_reward_for_test")
                second = character_env.load_reward_module("temporary_reward_for_test")
                first.values.append("only-first")
                self.assertIsNot(first, second)
                self.assertEqual(second.values, [])
            finally:
                sys.path.remove(tmp)
                sys.modules.pop("temporary_reward_for_test", None)
                importlib.invalidate_caches()


class CheckScriptTests(unittest.TestCase):
    @staticmethod
    def _load_check_script():
        spec = importlib.util.spec_from_file_location("walker_check_script_for_test", ROOT / "0_check.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_main_returns_success_only_when_all_checks_pass(self):
        module = self._load_check_script()

        def all_pass(name, fn, hint=""):
            module.results.append(True)
            return True

        with patch.object(module, "check", all_pass), patch.object(sys, "argv", ["0_check.py", "--no-window"]):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 0)

        def all_fail(name, fn, hint=""):
            module.results.append(False)
            return False

        with patch.object(module, "check", all_fail), patch.object(sys, "argv", ["0_check.py", "--no-window"]):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 1)

    def test_command_line_uses_main_result_as_process_exit_code(self):
        source = (ROOT / "0_check.py").read_text(encoding="utf-8")
        self.assertIn("raise SystemExit(main())", source)


class BatchFileEncodingTests(unittest.TestCase):
    def test_batch_files_remain_cp949_with_crlf(self):
        for name in ("setup.bat", "open_terminal.bat"):
            raw = (ROOT / name).read_bytes()
            text = raw.decode("cp949")
            self.assertNotIn(b"\n", raw.replace(b"\r\n", b""), name)
            self.assertIn("Python", text)


if __name__ == "__main__":
    unittest.main()
