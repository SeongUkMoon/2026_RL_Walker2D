"""
설치 점검 - 수업 전에 한 번 실행해서 "모든 점검 통과" 가 나오면 준비 완료입니다.

  python 0_check.py              라이브러리 + 물리 시뮬 + 화면 창(3초) 점검
  python 0_check.py --no-window  창 띄우기 생략 (원격 접속 등)

문제가 있으면 [실패] 줄과 힌트가 출력됩니다. 해결이 어려우면 그 출력 전체를 복사해 AI 에게 물어보거나 강사에게 보여 주세요.
"""
import argparse
import platform
import sys
import time

results = []


def check(name, fn, hint=""):
    """검사 하나를 실행하고 결과를 기록합니다. 실패는 다음 검사로 전파하지 않습니다."""
    try:
        detail = fn()
        print(f"  [OK]   {name}" + (f"  ({detail})" if detail else ""))
        results.append(True)
        return True
    except Exception as e:
        print(f"  [실패] {name}: {type(e).__name__}: {e}")
        if hint:
            print(f"         힌트: {hint}")
        if isinstance(e, ModuleNotFoundError):
            missing = getattr(e, "name", None) or "필요한 패키지"
            print(f"         진단: Python 패키지 '{missing}'을(를) 불러오지 못했습니다.")
        elif isinstance(e, FileNotFoundError):
            print("         진단: 필요한 파일 또는 실행 파일이 없거나 경로가 바뀌었습니다.")
        elif isinstance(e, PermissionError):
            print("         진단: 파일/폴더 접근 권한이 없습니다. 동기화·보안 프로그램도 확인하세요.")
        elif isinstance(e, OSError):
            print("         진단: 운영체제·그래픽 드라이버·실행 파일 관련 오류일 수 있습니다.")
        results.append(False)
        return False


def main() -> int:
    # 같은 Python 프로세스에서 main()을 다시 호출해도 이전 결과가 섞이지 않게 합니다.
    results.clear()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--no-window", action="store_true", help="화면 창 점검 생략")
    args = p.parse_args()

    print("=" * 60)
    print(" Walker2D 실습 환경 점검")
    print("=" * 60)
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro}  |  {platform.system()} {platform.release()}  |  {sys.executable}")
    if (v.major, v.minor) != (3, 11):
        print(f"  [주의] 이 실습은 Python 3.11 기준으로 준비되었습니다. (현재 {v.major}.{v.minor}) 대부분 동작하지만 문제가 생기면 3.11 을 설치하세요.")
    if ".venv" not in sys.executable and "venv" not in sys.executable.lower():
        print("  [주의] 가상환경(.venv)이 아닌 Python 으로 실행 중입니다. setup.bat 을 먼저 실행했는지, 터미널에서 .venv\\Scripts\\activate 를 했는지 확인하세요.")

    from walker_rl.utils import check_ascii_path
    w = check_ascii_path()
    if w:
        print("  [주의] " + w)

    print("\n1) 라이브러리")

    def v_mujoco():
        import mujoco
        return f"mujoco {mujoco.__version__}"

    def v_gym():
        import gymnasium
        return f"gymnasium {gymnasium.__version__}"

    def v_sb3():
        import stable_baselines3
        import torch
        return f"stable-baselines3 {stable_baselines3.__version__}, torch {torch.__version__}"

    def v_ffmpeg():
        import imageio_ffmpeg
        return "ffmpeg: " + imageio_ffmpeg.get_ffmpeg_exe().split("\\")[-1].split("/")[-1]

    def v_mpl():
        import matplotlib
        return f"matplotlib {matplotlib.__version__}"

    check("MuJoCo (물리 엔진)", v_mujoco, "pip install -r requirements.txt 를 다시 실행하세요")
    check("Gymnasium (환경)", v_gym)
    check("Stable-Baselines3 + PyTorch (강화학습)", v_sb3, "torch 설치가 오래 걸릴 수 있습니다. setup.bat 을 끝까지 기다렸는지 확인")
    check("imageio-ffmpeg (영상 저장)", v_ffmpeg)
    check("matplotlib (그래프)", v_mpl)

    print("\n2) 물리 시뮬레이션")

    def sim():
        from walker_rl.character_env import make_env
        env = make_env("walker")
        obs, _ = env.reset(seed=0)
        t0 = time.time()
        n = 0
        for _ in range(500):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            n += 1
            if term or trunc:
                env.reset()
        env.close()
        return f"observation {obs.shape[0]}개, 무작위 action 500 step 실행, {n / (time.time() - t0):,.0f} steps/s"

    check("기본 Walker2D 환경 만들고 실행", sim)

    def build():
        from walker_rl.spec2mjcf import build_character
        from walker_rl.utils import CHARACTERS_DIR
        xml, meta = build_character(CHARACTERS_DIR / "dog.json")
        return f'dog.json -> {xml.name}, 모터 {meta["n_motors"]}개'

    check("캐릭터 JSON -> XML 변환 (dog 예제)", build)

    def offscreen():
        from walker_rl.character_env import make_env
        env = make_env("dog", render_mode="rgb_array")
        env.reset(seed=0)
        frame = env.render()
        env.close()
        return f"화면 캡처 {frame.shape[1]}x{frame.shape[0]}"

    check("화면 캡처(offscreen rendering)", offscreen,
          "그래픽 드라이버 문제일 수 있습니다. 노트북 그래픽 드라이버를 최신으로 업데이트해 보세요")

    def learn():
        from stable_baselines3 import PPO
        from walker_rl.training import make_vec_env
        venv = make_vec_env("dog", "my_reward", n_envs=2)
        model = PPO("MlpPolicy", venv, n_steps=64, batch_size=64, device="cpu", verbose=0)
        model.learn(total_timesteps=128)
        venv.close()
        return "PPO 128 steps 학습 성공"

    check("강화학습 짧게 실행 (PPO)", learn)

    if not args.no_window:
        print("\n3) 화면 창 (3초 동안 창이 뜨면 성공)")

        def window():
            from walker_rl.character_env import make_env
            env = make_env("walker", render_mode="human")
            env.reset(seed=0)
            t0 = time.time()
            while time.time() - t0 < 3.0:
                obs, r, term, trunc, info = env.step(env.action_space.sample() * 0.3)
                if term or trunc:
                    env.reset()
            env.close()
            return "창 열기/닫기 성공"

        check("실시간 창 띄우기", window,
              "창이 안 뜨면 --record 옵션으로 영상 저장 방식만 써도 실습은 가능합니다")

    print("\n" + "=" * 60)
    if all(results):
        print(" 모든 점검 통과! 수업 준비가 끝났습니다.")
        print(" 다음:  python 1_run_walker.py")
    else:
        print(f" {results.count(False)}개 항목 실패. 위의 [실패] 줄과 힌트를 확인하세요.")
        print(" 이 출력을 그대로 복사해서 AI 에게 '이 오류를 해결해 달라'고 물어봐도 좋습니다.")
    print("=" * 60)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
