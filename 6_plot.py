"""
2교시 - 학습 곡선(learning curve) 비교

runs/ 에 저장된 학습 결과들의 진행 기록(progress.csv)을 읽어 두 개의 그래프를 그립니다.
  왼쪽 : 평균 reward  vs 학습 step        오른쪽 : 평균 전진 거리(m) vs 학습 step
reward 를 바꾸면 reward 값 자체는 비교가 안 되니(점수 기준이 다름) 오른쪽 '전진 거리'나 영상으로 비교하세요.

사용 예
  python 6_plot.py                          모든 run (최근 8개)
  python 6_plot.py --character dog          dog 로 학습한 run 만
  python 6_plot.py --runs runs/dog__my_reward__20260923-1530 runs/dog__my_reward_v2__20260923-1610
  python 6_plot.py --x minutes              가로축을 학습 시간(분)으로
  python 6_plot.py --show                   PNG 저장 후 창으로도 보기
"""
import argparse
import csv
import sys
from pathlib import Path

from walker_rl.utils import VIDEOS_DIR, banner, kv, list_runs, read_json

# 색: 구분되는 순서가 검증된 categorical palette (순서를 바꾸지 마세요)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def read_progress(run_dir: Path):
    f = run_dir / "progress.csv"
    if not f.exists():
        return None
    rows = []
    with open(f, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except (TypeError, ValueError):
                continue
    return rows or None


def short_label(run_dir: Path) -> str:
    # runs/dog__my_reward__20260923-153000  ->  "dog / my_reward (15:30)",   pretrained/dog -> "pretrained/dog"
    parts = run_dir.name.split("__")
    if len(parts) < 2:
        return f"{run_dir.parent.name}/{run_dir.name}"
    ts = parts[-1]
    hhmm = f"{ts[-6:-4]}:{ts[-4:-2]}" if len(ts) >= 6 and ts[-6:].isdigit() else ts
    return " / ".join(parts[:-1]) + f" ({hhmm})"


def direct_label(run_dir: Path, runs: list) -> str:
    """선 끝에 붙이는 짧은 라벨. 모든 run 이 같은 캐릭터면 캐릭터 이름은 생략."""
    label = short_label(run_dir)
    firsts = {short_label(r).split(" / ")[0] for r in runs}
    if len(firsts) == 1 and " / " in label:
        return label.split(" / ", 1)[1]
    return label


def setup_korean_font():
    """Windows(Malgun Gothic) / macOS(AppleGothic) / Linux(NanumGothic) 한글 글꼴 설정. 없으면 영문 라벨 사용."""
    import matplotlib
    from matplotlib import font_manager
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in installed:
            matplotlib.rcParams["font.family"] = c
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--character", "-c", default=None, help="이 캐릭터의 run 만")
    p.add_argument("--runs", nargs="*", default=None, help="비교할 run 폴더들")
    p.add_argument("--x", choices=["steps", "minutes"], default="steps", help="가로축 (기본 steps)")
    p.add_argument("--out", default=None, help="저장할 PNG 경로 (기본 videos/learning_curves.png)")
    p.add_argument("--show", action="store_true", help="창으로도 보기")
    args = p.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    ko = setup_korean_font()
    T = {  # 라벨 (한글 글꼴이 없으면 영어)
        "reward": "평균 reward (에피소드 합)" if ko else "mean episode reward",
        "dist": "평균 전진 거리 (m)" if ko else "mean distance (m)",
        "steps": "학습 step" if ko else "training steps",
        "minutes": "학습 시간 (분)" if ko else "training time (min)",
        "title": "학습 곡선 비교" if ko else "Learning curves",
    }

    runs = [Path(r) for r in args.runs] if args.runs else list_runs(args.character)
    runs = [r for r in runs if read_progress(r)]
    if not runs:
        print("[실패] 그릴 run 이 없습니다. 먼저 4_train.py 로 학습하세요 (progress.csv 는 30초마다 기록됩니다).")
        sys.exit(1)
    if len(runs) > len(SERIES):
        print(f"  run 이 {len(runs)}개라 최근 {len(SERIES)}개만 그립니다. (--runs 로 직접 고를 수 있습니다)")
        runs = runs[-len(SERIES):]

    banner(T["title"])
    xkey = "timesteps" if args.x == "steps" else "elapsed_sec"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=130, facecolor=SURFACE)
    for ax, ykey, ylabel in ((axes[0], "mean_reward", T["reward"]), (axes[1], "mean_distance", T["dist"])):
        ax.set_facecolor(SURFACE)
        for i, run in enumerate(runs):
            rows = read_progress(run)
            xs = [r[xkey] / (60.0 if xkey == "elapsed_sec" else 1.0) for r in rows]
            ys = [r[ykey] for r in rows]
            color = SERIES[i % len(SERIES)]
            ax.plot(xs, ys, color=color, linewidth=2, label=short_label(run), solid_capstyle="round")
            if len(runs) <= 4 and xs:  # 직접 라벨 (4개 이하일 때)
                ax.annotate(direct_label(run, runs), xy=(xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            fontsize=8, color=INK2, va="center")
        ax.set_xlabel(T[args.x], color=INK2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
        if xkey == "timesteps":
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v < 1e6 else f"{v / 1e6:.1f}M"))
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")
    fig.suptitle(T["title"], color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout()

    out = Path(args.out) if args.out else VIDEOS_DIR / "learning_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    for r in runs:
        res = read_json(r / "result.json") if (r / "result.json").exists() else {}
        kv(short_label(r), f"마지막 평균 reward {res.get('mean_reward_last')}, 결과 영상 전진 {res.get('eval_distance')} m")
    print(f"\n  그래프 저장: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
