"""공용 함수: 경로, 로그 출력, 최신 run 찾기, 플랫폼별 설정."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트 (이 파일의 상위 폴더)
ROOT = Path(__file__).resolve().parent.parent
CHARACTERS_DIR = ROOT / "characters"
GENERATED_DIR = CHARACTERS_DIR / "_generated"
RUNS_DIR = ROOT / "runs"
VIDEOS_DIR = ROOT / "videos"
PRETRAINED_DIR = ROOT / "pretrained"

TERRAINS = ("flat", "bumps", "bumps_hard")


def setup_gl_backend() -> None:
    """
    화면 없는 Linux(서버/WSL)에서 offscreen rendering 이 되도록 MUJOCO_GL 을 설정합니다.
    Windows / macOS 에서는 아무것도 하지 않습니다 (기본 backend 가 잘 동작함).
    """
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")


def is_windows() -> bool:
    return sys.platform.startswith("win")


def ensure_dirs() -> None:
    for d in (CHARACTERS_DIR, GENERATED_DIR, RUNS_DIR, VIDEOS_DIR, PRETRAINED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def check_ascii_path() -> str | None:
    """프로젝트 경로에 한글 등 non-ASCII 문자가 있으면 경고 문구를 돌려줍니다 (MuJoCo 가 XML 을 못 읽을 수 있음)."""
    try:
        str(ROOT).encode("ascii")
        return None
    except UnicodeEncodeError:
        return (
            f"프로젝트 경로에 영문/숫자가 아닌 글자가 있습니다: {ROOT}\n"
            "  MuJoCo 가 XML 파일을 열지 못할 수 있으니, 문제가 생기면 폴더를 C:\\Walker2D 처럼 영문 경로로 옮겨 주세요."
        )


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def fmt_minutes(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}분 {s:02d}초"


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def list_runs(character: str | None = None) -> list[Path]:
    """runs/ 안의 학습 결과 폴더를 오래된 순으로 돌려줍니다. character 를 주면 그 캐릭터의 run 만."""
    if not RUNS_DIR.exists():
        return []
    runs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "config.json").exists()]
    if character:
        runs = [p for p in runs if read_json(p / "config.json").get("character") == character]
    return sorted(runs, key=lambda p: p.stat().st_mtime)


def latest_run(character: str | None = None) -> Path | None:
    runs = list_runs(character)
    return runs[-1] if runs else None


def banner(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n {title}\n{line}")


def kv(label: str, value) -> None:
    print(f"  - {label}: {value}")
