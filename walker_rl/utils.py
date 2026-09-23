"""공용 함수: 경로, 로그 출력, 최신 run 찾기, 플랫폼별 설정."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
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
RUN_REQUIRED_FILES = ("model.zip", "vecnormalize.pkl", "config.json")


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


def file_sha256(path: str | Path) -> str:
    """파일 내용을 식별할 수 있는 SHA-256 해시를 돌려줍니다."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_entry(path: Path, run_dir: Path) -> dict:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": file_sha256(path),
    }


def snapshot_character_artifacts(character_ref: str, xml_path: str | Path, meta: dict,
                                 run_dir: str | Path) -> dict:
    """학습에 실제 사용한 캐릭터 파일을 run 폴더 안에 고정하고 그 해시를 기록합니다."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    xml_copy = run_dir / "character_used.xml"
    meta_copy = run_dir / "character_used.meta.json"
    shutil.copy2(Path(xml_path), xml_copy)
    write_json(meta_copy, meta)

    result = {
        "xml": _artifact_entry(xml_copy, run_dir),
        "meta": _artifact_entry(meta_copy, run_dir),
        "source_json": None,
    }

    ref = Path(character_ref)
    if ref.suffix.lower() == ".json" and ref.exists():
        source = ref
    elif not ref.suffix:
        candidate = CHARACTERS_DIR / f"{character_ref}.json"
        source = candidate if candidate.exists() else None
    else:
        source = None
    if source is not None:
        source_copy = run_dir / "character_source.json"
        shutil.copy2(source, source_copy)
        result["source_json"] = _artifact_entry(source_copy, run_dir)
    return result


def resolve_run_artifact(run_dir: str | Path, config: dict, key: str) -> Path | None:
    """config 의 캐릭터 snapshot 경로를 안전하게 풀고 내용 해시까지 확인합니다."""
    artifacts = config.get("character_artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("character_artifacts 형식이 올바르지 않습니다.")
    entry = artifacts.get(key)
    if not entry:
        return None
    if isinstance(entry, str):
        relative_path, expected_hash = entry, None
    elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
        relative_path, expected_hash = entry["path"], entry.get("sha256")
    else:
        raise ValueError(f"character_artifacts.{key} 형식이 올바르지 않습니다.")

    base = Path(run_dir).resolve()
    path = (base / relative_path).resolve()
    try:
        path.relative_to(base)
    except ValueError as e:
        raise ValueError(f"character_artifacts.{key} 경로가 run 폴더 밖을 가리킵니다: {relative_path}") from e
    if not path.is_file():
        raise FileNotFoundError(f"캐릭터 snapshot 파일이 없습니다: {path}")
    if expected_hash and file_sha256(path) != expected_hash:
        raise ValueError(f"캐릭터 snapshot 파일이 학습 후 변경되었습니다: {path.name}")
    return path


def fmt_minutes(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}분 {s:02d}초"


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def list_runs(character: str | None = None) -> list[Path]:
    """필수 산출물을 모두 가진 완료 run을 오래된 순으로 돌려줍니다."""
    if not RUNS_DIR.exists():
        return []
    wanted = Path(character).stem if character else None
    runs = []
    for path in RUNS_DIR.iterdir():
        if not path.is_dir() or not all((path / name).is_file() for name in RUN_REQUIRED_FILES):
            continue
        try:
            config = read_json(path / "config.json")
            if not isinstance(config, dict):
                continue
        except (OSError, ValueError, TypeError):
            continue
        if wanted is None or config.get("character") == wanted:
            runs.append(path)
    return sorted(runs, key=lambda p: p.stat().st_mtime)


def latest_run(character: str | None = None) -> Path | None:
    runs = list_runs(character)
    return runs[-1] if runs else None


def banner(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n {title}\n{line}")


def kv(label: str, value) -> None:
    print(f"  - {label}: {value}")
