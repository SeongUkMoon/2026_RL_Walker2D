"""
캐릭터 JSON spec -> MuJoCo XML(MJCF) 변환기

spec 은 "옆에서 본(side view) 2D 그림" 입니다.
  - x : 오른쪽(+)이 캐릭터의 앞(전진 방향), z : 위(+).  단위는 m.
  - 캐릭터는 capsule(캡슐) 모양의 segment(마디) 여러 개로 이루어지고,
    각 segment 는 parent segment 에 hinge joint(회전 관절)로 붙습니다.
  - parent 가 null 인 segment 가 정확히 하나 있어야 하며, 그것이 torso(몸통, root) 입니다.

segment 필드
  name         : 영문 이름 (고유해야 함)
  parent       : 붙을 부모 segment 이름. 몸통(root)만 null
  start        : [x, z] 시작점 = 부모와 연결되는 joint 위치
  end          : [x, z] 끝점
  radius       : 캡슐 두께(반지름) m
  joint_range  : [min, max] 회전 범위 (deg). 기본 [-90, 90]
  motor        : true/false. 이 joint 에 모터(actuator)를 달지 여부. 기본 true
  friction     : 바닥 마찰. 기본 0.9 (발은 1.5~2.0 추천)
  color        : "brown", "purple", "blue", "red", "green", "gray", "yellow", "white" 또는 "r g b a"

options (선택)
  strength           : 모터 힘 배율 (기본 1.0)
  fall_height_ratio  : 몸통 높이가 처음의 이 비율 아래로 내려가면 '넘어짐'으로 판정 (기본 0.55)
  fall_angle_deg     : 몸통 기울기가 이 각도(deg)를 넘으면 '넘어짐' (기본 60)
  frame_skip         : 한 번의 action 이 유지되는 물리 스텝 수 (기본 4  -> 제어 주기 0.008 s)

생성된 XML 은 Gymnasium 의 Walker2d-v5 환경에 xml_file 로 바로 넣을 수 있게
root joint 3개(rootx: slide x, rootz: slide z, rooty: hinge y)를 같은 순서로 넣습니다.
"""
from __future__ import annotations

import copy
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .utils import CHARACTERS_DIR, GENERATED_DIR, read_json, write_json

# Gymnasium 기본 Walker2d 의 총 질량(kg). 모터 힘(gear)을 질량에 비례해 맞추는 기준값.
WALKER_MASS_REF = 23.7
WALKER_GEAR_REF = 100.0
WALKER_TORSO_Z_REF = 1.25

COLOR_PRESETS = {
    "brown": "0.8 0.6 0.4 1",
    "purple": "0.7 0.3 0.6 1",
    "blue": "0.3 0.5 0.9 1",
    "red": "0.9 0.3 0.3 1",
    "green": "0.3 0.8 0.4 1",
    "gray": "0.6 0.6 0.6 1",
    "yellow": "0.95 0.85 0.3 1",
    "white": "0.95 0.95 0.95 1",
    "black": "0.15 0.15 0.15 1",
}

DEFAULT_OPTIONS = {
    "strength": 1.0,
    "fall_height_ratio": 0.55,
    "fall_angle_deg": 60.0,
    "frame_skip": 4,
    "timestep": 0.002,
}

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$", re.ASCII)
SEGMENT_FIELDS = {
    "name", "parent", "start", "end", "radius", "joint_range", "motor", "friction", "color", "rgba",
}
OPTION_FIELDS = set(DEFAULT_OPTIONS)

# terrain 종류별 장애물(bump) 목록: (x 위치, 폭, 높이).  기본 Walker2d(높이 1.25 m) 기준 값이며
# 캐릭터 크기에 비례해서 scale 됩니다. (이전 실습 프로젝트의 bump terrain 아이디어를 참고)
TERRAIN_BUMPS = {
    "flat": [],
    "bumps": [(6.0, 0.6, 0.12), (10.0, 0.8, 0.22)],
    "bumps_hard": [(3.0, 0.5, 0.06), (5.0, 0.4, 0.12), (7.0, 0.4, 0.18), (9.0, 0.6, 0.25),
                   (11.0, 0.4, 0.15), (12.0, 0.4, 0.30), (14.0, 0.8, 0.35), (16.0, 0.4, 0.20)],
}


class SpecError(ValueError):
    """spec 에 문제가 있을 때 (한글 메시지)."""


# --------------------------------------------------------------------------------------
# spec 읽기 / 검증
# --------------------------------------------------------------------------------------
def _strip_comments(obj):
    """'_' 로 시작하는 key 는 설명용 주석으로 보고 제거합니다."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _json_object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'중복된 JSON key가 있습니다: "{key}"')
        result[key] = value
    return result


def _reject_json_constant(value: str):
    # Python json 모듈은 기본적으로 JSON 표준에 없는 NaN/Infinity도 허용하므로 명시적으로 막습니다.
    raise ValueError(f"JSON 숫자는 유한해야 합니다: {value}")


def load_spec(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise SpecError(f"spec 파일을 찾을 수 없습니다: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            raw = json.load(
                file,
                object_pairs_hook=_json_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
    except Exception as e:  # JSON 문법 오류
        raise SpecError(
            f"JSON 문법 오류입니다: {e}\n  (쉼표 누락, 따옴표 짝, 마지막 항목 뒤의 쉼표 등을 확인하세요. "
            f"오류 메시지를 AI 에게 그대로 붙여 넣으면 고쳐 줍니다.)"
        )
    if not isinstance(raw, dict):
        raise SpecError('JSON 최상위 값은 객체({...})여야 합니다. "segments" 목록을 가진 객체로 작성하세요.')
    spec = _strip_comments(raw)
    if "name" not in spec:
        spec["name"] = path.stem
    return spec


def _is_finite_number(value) -> bool:
    """JSON 숫자로 사용 가능한 유한한 실수인지 확인합니다. bool 은 숫자로 취급하지 않습니다."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_point(point) -> bool:
    return (
        isinstance(point, (list, tuple))
        and len(point) == 2
        and all(_is_finite_number(value) for value in point)
    )


def _color_error(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "문자열이어야 합니다."
    color = value.strip()
    if color in COLOR_PRESETS:
        return None
    parts = color.split()
    if len(parts) != 4:
        return f'알 수 없는 색상 {value!r}입니다. 색상 이름 또는 "r g b a" 숫자 4개를 사용하세요.'
    try:
        rgba = [float(part) for part in parts]
    except ValueError:
        return f'RGBA 값 {value!r}에 숫자가 아닌 항목이 있습니다.'
    if not all(math.isfinite(channel) and 0.0 <= channel <= 1.0 for channel in rgba):
        return "RGBA 네 값은 모두 0~1 사이의 유한한 숫자여야 합니다."
    return None


def _dist_point_segment(p, a, b) -> float:
    """점 p 와 선분 ab 사이의 거리."""
    px, pz = p
    ax, az = a
    bx, bz = b
    dx, dz = bx - ax, bz - az
    L2 = dx * dx + dz * dz
    if L2 < 1e-12:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / L2))
    cx, cz = ax + t * dx, az + t * dz
    return math.hypot(px - cx, pz - cz)


def validate_spec(spec: dict) -> list[str]:
    """spec 을 검사합니다. 치명적 오류는 SpecError, 학습 품질 문제는 경고로 반환합니다."""
    if not isinstance(spec, dict):
        raise SpecError('spec 은 "segments" 목록을 가진 객체(dict)여야 합니다.')
    # 파일을 거치지 않고 dict 를 직접 넘겨도 설명용/AI 메모 필드의 동작이 같습니다.
    spec = _strip_comments(spec)

    errors: list[str] = []
    warnings: list[str] = []

    unknown_top = sorted(str(key) for key in spec if key not in {"name", "segments", "options"})
    if unknown_top:
        errors.append(f"최상위에 알 수 없는 필드가 있습니다: {', '.join(unknown_top)}")

    model_name = spec.get("name", "custom2d")
    if not isinstance(model_name, str) or not NAME_RE.fullmatch(model_name):
        errors.append('name 은 영문 소문자로 시작하고 영문 소문자/숫자/밑줄(_)만 사용해야 합니다.')

    options = spec.get("options", {})
    if not isinstance(options, dict):
        errors.append('"options" 는 객체({...})여야 합니다.')
        options = {}
    else:
        unknown_options = sorted(str(key) for key in options if key not in OPTION_FIELDS)
        if unknown_options:
            errors.append(f"options 에 알 수 없는 필드가 있습니다: {', '.join(unknown_options)}")

        strength = options.get("strength", DEFAULT_OPTIONS["strength"])
        if not _is_finite_number(strength) or not 0.0 < float(strength) <= 10.0:
            errors.append(f"options.strength 는 0보다 크고 10 이하인 유한한 숫자여야 합니다. 현재 {strength!r}")
        fall_ratio = options.get("fall_height_ratio", DEFAULT_OPTIONS["fall_height_ratio"])
        if not _is_finite_number(fall_ratio) or not 0.0 < float(fall_ratio) <= 1.0:
            errors.append(f"options.fall_height_ratio 는 0보다 크고 1 이하인 유한한 숫자여야 합니다. 현재 {fall_ratio!r}")
        fall_angle = options.get("fall_angle_deg", DEFAULT_OPTIONS["fall_angle_deg"])
        if not _is_finite_number(fall_angle) or not 0.0 < float(fall_angle) <= 180.0:
            errors.append(f"options.fall_angle_deg 는 0보다 크고 180 이하인 유한한 숫자여야 합니다. 현재 {fall_angle!r}")
        frame_skip = options.get("frame_skip", DEFAULT_OPTIONS["frame_skip"])
        if not isinstance(frame_skip, int) or isinstance(frame_skip, bool) or not 1 <= frame_skip <= 20:
            errors.append(f"options.frame_skip 은 1~20 사이의 정수여야 합니다. 현재 {frame_skip!r}")
        timestep = options.get("timestep", DEFAULT_OPTIONS["timestep"])
        if not _is_finite_number(timestep) or not 0.0001 <= float(timestep) <= 0.02:
            errors.append(f"options.timestep 은 0.0001~0.02 사이의 유한한 숫자여야 합니다. 현재 {timestep!r}")

    segs = spec.get("segments")
    if not isinstance(segs, list) or not segs:
        errors.append('"segments" 는 하나 이상의 segment 객체를 담은 목록이어야 합니다.')
        raise SpecError("spec 오류:\n  - " + "\n  - ".join(errors))
    if len(segs) > 32:
        errors.append(f"segment 가 {len(segs)}개입니다. 안전한 실습을 위해 32개 이하로 줄이세요.")

    valid_segs: list[dict] = []
    for i, segment in enumerate(segs):
        if not isinstance(segment, dict):
            errors.append(f"segment #{i + 1}: 객체({{...}})여야 합니다. 현재 {type(segment).__name__}")
            continue
        valid_segs.append(segment)
        raw_name = segment.get("name")
        label = raw_name if isinstance(raw_name, str) and raw_name else "이름없음"
        tag = f"segment #{i + 1} ({label})"

        unknown_fields = sorted(str(key) for key in segment if key not in SEGMENT_FIELDS)
        if unknown_fields:
            errors.append(f"{tag}: 알 수 없는 필드가 있습니다: {', '.join(unknown_fields)}")
        if not isinstance(raw_name, str) or not NAME_RE.fullmatch(raw_name):
            errors.append(f"{tag}: name 은 영문 소문자로 시작하고 영문 소문자/숫자/밑줄(_)만 사용하세요.")
        if "parent" not in segment:
            errors.append(f"{tag}: parent 필드가 없습니다. root 만 null, 나머지는 부모 name 을 적으세요.")
        else:
            parent = segment["parent"]
            if parent is not None and (not isinstance(parent, str) or not NAME_RE.fullmatch(parent)):
                errors.append(f"{tag}: parent 는 null 또는 유효한 segment name 문자열이어야 합니다. 현재 {parent!r}")

        start_ok = _is_point(segment.get("start"))
        end_ok = _is_point(segment.get("end"))
        if not start_ok:
            errors.append(f"{tag}: start 는 유한한 숫자 두 개의 [x, z] 목록이어야 합니다.")
        if not end_ok:
            errors.append(f"{tag}: end 는 유한한 숫자 두 개의 [x, z] 목록이어야 합니다.")
        if start_ok and end_ok:
            length = math.dist(segment["start"], segment["end"])
            if length < 0.02:
                errors.append(f"{tag}: 길이가 너무 짧습니다 ({length:.3f} m). start 와 end 를 다르게 주세요.")

        radius = segment.get("radius", 0.05)
        if not _is_finite_number(radius) or not 0.005 <= float(radius) <= 0.6:
            errors.append(f"{tag}: radius 는 0.005~0.6 m 사이의 유한한 숫자여야 합니다. 현재 {radius!r}")

        friction = segment.get("friction", 0.9)
        if not _is_finite_number(friction) or not 0.01 <= float(friction) <= 10.0:
            errors.append(f"{tag}: friction 은 0.01~10 사이의 유한한 숫자여야 합니다. 현재 {friction!r}")

        if "motor" in segment and not isinstance(segment["motor"], bool):
            errors.append(f"{tag}: motor 는 문자열이 아닌 JSON boolean true 또는 false 여야 합니다. 현재 {segment['motor']!r}")

        for color_field in ("color", "rgba"):
            if color_field in segment:
                color_problem = _color_error(segment[color_field])
                if color_problem:
                    errors.append(f"{tag}: {color_field} {color_problem}")
        if "color" in segment and "rgba" in segment:
            errors.append(f"{tag}: color 와 rgba 중 하나만 사용하세요.")

        if segment.get("parent") is not None:
            joint_range = segment.get("joint_range", [-90, 90])
            range_ok = (
                isinstance(joint_range, (list, tuple))
                and len(joint_range) == 2
                and all(_is_finite_number(value) for value in joint_range)
            )
            if not range_ok:
                errors.append(f"{tag}: joint_range 는 유한한 숫자 두 개의 [최소, 최대] 목록이어야 합니다. 현재 {joint_range!r}")
            else:
                low, high = float(joint_range[0]), float(joint_range[1])
                if not low < high:
                    errors.append(f"{tag}: joint_range 는 최소 < 최대여야 합니다. 현재 {joint_range!r}")
                if low < -180.0 or high > 180.0:
                    errors.append(f"{tag}: joint_range 는 -180~180 deg 안에 있어야 합니다. 현재 {joint_range!r}")
                if not low <= 0.0 <= high:
                    errors.append(
                        f"{tag}: 초기 관절 각도 0 deg 가 joint_range {joint_range!r} 안에 있어야 합니다. "
                        "한쪽 관절은 [-150, 0] 또는 [0, 150]처럼 0을 끝값으로 포함하세요."
                    )

    valid_names = [segment.get("name") for segment in valid_segs if isinstance(segment.get("name"), str)]
    duplicates = sorted({name for name in valid_names if valid_names.count(name) > 1})
    for name in duplicates:
        errors.append(f'segment name "{name}" 이(가) 여러 번 나옵니다. name 은 고유해야 합니다.')

    roots = [segment for segment in valid_segs if "parent" in segment and segment.get("parent") is None]
    if len(roots) != 1:
        errors.append(f"parent 가 null 인 segment(몸통, root)가 정확히 1개 있어야 합니다. 현재 {len(roots)}개")

    by_name = {
        segment["name"]: segment
        for segment in valid_segs
        if isinstance(segment.get("name"), str) and segment["name"] not in duplicates
    }
    for segment in valid_segs:
        parent = segment.get("parent")
        if isinstance(parent, str) and parent not in by_name:
            errors.append(f'segment "{segment.get("name", "이름없음")}": parent "{parent}" 라는 segment 가 없습니다.')
        if isinstance(parent, str) and parent == segment.get("name"):
            errors.append(f'segment "{parent}" 는 자기 자신을 parent 로 지정할 수 없습니다.')

    # 부모 필드가 모두 안전할 때 순환 참조를 별도로 검사합니다.
    graph_is_safe = (
        len(by_name) == len(valid_segs)
        and len(roots) == 1
        and all(
            segment.get("parent") is None
            or (isinstance(segment.get("parent"), str) and segment.get("parent") in by_name)
            for segment in valid_segs
        )
    )
    if graph_is_safe:
        reported_cycles: set[str] = set()
        for segment in valid_segs:
            seen: set[str] = set()
            current = segment
            while current.get("parent") is not None:
                name = current["name"]
                if name in seen:
                    if segment["name"] not in reported_cycles:
                        errors.append(f'segment "{segment["name"]}" 의 parent 연결이 순환합니다.')
                        reported_cycles.add(segment["name"])
                    break
                seen.add(name)
                current = by_name[current["parent"]]

    if errors:
        raise SpecError("spec 오류:\n  - " + "\n  - ".join(errors))

    # ---- 경고 (실행은 되지만 학습 전에 확인이 필요한 것들) ----
    if len(segs) < 4 or len(segs) > 9:
        warnings.append(f"segment 가 {len(segs)}개입니다. 첫 실습은 4~9개가 다루기 쉽습니다.")

    motors = [segment for segment in valid_segs if segment.get("parent") is not None and segment.get("motor", True)]
    if not motors:
        raise SpecError("모터(motor: true)가 달린 joint 가 하나도 없습니다. 강화학습으로 움직일 수 없습니다.")
    if len(motors) < 4:
        warnings.append(f"모터가 {len(motors)}개뿐입니다. 다양한 움직임을 학습하려면 보통 4~8개를 권장합니다.")
    elif len(motors) > 8:
        warnings.append(f"모터가 {len(motors)}개입니다. 첫 실습은 4~8개가 학습과 관찰에 유리합니다.")

    bottoms = {
        segment["name"]: min(float(segment["start"][1]), float(segment["end"][1])) - float(segment.get("radius", 0.05))
        for segment in valid_segs
    }
    tops = {
        segment["name"]: max(float(segment["start"][1]), float(segment["end"][1])) + float(segment.get("radius", 0.05))
        for segment in valid_segs
    }
    lowest = min(bottoms.values())
    height = max(tops.values()) - lowest
    if height < 0.2:
        warnings.append(f"캐릭터 전체 높이가 {height:.2f} m 로 매우 작습니다. 0.5~2 m 정도를 추천합니다.")
    if height > 4.0:
        warnings.append(f"캐릭터 전체 높이가 {height:.2f} m 로 매우 큽니다. 0.5~2 m 정도를 추천합니다.")

    for segment in valid_segs:
        parent = segment.get("parent")
        if parent is None:
            continue
        parent_segment = by_name[parent]
        gap = _dist_point_segment(segment["start"], parent_segment["start"], parent_segment["end"])
        if gap > float(parent_segment.get("radius", 0.05)) + 0.05:
            warnings.append(
                f'segment "{segment["name"]}" 의 start 점이 부모 "{parent}"에서 {gap:.2f} m 떨어져 있습니다. '
                "허공에 매달린 것처럼 붙습니다. 의도한 게 아니면 start 를 부모 위의 점으로 옮기세요."
            )

    feet = [
        segment for segment in valid_segs
        if segment.get("parent") is not None and float(segment.get("friction", 0.9)) >= 1.2
    ]
    if not feet:
        warnings.append("friction 이 1.2 이상인 발 후보가 없습니다. 바닥에 닿는 마디에 friction 1.9 를 주세요.")
    else:
        foot_tolerance = max(0.05, min(0.20, 0.10 * height))
        for foot in feet:
            above_ground = bottoms[foot["name"]] - lowest
            if above_ground > foot_tolerance:
                warnings.append(
                    f'발 후보 "{foot["name"]}"가 캐릭터의 최저점보다 {above_ground:.2f} m 높습니다. '
                    "초기 자세에서 실제로 바닥에 닿는지 확인하세요."
                )
    return warnings


# --------------------------------------------------------------------------------------
# XML 생성
# --------------------------------------------------------------------------------------
def _fmt(*vals) -> str:
    return " ".join(f"{float(v):.4f}" for v in vals)


def _rgba(s: dict, default="brown") -> str:
    c = s.get("color", s.get("rgba", default))
    return COLOR_PRESETS.get(str(c), str(c))


def spec_to_mjcf(spec_in: dict, terrain: str = "flat") -> tuple[str, dict]:
    """spec dict -> (xml 문자열, meta dict)."""
    spec = copy.deepcopy(spec_in)
    warnings = validate_spec(spec)
    if terrain not in TERRAIN_BUMPS:
        raise SpecError(f'terrain 은 {list(TERRAIN_BUMPS)} 중 하나여야 합니다: "{terrain}"')
    opts = {**DEFAULT_OPTIONS, **spec.get("options", {})}
    segs = spec["segments"]
    by_name = {s["name"]: s for s in segs}
    root = next(s for s in segs if s.get("parent") is None)

    # 1) 바닥 맞추기: 가장 낮은 캡슐 표면이 z = 0.005 에 오도록 전체를 위/아래로 이동
    lowest = min(min(s["start"][1], s["end"][1]) - s.get("radius", 0.05) for s in segs)
    dz = -lowest + 0.005
    for s in segs:
        s["start"] = [float(s["start"][0]), float(s["start"][1]) + dz]
        s["end"] = [float(s["end"][0]), float(s["end"][1]) + dz]
    highest = max(max(s["start"][1], s["end"][1]) + s.get("radius", 0.05) for s in segs)
    xs = [v for s in segs for v in (s["start"][0], s["end"][0])]

    root_x = 0.5 * (root["start"][0] + root["end"][0])
    root_z = 0.5 * (root["start"][1] + root["end"][1])

    mj = ET.Element("mujoco", model=spec.get("name", "custom2d"))
    ET.SubElement(mj, "compiler", angle="degree", inertiafromgeom="true")
    ET.SubElement(mj, "option", integrator="RK4", timestep=f'{opts["timestep"]}')
    vis = ET.SubElement(mj, "visual")
    ET.SubElement(vis, "global", offwidth="1920", offheight="1080")  # 큰 화면으로 녹화 가능하게
    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "joint", armature="0.01", damping="0.1", limited="true")
    ET.SubElement(default, "geom", conaffinity="0", condim="3", contype="1", density="1000",
                  friction="0.9 0.1 0.1", rgba=COLOR_PRESETS["brown"])

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1=".4 .5 .6", rgb2="0 0 0",
                  width="100", height="100")
    ET.SubElement(asset, "texture", builtin="checker", height="100", name="texplane",
                  rgb1="0 0 0", rgb2="0.8 0.8 0.8", type="2d", width="100")
    ET.SubElement(asset, "material", name="MatPlane", reflectance="0.5", shininess="1",
                  specular="1", texrepeat="60 60", texture="texplane")

    wb = ET.SubElement(mj, "worldbody")
    ET.SubElement(wb, "light", cutoff="100", diffuse="1 1 1", dir="0 0 -1.3", directional="true",
                  exponent="1", pos="0 0 1.3", specular=".1 .1 .1")
    ET.SubElement(wb, "geom", conaffinity="1", condim="3", name="floor", pos="0 0 0",
                  rgba="0.8 0.9 0.8 1", size="100 100 100", type="plane", material="MatPlane")

    # 2) 몸통(root body): 위치 = 캡슐 중심, root joint 3개 (Walker2d 와 같은 순서!)
    origins = {root["name"]: (root_x, root_z)}
    bodies: dict[str, ET.Element] = {}
    torso = ET.SubElement(wb, "body", name=root["name"], pos=_fmt(root_x, 0, root_z))
    ET.SubElement(torso, "camera", name="track", mode="trackcom", pos="0 -3 -0.25", xyaxes="1 0 0 0 0 1")
    ET.SubElement(torso, "joint", armature="0", axis="1 0 0", damping="0", limited="false",
                  name="rootx", pos="0 0 0", stiffness="0", type="slide")
    ET.SubElement(torso, "joint", armature="0", axis="0 0 1", damping="0", limited="false",
                  name="rootz", pos="0 0 0", ref=f"{root_z:.4f}", stiffness="0", type="slide")
    ET.SubElement(torso, "joint", armature="0", axis="0 1 0", damping="0", limited="false",
                  name="rooty", pos="0 0 0", stiffness="0", type="hinge")
    ET.SubElement(torso, "geom", name=f'{root["name"]}_geom', type="capsule",
                  size=f'{root.get("radius", 0.05):.4f}', rgba=_rgba(root),
                  friction=str(root.get("friction", 0.9)),
                  fromto=_fmt(root["start"][0] - root_x, 0, root["start"][1] - root_z,
                              root["end"][0] - root_x, 0, root["end"][1] - root_z))
    bodies[root["name"]] = torso

    motor_joints: list[str] = []
    joint_names: list[str] = []
    seg_order: list[str] = [root["name"]]

    # 3) 자식 segment 들을 트리 순서대로 붙이기 (spec 에 적힌 순서 유지)
    def add_children(parent_name: str) -> None:
        for s in segs:
            if s.get("parent") != parent_name:
                continue
            px, pz = origins[parent_name]
            ox, oz = s["start"]
            body = ET.SubElement(bodies[parent_name], "body", name=s["name"], pos=_fmt(ox - px, 0, oz - pz))
            origins[s["name"]] = (ox, oz)
            lo, hi = s.get("joint_range", [-90, 90])
            jname = f'{s["name"]}_joint'
            ET.SubElement(body, "joint", axis="0 -1 0", name=jname, pos="0 0 0",
                          range=f"{lo} {hi}", type="hinge")
            ET.SubElement(body, "geom", name=f'{s["name"]}_geom', type="capsule",
                          size=f'{s.get("radius", 0.05):.4f}', friction=str(s.get("friction", 0.9)),
                          rgba=_rgba(s), fromto=_fmt(0, 0, 0, s["end"][0] - ox, 0, s["end"][1] - oz))
            bodies[s["name"]] = body
            joint_names.append(jname)
            seg_order.append(s["name"])
            if s.get("motor", True):
                motor_joints.append(jname)
            add_children(s["name"])

    add_children(root["name"])

    # 4) terrain (장애물) 추가 - 캐릭터 크기에 비례
    scale = root_z / WALKER_TORSO_Z_REF
    _add_bumps(wb, terrain, scale)

    # 5) 모터 힘: 총 질량에 비례 (작은 캐릭터는 약하게, 큰 캐릭터는 강하게)
    # JSON 자체의 검증은 MuJoCo 없이도 쓸 수 있어야 하므로 무거운 의존성은 빌드 시점에만 불러옵니다.
    try:
        import mujoco
    except ImportError as exc:
        raise SpecError(
            "MuJoCo 를 불러올 수 없습니다. 먼저 setup.bat 을 실행해 실습 환경을 설치하세요. "
            f"({exc})"
        ) from exc
    try:
        tmp_model = mujoco.MjModel.from_xml_string(ET.tostring(mj, encoding="unicode"))
    except Exception as exc:
        raise SpecError(
            "검증을 통과했지만 MuJoCo 모델을 만들지 못했습니다. 아래 오류와 JSON 을 함께 확인하세요:\n"
            f"  {type(exc).__name__}: {exc}"
        ) from exc
    total_mass = float(tmp_model.body_mass.sum())
    gear = WALKER_GEAR_REF * (total_mass / WALKER_MASS_REF) * float(opts["strength"])
    gear = max(5.0, gear)

    act = ET.SubElement(mj, "actuator")
    for j in motor_joints:
        ET.SubElement(act, "motor", ctrllimited="true", ctrlrange="-1.0 1.0", gear=f"{gear:.1f}", joint=j)

    ET.indent(mj)
    xml = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(mj, encoding="unicode")

    size = max(highest, 0.6 * (max(xs) - min(xs)))
    meta = {
        "name": spec.get("name", "custom2d"),
        "terrain": terrain,
        "torso_z0": round(root_z, 4),                 # 처음 몸통 높이 (qpos[1] 의 초기값)
        "height": round(highest, 4),                  # 캐릭터 꼭대기 높이
        "total_mass": round(total_mass, 2),
        "gear": round(gear, 1),
        "frame_skip": int(opts["frame_skip"]),
        "segments": seg_order,                        # body 순서
        "joint_names": joint_names,                   # root 3개를 제외한 hinge joint 순서 (qpos[3:])
        "motor_joints": motor_joints,                 # action 순서
        "n_motors": len(motor_joints),
        "healthy_z_range": [round(float(opts["fall_height_ratio"]) * root_z, 4), round(4.0 * root_z + 1.0, 4)],
        "healthy_angle_range": [-math.radians(float(opts["fall_angle_deg"])), math.radians(float(opts["fall_angle_deg"]))],
        "camera": {"distance": round(min(max(2.0 * size, 1.8), 8.0), 2), "lookat_z": round(0.5 * highest, 3)},
        "warnings": warnings,
    }
    return xml, meta


def _add_bumps(worldbody: ET.Element, terrain: str, scale: float) -> None:
    if terrain not in TERRAIN_BUMPS:
        raise SpecError(f'terrain 은 {list(TERRAIN_BUMPS)} 중 하나여야 합니다: "{terrain}"')
    for i, (x, w, h) in enumerate(TERRAIN_BUMPS[terrain]):
        x, w, h = x * scale, w * scale, h * scale
        ET.SubElement(worldbody, "geom", name=f"bump_{i}", type="box", conaffinity="1", condim="3",
                      pos=_fmt(x, 0, h / 2), size=_fmt(w / 2, 2.0, h / 2), rgba="0.55 0.55 0.6 1",
                      friction="1.0 0.1 0.1")


# --------------------------------------------------------------------------------------
# 파일 단위 빌드
# --------------------------------------------------------------------------------------
def build_character(spec_path: str | Path, terrain: str = "flat", out_dir: str | Path | None = None) -> tuple[Path, dict]:
    """
    characters/<name>.json -> characters/<name>.xml + characters/<name>.meta.json
    terrain != "flat" 이면 characters/_generated/<name>__<terrain>.xml 로 저장합니다.
    반환: (xml 경로, meta)
    """
    spec_path = Path(spec_path)
    spec = load_spec(spec_path)
    xml, meta = spec_to_mjcf(spec, terrain=terrain)
    meta["source"] = spec_path.name
    name = spec_path.stem
    if terrain == "flat":
        out_dir = Path(out_dir) if out_dir else spec_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        xml_path = out_dir / f"{name}.xml"
        meta_path = out_dir / f"{name}.meta.json"
    else:
        out_dir = Path(out_dir) if out_dir else GENERATED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        xml_path = out_dir / f"{name}__{terrain}.xml"
        meta_path = out_dir / f"{name}__{terrain}.meta.json"
    xml_path.write_text(xml, encoding="utf-8")
    write_json(meta_path, meta)
    return xml_path, meta


def stock_walker_with_terrain(walker_xml: Path, meta: dict, terrain: str) -> Path:
    """기본 Walker2d XML(characters/walker.xml)에 terrain 을 추가한 복사본을 만들어 경로를 돌려줍니다."""
    if terrain == "flat":
        return walker_xml
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out = GENERATED_DIR / f"{walker_xml.stem}__{terrain}.xml"
    tree = ET.parse(walker_xml)
    wb = tree.getroot().find("worldbody")
    _add_bumps(wb, terrain, meta.get("torso_z0", WALKER_TORSO_Z_REF) / WALKER_TORSO_Z_REF)
    ET.indent(tree.getroot())
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def resolve_character(name_or_path: str, terrain: str = "flat") -> tuple[Path, dict]:
    """
    캐릭터 이름(예: "dog") 또는 json/xml 경로를 받아 (xml 경로, meta) 를 돌려줍니다.
    json 만 있고 xml 이 없으면 자동으로 빌드합니다.
    """
    p = Path(name_or_path)
    if p.suffix.lower() == ".json" and p.exists():
        spec_path = p
    elif p.suffix.lower() == ".xml" and p.exists():
        # 직접 만든 XML: meta 파일이 옆에 있어야 함
        meta_path = p.parent / (p.stem + ".meta.json")
        if not meta_path.exists():
            raise SpecError(f"{p.name} 옆에 {meta_path.name} 이 없습니다. JSON spec 으로 빌드한 캐릭터를 사용하세요.")
        meta = read_json(meta_path)
        if p.stem == "walker":
            return stock_walker_with_terrain(p, meta, terrain), meta
        return p, meta
    else:
        name = p.stem if p.suffix else str(name_or_path)
        spec_path = CHARACTERS_DIR / f"{name}.json"
        xml_path = CHARACTERS_DIR / f"{name}.xml"
        meta_path = CHARACTERS_DIR / f"{name}.meta.json"
        if name == "walker" and xml_path.exists() and meta_path.exists():
            meta = read_json(meta_path)
            return stock_walker_with_terrain(xml_path, meta, terrain), meta
        if not spec_path.exists():
            have = sorted(q.stem for q in CHARACTERS_DIR.glob("*.json"))
            raise SpecError(f'캐릭터 "{name}" 을(를) 찾을 수 없습니다. characters/ 폴더에 있는 것: {have}')

    name = spec_path.stem
    if terrain == "flat":
        xml_path = spec_path.parent / f"{name}.xml"
        meta_path = spec_path.parent / f"{name}.meta.json"
    else:
        xml_path = GENERATED_DIR / f"{name}__{terrain}.xml"
        meta_path = GENERATED_DIR / f"{name}__{terrain}.meta.json"
    # json 이 xml 보다 새로우면 다시 빌드
    if not xml_path.exists() or not meta_path.exists() or xml_path.stat().st_mtime < spec_path.stat().st_mtime:
        xml_path, meta = build_character(spec_path, terrain=terrain)
    else:
        meta = read_json(meta_path)
    return xml_path, meta
