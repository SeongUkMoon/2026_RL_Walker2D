from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import walker_rl.spec2mjcf as spec_module
from walker_rl.spec2mjcf import SpecError, load_spec, validate_spec


ROOT = Path(__file__).resolve().parents[1]


def valid_spec() -> dict:
    """경고 없이 통과하는 작은 2족 캐릭터."""
    return {
        "name": "test_bot",
        "segments": [
            {
                "name": "torso",
                "parent": None,
                "start": [-0.2, 0.8],
                "end": [0.2, 0.8],
                "radius": 0.1,
                "color": "blue",
            },
            {
                "name": "leg_r",
                "parent": "torso",
                "start": [0.1, 0.8],
                "end": [0.1, 0.4],
                "radius": 0.04,
                "joint_range": [-90, 90],
                "motor": True,
                "color": "blue",
            },
            {
                "name": "foot_r",
                "parent": "leg_r",
                "start": [0.1, 0.4],
                "end": [0.25, 0.1],
                "radius": 0.05,
                "joint_range": [-45, 45],
                "motor": True,
                "friction": 1.9,
                "color": "0.2 0.4 0.8 1",
            },
            {
                "name": "leg_l",
                "parent": "torso",
                "start": [-0.1, 0.8],
                "end": [-0.1, 0.4],
                "radius": 0.04,
                "joint_range": [-90, 90],
                "motor": True,
                "color": "red",
            },
            {
                "name": "foot_l",
                "parent": "leg_l",
                "start": [-0.1, 0.4],
                "end": [0.05, 0.1],
                "radius": 0.05,
                "joint_range": [-45, 45],
                "motor": True,
                "friction": 1.9,
                "color": "red",
            },
        ],
        "options": {
            "strength": 1.0,
            "fall_height_ratio": 0.55,
            "fall_angle_deg": 60,
            "frame_skip": 4,
            "timestep": 0.002,
        },
    }


class SpecValidationTests(unittest.TestCase):
    def assert_spec_error(self, spec: object, text: str | None = None) -> str:
        with self.assertRaises(SpecError) as caught:
            validate_spec(spec)  # type: ignore[arg-type]
        message = str(caught.exception)
        if text is not None:
            self.assertIn(text, message)
        return message

    def test_valid_spec_has_no_warnings(self):
        self.assertEqual(validate_spec(valid_spec()), [])

    def test_shipped_character_specs_validate(self):
        for name in ("biped", "dog", "my_character", "ostrich"):
            with self.subTest(name=name):
                spec = load_spec(ROOT / "characters" / f"{name}.json")
                validate_spec(spec)

    def test_validation_module_does_not_import_mujoco_at_module_load(self):
        self.assertNotIn("mujoco", vars(spec_module))

    def test_non_object_segment_is_reported_as_spec_error(self):
        spec = valid_spec()
        spec["segments"][1] = None
        self.assert_spec_error(spec, "객체")

    def test_ascii_lowercase_names_are_required(self):
        for bad_name in ("다리", "Leg", "leg-1", "1leg"):
            with self.subTest(name=bad_name):
                spec = valid_spec()
                spec["segments"][1]["name"] = bad_name
                self.assert_spec_error(spec, "name 은 영문 소문자로 시작")

    def test_numeric_fields_reject_bool_and_non_finite_values(self):
        cases = (
            ("radius", True),
            ("radius", float("nan")),
            ("friction", float("inf")),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                spec = valid_spec()
                spec["segments"][1][field] = value
                self.assert_spec_error(spec, field)

        spec = valid_spec()
        spec["segments"][1]["start"] = [True, 0.8]
        self.assert_spec_error(spec, "start")

        spec = valid_spec()
        spec["options"]["strength"] = True
        self.assert_spec_error(spec, "options.strength")

    def test_motor_must_be_json_boolean(self):
        spec = valid_spec()
        spec["segments"][1]["motor"] = "false"
        self.assert_spec_error(spec, "JSON boolean")

    def test_joint_range_must_be_finite_ordered_and_include_zero(self):
        for bad_range in ([10, 150], [-150, -10], [20, 20], [-200, 20], [False, 20]):
            with self.subTest(joint_range=bad_range):
                spec = valid_spec()
                spec["segments"][1]["joint_range"] = bad_range
                self.assert_spec_error(spec, "joint_range")

    def test_color_must_be_preset_or_bounded_rgba(self):
        for bad_color in ("orange", "1 0 0", "1.2 0 0 1", "nan 0 0 1", False):
            with self.subTest(color=bad_color):
                spec = valid_spec()
                spec["segments"][1]["color"] = bad_color
                self.assert_spec_error(spec, "color")

    def test_options_type_ranges_and_unknown_keys_are_checked(self):
        bad_options = (
            "strong",
            {"strength": 0},
            {"fall_height_ratio": 1.1},
            {"fall_angle_deg": 0},
            {"frame_skip": 4.0},
            {"timestep": 0},
            {"strenght": 1.0},
        )
        for options in bad_options:
            with self.subTest(options=options):
                spec = valid_spec()
                spec["options"] = options
                self.assert_spec_error(spec, "options")

    def test_parent_cycle_and_unknown_parent_are_spec_errors(self):
        spec = valid_spec()
        spec["segments"][1]["parent"] = "foot_r"
        spec["segments"][2]["parent"] = "leg_r"
        self.assert_spec_error(spec, "순환")

        spec = valid_spec()
        spec["segments"][1]["parent"] = "missing"
        self.assert_spec_error(spec, "라는 segment 가 없습니다")

    def test_motor_count_and_raised_foot_produce_warnings(self):
        spec = valid_spec()
        for segment in spec["segments"][2:]:
            segment["motor"] = False
        warnings = validate_spec(spec)
        self.assertTrue(any("모터가 1개" in warning for warning in warnings))

        spec = valid_spec()
        spec["segments"][2]["start"] = [0.1, 0.7]
        spec["segments"][2]["end"] = [0.25, 0.6]
        warnings = validate_spec(spec)
        self.assertTrue(any('발 후보 "foot_r"' in warning for warning in warnings))

    def test_comment_and_ai_uncertainty_fields_are_ignored(self):
        spec = valid_spec()
        spec["_ai_assumptions"] = ["다리 길이를 그림에서 추정함"]
        spec["_uncertain_parts"] = ["왼쪽 발 관절"]
        spec["segments"][0]["_comment"] = "몸통"
        self.assertEqual(validate_spec(spec), [])

    def test_unknown_fields_and_multiple_errors_are_aggregated(self):
        spec = valid_spec()
        spec["segments"][1]["motro"] = True
        spec["segments"][1]["motor"] = "yes"
        message = self.assert_spec_error(spec)
        self.assertIn("알 수 없는 필드", message)
        self.assertIn("JSON boolean", message)

    def test_load_spec_rejects_duplicate_keys_non_finite_and_non_object(self):
        payloads = (
            ('{"name":"a","name":"b","segments":[]}', "중복된 JSON key"),
            ('{"name":"a","segments":[],"value":NaN}', "유한해야"),
            ('[1, 2, 3]', "최상위 값은 객체"),
        )
        for payload, expected in payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "character.json"
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(SpecError) as caught:
                    load_spec(path)
                self.assertIn(expected, str(caught.exception))

    def test_character_schema_is_valid_json(self):
        schema_path = ROOT / "characters" / "character.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")

    def test_build_cli_validate_only_and_argument_guards(self):
        script = ROOT / "2_build_character.py"
        valid = subprocess.run(
            [sys.executable, "-X", "utf8", str(script), "--validate-only"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
        self.assertIn("[통과]", valid.stdout)

        cases = (
            (["--validate-only", "--no-video"], "not allowed with argument"),
            (["--validate-only", "--seconds", "NaN"], "유한한 숫자"),
            (["--validate-only", "--seconds", "0"], "0보다 큰"),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", str(script), *args],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
