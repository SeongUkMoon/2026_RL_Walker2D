import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from walker_rl import utils


class CharacterArtifactTests(unittest.TestCase):
    def test_snapshot_records_and_verifies_exact_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "animal.json"
            xml = root / "animal.xml"
            run = root / "run"
            source.write_text('{"name": "animal"}', encoding="utf-8")
            xml.write_text("<mujoco/>", encoding="utf-8")

            artifacts = utils.snapshot_character_artifacts(
                str(source), xml, {"joint_names": ["hip"]}, run
            )
            config = {"character_artifacts": artifacts}

            self.assertEqual(artifacts["xml"]["path"], "character_used.xml")
            self.assertEqual(len(artifacts["xml"]["sha256"]), 64)
            self.assertFalse(Path(artifacts["source_json"]["path"]).is_absolute())
            self.assertEqual(utils.resolve_run_artifact(run, config, "xml").name, "character_used.xml")
            self.assertEqual(utils.resolve_run_artifact(run, config, "meta").name, "character_used.meta.json")
            self.assertEqual(utils.resolve_run_artifact(run, config, "source_json").name, "character_source.json")
            self.assertEqual((run / "character_used.xml").read_text(encoding="utf-8"), "<mujoco/>")

            (run / "character_used.xml").write_text("<changed/>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "변경"):
                utils.resolve_run_artifact(run, config, "xml")

    def test_artifact_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            outside = root / "outside.xml"
            outside.write_text("<mujoco/>", encoding="utf-8")
            config = {"character_artifacts": {"xml": {"path": "../outside.xml"}}}
            with self.assertRaisesRegex(ValueError, "밖"):
                utils.resolve_run_artifact(run, config, "xml")


class CompletedRunTests(unittest.TestCase):
    @staticmethod
    def _make_run(root: Path, name: str, character: str = "dog", complete: bool = True) -> Path:
        run = root / name
        run.mkdir()
        (run / "config.json").write_text(json.dumps({"character": character}), encoding="utf-8")
        (run / "model.zip").write_bytes(b"model")
        if complete:
            (run / "vecnormalize.pkl").write_bytes(b"stats")
        return run

    def test_list_and_latest_ignore_incomplete_and_invalid_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self._make_run(root, "old", "dog")
            incomplete = self._make_run(root, "newer-but-incomplete", "dog", complete=False)
            latest = self._make_run(root, "latest", "dog")
            invalid = self._make_run(root, "invalid", "dog")
            (invalid / "config.json").write_text("not-json", encoding="utf-8")
            old.touch()
            latest.touch()
            # mtime 정렬이 파일 시스템 해상도에 좌우되지 않도록 명시합니다.
            import os
            os.utime(old, (1, 1))
            os.utime(incomplete, (2, 2))
            os.utime(latest, (3, 3))
            os.utime(invalid, (4, 4))

            with patch.object(utils, "RUNS_DIR", root):
                self.assertEqual(utils.list_runs("dog"), [old, latest])
                self.assertEqual(utils.latest_run("dog"), latest)
                self.assertEqual(utils.list_runs("cat"), [])


if __name__ == "__main__":
    unittest.main()
