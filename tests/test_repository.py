import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_settings_load_and_required_paths_are_relative(self):
        settings = json.loads((ROOT / "settings.json").read_text(encoding="utf-8"))
        required = {
            "DATA_DIR",
            "IMAGE_TEST_DIR",
            "IMAGE_TRAIN_DIR",
            "LoRA_DIR",
            "LORA_ADAPTER",
            "OUTPUT_PREDICTv1",
            "OUTPUT_PREDICTv2",
            "OUTPUT_FINAL_SUBMIT",
            "QWEN_MODEL_PATH",
            "PREDICTOR_MODEL_PATH",
            "SEED",
        }
        self.assertTrue(required.issubset(settings), required - settings.keys())
        self.assertEqual(settings["SEED"], 42)

        for key in required - {"SEED"}:
            value = Path(settings[key])
            self.assertFalse(value.is_absolute(), f"{key} must be repository-relative")
            self.assertNotIn("..", value.parts, f"{key} must stay inside the repository")

    def test_python_sources_compile(self):
        sources = [
            ROOT / "setup_dirs.py",
            ROOT / "setup_model_downloders.py",
            *(ROOT / "code").glob("*.py"),
        ]
        for source in sources:
            with self.subTest(source=source.name):
                compile(source.read_text(encoding="utf-8"), str(source), "exec")

    def test_setup_dirs_creates_expected_layout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "setup_dirs.py")],
                cwd=temporary_directory,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = {
                "code",
                "data",
                "images/test",
                "images/train",
                "models",
                "outputs",
            }
            for relative_path in expected:
                self.assertTrue(
                    (Path(temporary_directory) / relative_path).is_dir(),
                    relative_path,
                )

    def test_public_artifact_boundaries_are_ignored(self):
        ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("/data/*", "/images/*", "/models/*", "/outputs/*"):
            self.assertIn(pattern, ignore_text)

        docker_ignore_text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        for pattern in ("data/*", "images/*", "models/*", "outputs/*"):
            self.assertIn(pattern, docker_ignore_text)

    def test_synthetic_sample_layout(self):
        with (ROOT / "data" / "train.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))

        self.assertEqual([row["ID"] for row in rows], ["demo1", "demo2"])
        self.assertTrue(all(row["label"] for row in rows))

        for item_id in ("demo1", "demo2"):
            image_path = ROOT / "images" / "train" / item_id / "1.png"
            self.assertTrue(image_path.is_file(), image_path)
            self.assertEqual(image_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
