"""Exercise the CLI boundary without downloading dataset audio."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ManifestCLITests(unittest.TestCase):
    def test_metadata_only_export_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, code, speaker in [("train", "T", "LA_0001"), ("dev", "D", "LA_0002")]:
                text = (
                    f"{speaker} LA_{code}_1000001 - - bonafide\n"
                    f"{speaker} LA_{code}_1000002 - A01 spoof\n"
                )
                (root / f"{split}.txt").write_text(text, encoding="utf-8")
            output = root / "out"
            command = [
                sys.executable, "-m", "audio_deepfake", "manifests",
                "--train-protocol", str(root / "train.txt"),
                "--dev-protocol", str(root / "dev.txt"),
                "--train-limit", "2", "--dev-limit", "2", "--output-dir", str(output),
            ]
            # Keep the child interpreter's temporary files in the same test workspace.
            env = dict(os.environ, TMP=directory, TEMP=directory)
            first = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(first.returncode, 0, first.stderr)
            summary = json.loads(first.stdout)
            self.assertEqual(summary["audio_validation"], "not_checked")
            self.assertFalse(summary["official_distribution_verified"])
            self.assertNotIn("eval", summary["pilot"]["splits"])
            self.assertEqual(len(summary["manifest_sha256"]), 4)
            self.assertNotIn(str(root), (output / "summary.json").read_text())
            before = (output / "train.csv").read_bytes()
            repeated = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)
            self.assertEqual((output / "train.csv").read_bytes(), before)
            (output / "eval.csv").write_text("previous eval inventory", encoding="utf-8")
            stale = subprocess.run(command + ["--overwrite"], capture_output=True, text=True, env=env)
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn("no eval protocol", stale.stderr)
            self.assertEqual((output / "train.csv").read_bytes(), before)

    def test_speaker_leakage_fails_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, code in [("train", "T"), ("dev", "D")]:
                (root / f"{split}.txt").write_text(
                    f"LA_0001 LA_{code}_1000001 - - bonafide\n"
                    f"LA_0001 LA_{code}_1000002 - A01 spoof\n", encoding="utf-8"
                )
            output = root / "out"
            result = subprocess.run([
                sys.executable, "-m", "audio_deepfake", "manifests",
                "--train-protocol", str(root / "train.txt"),
                "--dev-protocol", str(root / "dev.txt"),
                "--train-limit", "2", "--dev-limit", "2", "--output-dir", str(output),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
