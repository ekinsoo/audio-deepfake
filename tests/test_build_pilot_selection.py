"""Combined selection preserves frozen protocol fields and protects outputs."""

import csv
from dataclasses import asdict
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_deepfake.manifests import ManifestRow, write_manifest

SPEC = importlib.util.spec_from_file_location(
    "build_pilot_selection", Path(__file__).resolve().parents[1] / "scripts/build_pilot_selection.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)
import prepare_pilot_data as prepare


class BuildPilotSelectionTests(unittest.TestCase):
    def fixture(self, root, *, mixed_split=False, shared_speaker=False):
        expected, hashes = [], {}
        for split, prefix, count in (("train", "T", 2000), ("dev", "D", 1000)):
            rows = []
            for index in range(count):
                label = "bonafide" if index < count // 2 else "spoof"
                actual_split = "eval" if mixed_split and split == "dev" and index == 0 else split
                actual_prefix = "E" if actual_split == "eval" else prefix
                rows.append(ManifestRow(actual_split,
                    "LA_shared" if shared_speaker else f"LA_{prefix}_speaker",
                    f"LA_{actual_prefix}_{index}", "-" if label == "bonafide" else "A01", label,
                    f"ASVspoof2019_LA_{actual_split}/flac/LA_{actual_prefix}_{index}.flac"))
            path = root / f"pilot_{split}.csv"
            write_manifest(rows, path)
            hashes[path.name] = prepare.digest(path)
            expected.extend(asdict(row) for row in rows)
        return expected, hashes

    def test_canonical_export_preserves_all_fields_and_original_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected, hashes = self.fixture(root)
            destination = root / "nested/selection.csv"
            with patch.object(prepare, "MANIFEST_SHA256", hashes):
                report = build.build_selection(root, destination)
                before = destination.stat().st_mtime_ns
                second = build.build_selection(root, destination)
            self.assertNotIn(b"\r", destination.read_bytes())
            with destination.open(encoding="utf-8", newline="") as source:
                observed = list(csv.DictReader(source))
            self.assertEqual(observed, expected)
            self.assertEqual(report, second)
            self.assertEqual(before, destination.stat().st_mtime_ns)
            self.assertEqual(report["selection_sha256"], prepare.digest(destination))
            self.assertEqual(report["source_manifest_sha256"], hashes)

    def test_changed_source_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, hashes = self.fixture(root)
            with (root / "pilot_dev.csv").open("a") as target:
                target.write("\n")
            with patch.object(prepare, "MANIFEST_SHA256", hashes):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    build.build_selection(root, root / "selection.csv")
            self.assertFalse((root / "selection.csv").exists())

    def test_conflicting_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, hashes = self.fixture(root)
            destination = root / "selection.csv"
            destination.write_bytes(b"existing different content")
            with patch.object(prepare, "MANIFEST_SHA256", hashes):
                with self.assertRaisesRegex(ValueError, "different content"):
                    build.build_selection(root, destination)
            self.assertEqual(destination.read_bytes(), b"existing different content")

    def test_eval_row_is_rejected_even_with_matching_source_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, hashes = self.fixture(root, mixed_split=True)
            with patch.object(prepare, "MANIFEST_SHA256", hashes):
                with self.assertRaisesRegex(ValueError, "Expected fixed 1000 dev"):
                    build.build_selection(root, root / "selection.csv")

    def test_speaker_overlap_is_rejected_even_with_matching_source_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, hashes = self.fixture(root, shared_speaker=True)
            with patch.object(prepare, "MANIFEST_SHA256", hashes):
                with self.assertRaisesRegex(ValueError, "Speaker overlap"):
                    build.build_selection(root, root / "selection.csv")


if __name__ == "__main__":
    unittest.main()
