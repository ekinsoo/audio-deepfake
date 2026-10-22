import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from audio_deepfake.manifests import ManifestRow

spec = importlib.util.spec_from_file_location("freeze_full_robustness", Path(__file__).resolve().parents[1] / "scripts/freeze_full_robustness.py")
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


class FreezeTests(unittest.TestCase):
    def fixture(self):
        bona = [ManifestRow("dev", "D", f"LA_D_{i}", "-", "bonafide", f"ASVspoof2019_LA_dev/flac/LA_D_{i}.flac") for i in range(40)]
        spoof = ManifestRow("dev", "D", "LA_D_100", "A01", "spoof", "ASVspoof2019_LA_dev/flac/LA_D_100.flac")
        rows = bona + [spoof]
        clean = {row.utterance_id: float(index) for index, row in enumerate(rows)}
        return rows, clean

    def test_threshold_is_calibration_bonafide_only_and_order_invariant(self):
        rows, clean = self.fixture()
        result = freeze.freeze_threshold(rows, clean)
        self.assertEqual(result["threshold"], 2)
        self.assertEqual(result["ascending_index"], 2)
        self.assertEqual(result["accepted_count"], 38)
        clean[rows[-1].utterance_id] = -999
        self.assertEqual(freeze.freeze_threshold(rows[::-1], clean), result)

    def test_ties_can_increase_acceptance_but_not_rejection(self):
        rows, clean = self.fixture()
        clean = dict.fromkeys(clean, 1.0)
        result = freeze.freeze_threshold(rows, clean)
        self.assertEqual(result["threshold"], 1)
        self.assertEqual(result["acceptance_fraction"], 1)

    def test_missing_or_nonfinite_score_refuses_freeze(self):
        rows, clean = self.fixture()
        for invalid in ({}, {**clean, rows[0].utterance_id: float("nan")}):
            with self.assertRaises((ValueError, KeyError)):
                freeze.freeze_threshold(rows, invalid)

    def test_existing_audit_or_partition_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, audit = root / "partition", root / "audit.json"
            audit.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "Preserve"):
                freeze.save_plan({}, [], [], {}, output, audit)
            self.assertEqual(audit.read_text(), "preserve")
            self.assertFalse(output.exists())
            audit.unlink()
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "Preserve"):
                freeze.save_plan({}, [], [], {}, output, audit)

    def test_serialized_partition_is_reproducible_and_public_paths_are_relative(self):
        rows, _ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, audit = root / "partition", root / "audit.json"
            contract = {"status": "frozen"}
            with patch.object(freeze, "ROOT", root):
                freeze.save_plan(contract, rows[:20], rows[20:], {"full_ids": [row.utterance_id for row in rows]}, output, audit)
            result = freeze.read(audit)
            self.assertEqual(result["local_partition_dir"], "partition")
            self.assertEqual(result["partition_file_sha256"]["calibration.csv"], freeze.full.canonical_hash(rows[:20]))
            self.assertEqual(result["partition_file_sha256"]["analysis.csv"], freeze.full.canonical_hash(rows[20:]))
            self.assertEqual(result["partition_file_sha256"]["partition.json"], freeze.full.digest(output / "partition.json"))
