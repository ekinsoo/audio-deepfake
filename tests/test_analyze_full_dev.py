import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("analyze_full_dev", Path(__file__).resolve().parents[1] / "scripts/analyze_full_dev.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class FullDevAnalysisTests(unittest.TestCase):
    def test_literal_boundary_treats_tied_bonafide_as_accepted_and_spoof_as_error(self):
        rows = [{"utterance_id": "b", "split": "dev", "label": "bonafide", "attack_id": "-", "score": "1"},
                {"utterance_id": "s", "split": "dev", "label": "spoof", "attack_id": "A01", "score": "1"}]
        result = analysis.literal_errors(rows, 1.0)
        self.assertEqual(result["threshold_ties"], 2)
        self.assertEqual(result["by_attack"]["-"]["errors"], 0)
        self.assertEqual(result["by_attack"]["A01"]["errors"], 1)
        self.assertEqual(result["total_errors"], 1)

    def test_rejects_nonfinite_and_invalid_split_attack_labels(self):
        good = {"utterance_id": "s", "split": "dev", "label": "spoof", "attack_id": "A01", "score": "0"}
        for key, value in (("score", "nan"), ("split", "eval"), ("label", "unknown"), ("attack_id", "-")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                analysis.literal_errors([{**good, key: value}], 1.0)
