"""Metric input contracts plus optional parity checks with the external evaluator."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audio_deepfake.manifests import ManifestRow
from audio_deepfake.metrics import DEFAULT_REFERENCE, evaluate_eer, evaluate_scores


def row(utterance, label):
    return ManifestRow("dev", "speaker", utterance, "-" if label == "bonafide" else "A01", label, "")


class MetricValidationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row("b1", "bonafide"), row("b2", "bonafide"), row("s1", "spoof")]
        self.scores = [("b1", 0.9), ("b2", 0.7), ("s1", -0.2)]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.asv = Path(self.temp.name) / "asv.txt"
        self.asv.write_text("- target 1\n- nontarget 0\nA01 spoof 0.5\n")
        self.missing_reference = Path(self.temp.name) / "missing.py"

    def evaluate(self, rows=None, scores=None):
        return evaluate_scores(self.rows if rows is None else rows,
                               self.scores if scores is None else scores,
                               self.asv, self.missing_reference)

    def test_duplicate_missing_and_extra_cm_ids(self):
        cases = [(self.scores + [("b1", 0.4)], "Duplicate CM"),
                 (self.scores[:-1], "1 missing"),
                 (self.scores + [("extra", 0.3)], "1 extra"),
                 ([("wrong", 0.9)] + self.scores[1:], "1 missing, 1 extra")]
        for scores, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.evaluate(scores=scores)

    def test_invalid_cm_values_and_soft_score_requirement(self):
        for value in [np.nan, np.inf, "bad"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.evaluate(scores=[("b1", value)] + self.scores[1:])
        with self.assertRaisesRegex(ValueError, "three unique"):
            self.evaluate(scores=[("b1", 1), ("b2", 1), ("s1", 0)])

    def test_manifest_contracts(self):
        cases = [(self.rows + [self.rows[0]], "Duplicate manifest"),
                 (self.rows[:2], "both bonafide and spoof"),
                 ([row("b1", "unknown")] + self.rows[1:], "unknown labels"),
                 ([], "empty")]
        for rows, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.evaluate(rows=rows)

    def test_invalid_asv_rows_before_reference_loading(self):
        for text, message in [("- unknown 1\n", "Unknown ASV"),
                              ("- target nan\n", "Non-finite ASV"),
                              ("- target\n", "three fields"),
                              ("- target 1\n- nontarget 0\n", "ASV spoof"),
                              ("- target bad\n", "Invalid ASV")]:
            with self.subTest(message=message):
                self.asv.write_text(text)
                with self.assertRaisesRegex(ValueError, message):
                    self.evaluate()

    def test_missing_and_modified_reference_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "evaluator is missing"):
            self.evaluate()
        altered = Path(self.temp.name) / "altered.py"
        altered.write_text("raise RuntimeError('must not execute')\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            evaluate_scores(self.rows, self.scores, self.asv, altered)

    def test_eer_invalid_classes_without_reference(self):
        for bona, spoof in [([], [0]), ([1], [np.nan]), ([[1]], [0])]:
            with self.subTest(bona=bona, spoof=spoof):
                with self.assertRaises(ValueError):
                    evaluate_eer(bona, spoof, self.missing_reference)


@unittest.skipUnless(DEFAULT_REFERENCE.is_file(), "External official 2019 evaluator is not downloaded")
class OfficialMetricParityTests(unittest.TestCase):
    def test_reference_tie_behavior(self):
        result = evaluate_eer([0.0], [0.0])
        self.assertEqual(result["eer"], 1.0)
        self.assertEqual(result["threshold"], 0.0)

    def test_bundled_reference_scores_and_reordered_id_join(self):
        base = DEFAULT_REFERENCE.parent
        cm_path, asv_path = base / "scores/cm_dev.txt", base / "scores/asv_dev.txt"
        if not cm_path.is_file() or not asv_path.is_file():
            self.skipTest("External example score files are not downloaded")
        records = [line.split() for line in cm_path.read_text().splitlines() if line.strip()]
        # Historical bundled trials repeat IDs. Use unique fixture-row IDs here;
        # the untouched scores/labels still exercise the exact numeric reference.
        rows = [row(f"fixture_{index}", fields[2]) for index, fields in enumerate(records)]
        scores = [(f"fixture_{index}", float(fields[3])) for index, fields in enumerate(records)]
        result = evaluate_scores(rows, reversed(scores), asv_path)
        self.assertAlmostEqual(result["eer"], 0.046701084744563005, places=14)
        self.assertAlmostEqual(result["min_tDCF"], 0.1104983661284351, places=14)
        self.assertEqual(result["evaluation_scope"], "pilot")
        self.assertEqual(result["split"], "dev")
        self.assertEqual(result["asv_score_sha256"],
                         "43f8ed0380acc20cab28aca113ffd7813811132f092ceb5596c2cbd92f3101e6")
        # The output must be serializable and must not disclose machine paths.
        self.assertNotIn(str(base), json.dumps(result))

    def test_undefined_tdcf_rejected(self):
        rows = [row("b1", "bonafide"), row("b2", "bonafide"), row("s1", "spoof")]
        with tempfile.TemporaryDirectory() as directory:
            asv = Path(directory) / "asv.txt"
            # Every spoof is below the ASV threshold, making C2 zero.
            asv.write_text("- target 1\n- nontarget 0\nA01 spoof -2\n")
            with self.assertRaisesRegex(ValueError, "undefined"):
                evaluate_scores(rows, [("b1", 2), ("b2", 1), ("s1", 0)], asv)


if __name__ == "__main__":
    unittest.main()
